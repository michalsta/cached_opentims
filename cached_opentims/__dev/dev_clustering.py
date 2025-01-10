import collections
from collections import Counter
from subprocess import run

from IPython import get_ipython
from numba_progress import ProgressBar
from tqdm import tqdm

import clusterMS.plotting
import duckdb
import matplotlib.pyplot as plt
import mmapped_df
import numba
import numpy as np
import numpy.typing as npt
import opentimspy
import pandas as pd
from cached_opentims.misc import get_min_unsign_int_data_type
from cached_opentims.neighborhood_ops import (
    counts_to_index,
    extract_box_inefficiently,
    funny_map,
    map_3D_box,
)
from cached_opentims.stats import count2D, minmax
from clusterMS.plotting import scat3D
from kilograms.histogramming import discrete_histogram1D, min_max, scatterplot_matrix
from matplotlib.colors import Normalize
from numpy.lib.stride_tricks import as_strided
from pandas_ops.io import read_df
from snakemaketools.datastructures import DotDict

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
pd.set_option("display.max_columns", None)

path = "spectra/G8602.d"


raw_data_handler = opentimspy.OpenTIMS(path)


def get_data_for_clustering_precursors(
    raw_data_handler, use_frames=False, **tqdm_kwargs
):
    Frames = pd.DataFrame(
        {
            col: raw_data_handler.frames[col]
            for col in ["Id", "NumPeaks", "MsMsType", "MaxIntensity", "NumScans"]
        }
    )
    MS1Frames = Frames.query("MsMsType==0")

    peaks_cnt = MS1Frames.NumPeaks.sum()

    maxes = DotDict()
    maxes.intensity = MS1Frames.MaxIntensity.max()
    maxes.scan = MS1Frames.NumScans.max()
    if use_frames:
        maxes.frame = MS1Frames.Id.max()
    else:
        maxes.cycle = len(MS1Frames)

    dtypes = {_col: get_min_unsign_int_data_type(_max) for _col, _max in maxes.items()}
    dtypes["tof"] = np.uint32
    df = DotDict()
    for _col, _dtype in dtypes.items():
        df[_col] = np.zeros(dtype=_dtype, shape=peaks_cnt)

    i = 0
    cycle = 0
    for frame in tqdm(raw_data_handler.ms1_frames, **tqdm_kwargs):
        Frame = raw_data_handler.query(frame, columns=["scan", "tof", "intensity"])
        n = len(Frame["intensity"])
        if use_frames:
            df.frame[i : i + n] = frame
        else:
            df.cycle[i : i + n] = cycle
        for col in ["scan", "tof", "intensity"]:
            df[col][i : i + n] = Frame[col]
        i += n
        cycle += 1

    return MS1Frames, df, maxes, peaks_cnt


MS1Frames, df, maxes, peaks_cnt = get_data_for_clustering_precursors(
    raw_data_handler, use_frames=False, desc="Getting MS1 events"
)

counts = np.zeros(dtype=np.uint32, shape=(maxes.cycle + 1, maxes.scan + 2))
count2D(counts, df.cycle, df.scan)

cycle_scan_to_index = counts_to_index(counts)


cycle_tol = 2
scan_tol = 5
tof_tol = 10


@numba.njit(boundscheck=True)
def get_event_stats(
    IDX,
    idx,
    X,
    Y,
    Z,
    x,
    y,
    z,
    intensities,
    nb_cnts,
    nb_intensities,
    nb_top_intensities,
    nb_top_intense_idxs,
    # correlation,
):
    I = intensities[idx]
    nb_intensities[IDX] += I
    nb_cnts[IDX] += 1
    if nb_top_intensities[IDX] < I:
        nb_top_intensities[IDX] = I
        nb_top_intense_idxs[IDX] = idx


max_nb_cnt = (cycle_tol * 2 + 1) * (scan_tol * 2 + 1) * (tof_tol * 2 + 1)


nb = pd.DataFrame(
    dict(
        cnts=np.zeros(dtype=get_min_unsign_int_data_type(max_nb_cnt), shape=peaks_cnt),
        intensities=np.zeros(dtype=np.uint32, shape=peaks_cnt),
        top_intensities=np.zeros(
            dtype=get_min_unsign_int_data_type(maxes.intensity), shape=peaks_cnt
        ),
        top_intense_idxs=np.zeros(dtype=np.uint32, shape=peaks_cnt),
    ),
    copy=False,
)

with ProgressBar(total=peaks_cnt, desc="test") as progress_proxy:
    funny_map(
        progress_proxy,
        peaks_cnt,
        map_3D_box,
        df.cycle,
        df.scan,
        df.tof,
        cycle_scan_to_index,
        cycle_tol,
        scan_tol,
        tof_tol,
        get_event_stats,  # TODO: implement
        df.intensity,
        nb.cnts.to_numpy(),
        nb.intensities.to_numpy(),
        nb.top_intensities.to_numpy(),
        nb.top_intense_idxs.to_numpy(),
    )


cnts_hist = discrete_histogram1D(nb.cnts.to_numpy())
max_nb_cnt
(cnts_hist.cumsum() / len(nb) * 100).round(1)

nb["intensity"] = df.intensity


# how many points are not local max?

duckdb.query(
    """
SELECT
COUNT(*) AS cnt
FROM 'nb'
WHERE top_intensities == intensity
"""
)

top_hist = duckdb.query(
    """
SELECT
cnts AS events,
COUNT(*) AS cnt
FROM 'nb'
WHERE top_intensities == intensity
GROUP BY cnts
ORDER BY cnts
"""
).df()

plt.plot(top_hist.events, top_hist.cnt)
plt.show()
# we should find a good shape to `correlate to`.


@numba.njit
def non_local_max(top_intense_idxs):
    cnt = 0
    for i, top_intense_idx in enumerate(top_intense_idxs):
        cnt += i == top_intense_idx
    return cnt


non_local_max(nb.top_intense_idxs.to_numpy())
len(nb)

dfpd = pd.DataFrame(df, copy=False)
_, S, C, T = dfpd.iloc[nb.intensity.argmax()]


def get_events_duckdb(events, cycle, scan, tof, cycle_tol, scan_tol, tof_tol, **kwargs):
    return duckdb.query(
        f"""
    SELECT *, sqrt(intensity) AS sqrt_intensity
    FROM 'dfpd' 
    WHERE cycle >= {cycle}-{cycle_tol} and cycle <= {cycle}+{cycle_tol} and scan >= {scan} - {scan_tol} AND scan <= {scan} + {scan_tol} and tof >= {tof} - {tof_tol} and tof <= {tof} + {tof_tol}
    """
    ).df()


some_events = get_events_duckdb(
    dfpd,
    **dict(dfpd.iloc[nb.intensity.argmax()]),
    scan_tol=scan_tol * 10,
    tof_tol=tof_tol,
    cycle_tol=cycle_tol * 10,
)


clusterMS.plotting.df_to_scatterplot3D(
    some_events[["cycle", "scan", "tof"]],
    c=some_events.sqrt_intensity,
    cmap="viridis",
    alpha=0.8,
)

# code discrete scatterplot matrix.
some_events["mz"] = raw_data_handler.tof_to_mz(
    tof=some_events.tof.to_numpy(), frame=(some_events.cycle.to_numpy() + 1) * 21
)

cols_to_show = ["cycle", "scan", "tof", "mz"]
extents = {
    col: minmax(vals.to_numpy()) for col, vals in some_events[cols_to_show].items()
}
bins = {col: _max - _min for col, (_min, _max) in extents.items()}
bins["mz"] = bins["tof"]

scatterplot_matrix(
    some_events[cols_to_show], weights=some_events.intensity, extents=extents, bins=bins
)


def discrete_histogram1D(xx, weights=None):
    _min, _max = min_max(xx)
    hist = np.zeros(dtype=np.uintp, shape=_max - _min + 1)
    if weights != None:
        assert len(weights) == len(xx)
    for i, x in enumerate(xx):
        hist[x - _min] += weights[i] if weights != None else 1
    return np.arange(_min, _max + 1), hist


discrete_histogram1D(some_events.tof.to_numpy(), some_events.intensity.to_numpy())
