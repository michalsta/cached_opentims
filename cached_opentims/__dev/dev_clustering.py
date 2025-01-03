import collections
from collections import Counter
from subprocess import run

from IPython import get_ipython
from numba_progress import ProgressBar
from tqdm import tqdm

import matplotlib.pyplot as plt
import mmapped_df
import numba
import numpy as np
import numpy.typing as npt
import opentimspy
import pandas as pd
from cached_opentims.misc import get_min_unsign_int_data_type
from cached_opentims.neighborhood_ops import (counts_to_index, funny_map,
                                              map_3D_box)
from cached_opentims.stats import count2D, minmax
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
scan_tol = 3
tof_tol = 3


@numba.njit(boundscheck=True)
def update_count(IDX, idx, X, Y, Z, x, y, z, intensities, results):
    results[IDX] += intensities[idx]


def discrete_histogram(x, value_name="value"):
    """Summarize discrete data."""
    return (
        pd.DataFrame(collections.Counter(x).items(), columns=(value_name, "cnt"))
        .sort_values(value_name)
        .set_index(value_name)
    )


neighbors = np.zeros(dtype=np.uint32, shape=peaks_cnt)

with ProgressBar(total=peaks_cnt, desc="test") as progress_proxy:
    funny_map(
        progress_proxy,
        peaks_cnt,
        map_3D_box2,
        df.cycle,
        df.scan,
        df.tof,
        cycle_scan_to_index,
        cycle_tol,
        scan_tol,
        tof_tol,
        update_count,  # TODO: implement
        df.intensity,
        neighbors,
    )

minmax(neighbors)


@numba.njit
def numba_print(IDX, idx, X, Y, Z, x, y, z, *args):
    print(
        "IDX =",
        IDX,
        "idx =",
        idx,
        "X =",
        X,
        "Y =",
        Y,
        "Z =",
        Z,
        "x =",
        x,
        "y =",
        y,
        "z =",
        z,
        "args = ",
        args,
    )


@numba.njit
def binary_search(sorted_list, left, right, value):
    """np.searchsorted equivalent. but a bit slower."""
    while left < right:
        mid = (left + right) // 2
        if sorted_list[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left



@numba.njit(boundscheck=True)
def map_3D_box2(IDX, XX, YY, ZZ, XY2IDX, X_tol, Y_tol, Z_tol, foo, *foo_args):
    MIN_X = np.int64(0)
    MAX_X = np.int64(XY2IDX.shape[0] - 1)
    MIN_Y = np.int64(0)
    MAX_Y = np.int64(XY2IDX.shape[1] - 2)
    X = np.int64(XX[IDX])
    Y = np.int64(YY[IDX])
    Z = np.int64(ZZ[IDX])
    for x in range(max(X - X_tol, MIN_X), min(X + X_tol, MAX_X) + 1):
        for y in range(max(Y - Y_tol, MIN_Y), min(Y + Y_tol, MAX_Y) + 1):
            idx_start = np.int64(XY2IDX[x, y])
            idx_end = np.int64(XY2IDX[x, y + 1])
            if idx_start < idx_end:
                idx_start += np.searchsorted(
                    ZZ[idx_start:idx_end], Z - Z_tol
                )  # get on right stripe.
                for idx in range(idx_start, idx_end): # at most one look-up too far
                    z = np.int64(ZZ[idx])
                    if abs(z - Z) > Z_tol:
                        break
                    foo(IDX, idx, X, Y, Z, x, y, z, *foo_args)


map_3D_box2(IDX, XX, YY, ZZ, XY2IDX, X_tol, Y_tol, Z_tol, foo)

IDX = 1
XX = df.cycle
YY = df.scan
ZZ = df.tof
XY2IDX = cycle_scan_to_index
X_tol = cycle_tol
Y_tol = scan_tol
Z_tol = tof_tol
foo = numba_print

map_3D_box(IDX, XX, YY, ZZ, XY2IDX, X_tol, Y_tol, Z_tol, foo)


N = 1
with ProgressBar(total=N, desc=None) as progress_proxy:
    funny_map(
        progress_proxy,
        N,
        map_3D_box2,
        df.cycle,
        df.scan,
        df.tof,
        cycle_scan_to_index,
        cycle_tol,
        scan_tol,
        tof_tol,
        numba_print,
    )

with ProgressBar(total=N, desc=None) as progress_proxy:
    funny_map(
        progress_proxy,
        N,
        map_3D_box,
        df.cycle,
        df.scan,
        df.tof,
        cycle_scan_to_index,
        cycle_tol,
        scan_tol,
        tof_tol,
        numba_print,
    )

N = 10_000_000

%%timeit
neighbors4 = np.zeros(dtype=np.uint32, shape=N)
with ProgressBar(total=N, desc="test") as progress_proxy:
    funny_map(
        progress_proxy,
        N,
        map_3D_box3,
        df.cycle,
        df.scan,
        df.tof,
        cycle_scan_to_index,
        cycle_tol,
        scan_tol,
        tof_tol,
        update_count,  # TODO: implement
        df.intensity,
        neighbors4,
    )


%%timeit
neighbors2 = np.zeros(dtype=np.uint32, shape=N)
with ProgressBar(total=N, desc="test") as progress_proxy:
    funny_map(
        progress_proxy,
        N,
        map_3D_box,
        df.cycle,
        df.scan,
        df.tof,
        cycle_scan_to_index,
        cycle_tol,
        scan_tol,
        tof_tol,
        update_count,  # TODO: implement
        df.intensity,
        neighbors2,
    )

%%timeit
neighbors3 = np.zeros(dtype=np.uint32, shape=N)
with ProgressBar(total=N, desc="test") as progress_proxy:
    funny_map(
        progress_proxy,
        N,
        map_3D_box2,
        df.cycle,
        df.scan,
        df.tof,
        cycle_scan_to_index,
        cycle_tol,
        scan_tol,
        tof_tol,
        update_count,  # TODO: implement
        df.intensity,
        neighbors3,
    )

np.all(neighbors2 == neighbors3)


def extract_box_inefficiently(
    raw_data_handler, cycle, scan, tof, cycle_tol, scan_tol, tof_tol
):
    min_cycle = 0
    if cycle > cycle_tol:
        min_cycle = cycle - cycle_tol
    max_cycle = cycle + cycle_tol
    frames = raw_data_handler.ms1_frames[min_cycle : max_cycle + 1]

    datasets = []
    for _cycle in range(min_cycle, max_cycle + 1):
        X = pd.DataFrame(
            raw_data_handler.query(
                frames=raw_data_handler.ms1_frames[_cycle],
                columns=["scan", "tof", "intensity"],
            )
        )
        X["cycle"] = _cycle
        datasets.append(X)

    data = pd.concat(datasets).sort_values(["cycle", "scan", "tof"])

    min_scan = 0
    if scan > scan_tol:
        min_scan = scan - scan_tol
    max_scan = scan + scan_tol

    min_tof = 0
    if tof > tof_tol:
        min_tof = tof - tof_tol
    max_tof = tof + tof_tol
    return data.query(
        f"scan >= @min_scan and scan <= @max_scan and tof >= @min_tof and tof <= @max_tof"
    )


extract_box_inefficiently(
    raw_data_handler, cycle, scan, tof, cycle_tol, scan_tol, tof_tol
)


@numba.njit
def countevents(IDX, idx, X, Y, Z, x, y, z, res):
    res[0] += 1


IDX = 10
cycle = df.cycle[IDX]
scan = df.scan[IDX]
tof = df.tof[IDX]

res = np.array([0])
map_3D_box(IDX, XX, YY, ZZ, XY2IDX, 2, 500, 1000, countevents, res)
res2 = np.array([0])
map_3D_box2(IDX, XX, YY, ZZ, XY2IDX, 2, 500, 1000, countevents, res2)

extract_box_inefficiently(raw_data_handler, cycle, scan, tof, 2, 500, 1000)
res
res2
# OK, map3D_box works, map3D_box2 not
