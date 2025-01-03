#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from pprint import pprint
from subprocess import run
from types import SimpleNamespace
from warnings import warn

import duckdb
import matplotlib.pyplot as plt
import mmapped_df
import numba
import numpy as np
import numpy.typing as npt
import opentimspy
import pandas as pd
import tomllib
from cached_opentims import MmappedOpenTIMS
from cached_opentims.misc import expand_left_right_indices
from cached_opentims.neighborhood_ops import (
    get_ms1_spacing,
    get_triangular_1D_kernel,
    map3Dsubsets,
    multiply_marginals,
    update_count_maxIntensity_TIC,
    update_count_maxIntensity_TIC_kernelScore,
    visit_neighbors,
    write_neighbor,
)
from cached_opentims.stats import count2D, counts_to_cumsum_idx
from IPython import get_ipython
from mmapped_df import DatasetWriter, open_dataset_dct
from numba_progress import ProgressBar
from recapuccino.misc import in_ipython

# there should be a separate module for clustering.


warn("Development mode.")
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
pd.set_option("display.max_columns", None)


path = "tmp/datasets/memmapped/140/raw.d.cache"
dataset = MmappedOpenTIMS(path)

dataset.dataset_df
dataset.frame_scan_starts
dataset.frame_scan_counts
dataset.maxes


frames, scans, tofs, intensities, mzs, inv_ion_mobilities, retentions_times = [
    dataset.dataset_df[c].to_numpy()
    for c in (
        "frame",
        "scan",
        "tof",
        "intensity",
        "mz",
        "inv_ion_mobility",
        "retention_time",
    )
]

ms1frames = dataset.Frames.query("MsMsType == 0").Id.to_numpy()
ms1frame_spacing = get_ms1_spacing(ms1frames)
ms1frame_offset = frame_offset * ms1frame_spacing

cumsum_idx = counts_to_cumsum_idx(frame_scan_counts)  # numby it
frame_idxs = frame_scan_counts.sum(axis=1).cumsum()
ms1events_cnt = frame_scan_counts[ms1frames].sum()
ms1frames_cnt = len(ms1frames)
ms1_left_right_idxs = np.vstack([frame_idxs[ms1frames - 1], frame_idxs[ms1frames]]).T
ms1_event_idxs = expand_left_right_indices(ms1_left_right_idxs)
ms1frame_min = ms1frames[0]
ms1frame_max = ms1frames[-1]
scan_min = 0
scan_max = dimension_maxes["scan"]
tof_min = 0
tof_max = dimension_maxes["tof"]

kernel_marginals = [
    get_triangular_1D_kernel(offset)
    for offset in (frame_offset * ms1frame_spacing, scan_offset, tof_offset)
]
kernel = multiply_marginals(*kernel_marginals)

visit_neighbors = numba.njit(boundscheck=True)(visit_neighbors)
nb_update_count_maxIntensity_TIC_kernelScore = numba.njit(boundscheck=True)(
    update_count_maxIntensity_TIC_kernelScore
)
io = np.zeros(shape=(ms1events_cnt, 4), dtype=np.uint32)
with ProgressBar(
    total=ms1events_cnt, desc="Gathering neighborhood statistics"
) as progress:
    map3Dsubsets(  # provide all arguments without names: **kwargs not supported
        visit_neighbors,
        frames,
        scans,
        tofs,
        intensities,
        ms1_event_idxs,
        io,
        progress,
        # here start *args of visit_neighbors
        ms1frame_offset,
        ms1frame_spacing,
        scan_offset,
        tof_offset,
        tofs,
        cumsum_idx,
        # results_updater & *args
        nb_update_count_maxIntensity_TIC_kernelScore,  # just few seconds more to get the kernel-scores
        intensities,
        kernel,
    )
# np.save("io.npy", io)
# io = np.load("io.npy") # to avoid reruning the development.

io_df = pd.DataFrame(
    io, copy=False, columns=["neighbor_cnt", "max_intensity", "TIC", "kernel_score"]
)
# TODO: optimize: this copies RAM
io_df["idx"] = ms1_event_idxs
io_df["intensity"] = intensities[ms1_event_idxs]

conn = duckdb.connect()
neighbor_cnt_distr = conn.execute(
    """
SELECT neighbor_cnt, COUNT(*) AS cases
FROM 'io_df'
WHERE max_intensity == intensity
GROUP BY neighbor_cnt
ORDER BY neighbor_cnt DESC
"""
).df()
neighbor_cnt_distr["cum_cases"] = neighbor_cnt_distr.cases.cumsum()

plt.scatter(neighbor_cnt_distr.neighbor_cnt, neighbor_cnt_distr.cum_cases)
plt.yscale("log")
plt.xlabel("Neighbor Count")
plt.ylabel("Cumulated Cases")
plt.title(str(startrek_path))
if _development:
    plt.show()
else:
    plt.savefig(QC_folder / "neighbor_cnt_as_threshold.png")

tops = conn.execute(config["events_filter"].format(table="io_df")).df()

columns = ("frame", "scan", "tof")
for c in columns:
    tops[c] = events[c].to_numpy()[tops.idx.to_numpy()]

# doing calculations on a subset of events -> need reindexing
tops_frame_scan_counts = np.zeros(
    shape=frame_scan_counts.shape, dtype=frame_scan_counts.dtype
)
count2D(tops_frame_scan_counts, tops.frame.to_numpy(), tops.scan.to_numpy())
tops_cumsum_idx = counts_to_cumsum_idx(tops_frame_scan_counts)  # numby it


@numba.njit
def count_neighbors_and_max_neighborhood_score(
    event_io,
    idx,
    relative_frame_idx,
    relative_scan_idx,
    relative_tof_idx,
    scores,
    *args,
) -> None:
    event_io[0] += 1  # count
    score = scores[idx]
    event_io[1] = max(event_io[1], score)


tops_stats = np.zeros(shape=(len(tops), 2), dtype=np.uint32)

# tops: still lexicographically sorted and so amenable to the analysis below.
with ProgressBar(
    total=len(tops), desc="Finding points that max out the plateau score."
) as progress:
    map3Dsubsets(  # provide all arguments without names: **kwargs not supported
        visit_neighbors,
        tops.frame.to_numpy(),
        tops.scan.to_numpy(),
        tops.tof.to_numpy(),
        tops.intensity.to_numpy(),
        np.arange(len(tops)),  # optimize later
        tops_stats,
        progress,
        # here start *args of visit_neighbors
        ms1frame_offset,
        ms1frame_spacing,
        scan_offset,
        tof_offset,
        tops.tof.to_numpy(),  # this can be used ...
        tops_cumsum_idx,  # .. as here we updated the index.
        # results_updater & *args
        count_neighbors_and_max_neighborhood_score,
        tops.kernel_score.to_numpy(),
    )

tops["top_neighbors"] = tops_stats[:, 0]
tops["max_plateau_score"] = tops_stats[:, 1]

candidates = tops.query("kernel_score == max_plateau_score")
min_neighbor_cnt = config["min_neighbor_cnt"]

good_candidates = candidates.query("neighbor_cnt > @min_neighbor_cnt").copy()
good_candidates["right_idx"] = good_candidates.neighbor_cnt.cumsum()
good_candidates["left_idx"] = np.insert(good_candidates.right_idx.to_numpy()[:-1], 0, 0)
good_candidates["ClusterID"] = np.arange(1, len(good_candidates) + 1)

dataframe_scheme = pd.DataFrame(
    {
        c: pd.Series(dtype=t)
        for c, t in {"ClusterID": np.uint32, **dict(events.dtypes)}.items()
    }
)
DatasetWriter.preallocate_dataset(
    output_folder,
    dataframe_scheme,
    nrows=int(good_candidates.neighbor_cnt.sum()),
)
datasets = open_dataset_dct(output_folder, read_write=True)

io2 = good_candidates[["ClusterID", "left_idx"]].to_numpy(
    copy=True,
    dtype=np.uint32,
)
write_neighbor = numba.njit(write_neighbor)
with ProgressBar(total=len(good_candidates), desc="Writing down data") as progress:
    map3Dsubsets(  # provide all arguments without names: **kwargs not supported
        visit_neighbors,
        good_candidates.frame.to_numpy(),
        good_candidates.scan.to_numpy(),
        good_candidates.tof.to_numpy(),
        good_candidates.intensity.to_numpy(),
        np.arange(len(good_candidates)),  # optimize later
        io2,
        progress,
        # visit_neighbors' *args
        ms1frame_offset,
        ms1frame_spacing,
        scan_offset,
        tof_offset,
        tofs,  # this only when ..
        cumsum_idx,  # this is used...
        # results_updater & its *args
        write_neighbor,
        retentions_times,
        frames,
        inv_ion_mobilities,
        scans,
        mzs,
        tofs,
        intensities,
        datasets["ClusterID"],
        datasets["retention_time"],
        datasets["frame"],
        datasets["inv_ion_mobility"],
        datasets["scan"],
        datasets["mz"],
        datasets["tof"],
        datasets["intensity"],
    )

additional_cluster_stats_df = pd.DataFrame()
additional_cluster_stats_df.to_parquet(additional_cluster_stats)
