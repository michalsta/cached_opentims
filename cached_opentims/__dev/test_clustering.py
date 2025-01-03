from numba_progress import ProgressBar
from tqdm import tqdm as progressbar

import matplotlib.pyplot as plt
import mmapped_df
import numba
import numpy as np
import numpy.typing as npt
import opentimspy
import pandas as pd
from cached_opentims.neighborhood_ops import (
    get_ms1_spacing,
    get_triangular_1D_kernel,
    map3Dsubsets,
    multiply_marginals,
    update_count_maxIntensity_TIC,
    update_count_maxIntensity_TIC_kernelScore,
    write_neighbor,
)
from numpy.lib.stride_tricks import as_strided
from pandas_ops.io import read_df

folder_startrek = "test.startrek"
path = "spectra/G8602.d"

raw_data_handler = opentimspy.OpenTIMS(path)

counts = np.load("test_counts.npy")
df = mmapped_df.open_dataset_simple_namespace(folder_startrek)
tofs = df.tof
intensities = df.intensity


def counts_to_index(counts):
    """Change counts to index.

    DO NOT USE NUMBA ON IT: IT DOES NOT WORK.

    index[a,b,...,z] = the total number of entries before observing (a,b,...,z).
    Given python 0-based indexing = the first occurrence of (a,b,...,z).
    """
    cumsums = counts.cumsum()
    index = np.zeros(cumsums.shape, dtype=cumsums.dtype)
    index[1:] = cumsums[:-1]
    index = index.reshape(counts.shape)
    return index


frame_scan_to_idx = counts_to_index(counts)


@numba.njit(parallel=True, boundscheck=True)
def map_on_all_points(
    progress_proxy,
    frames,
    scans,
    frame_scan_to_idx,
    foo,
    *foo_args,
):
    for i in numba.prange(len(frames)):
        middle_frame = frames[i]
        middle_scan = scans[i]
        idx_start = frame_scan_to_idx[middle_frame, middle_scan]
        idx_end = frame_scan_to_idx[
            middle_frame, middle_scan + 1
        ]  # that why we have one scan more than necessary.
        for idx in numba.prange(idx_start, idx_end):
            middle_tof = tofs[idx]
            foo(idx, middle_frame, middle_scan, middle_tof, *foo_args)
            progress_proxy.update(1)
            # tofs_to_seach_in = tofs[left_idx:right_idx]


@numba.njit(boundscheck=True)
def visit_neighbors(
    idx: np.uint32,
    middle_frame: np.uint32,
    middle_scan: np.uint32,
    middle_tof: np.uint32,
    # *foo_args of `map_on_all_points`
    frame_scan_to_idx: npt.NDArray,
    frame_diffs: npt.NDArray,
    scan_diffs: npt.NDArray,
    tofs: npt.NDArray,
    min_tof_diff: np.uint32,
    max_tof_diff: np.uint32,
    foo: numba.core.registry.CPUDispatcher,
    *foo_args,
):
    MAX_FRAME = frame_scan_to_idx.shape[0] - 1
    MAX_SCAN = frame_scan_to_idx.shape[1] - 2
    assert (
        min_tof_diff <= max_tof_diff
    ), "Min tof difference must be smaller than max tof difference."
    for frame_diff in frame_diffs:
        if middle_frame >= -frame_diff:  # be above 0, avoid uint pitfalls
            frame = middle_frame + frame_diff
            if frame <= MAX_FRAME:
                for scan_diff in scan_diffs:
                    if middle_scan >= -scan_diff:  # be above 0, avoid uint pitfalls
                        scan = middle_scan + scan_diff
                        if scan <= MAX_SCAN:
                            stencil_idx_start = frame_scan_to_idx[frame, scan]
                            stencil_idx_end = frame_scan_to_idx[frame, scan + 1]
                            if stencil_idx_start < stencil_idx_end:
                                tofs_to_seach_in = tofs[
                                    stencil_idx_start:stencil_idx_end
                                ]
                                tof_start = middle_tof + min_tof_diff
                                tof_end = middle_tof + max_tof_diff + 1
                                stencil_idx_left = stencil_idx_start + np.searchsorted(
                                    tofs_to_seach_in, tof_start
                                )
                                stencil_idx_right = stencil_idx_start + np.searchsorted(
                                    tofs_to_seach_in, tof_end
                                )
                                for stencil_idx in range(
                                    stencil_idx_left, stencil_idx_right
                                ):
                                    tof = tofs[stencil_idx]
                                    foo(
                                        idx,
                                        stencil_idx,
                                        middle_frame,
                                        middle_scan,
                                        middle_tof,
                                        frame,
                                        scan,
                                        tof,
                                        *foo_args,
                                    )


@numba.njit(boundscheck=True)
def count_points(
    idx,
    stencil_idx,
    middle_frame,
    middle_scan,
    middle_tof,
    frame,
    scan,
    tof,
    results,
    intensities,
    *args,
):
    results[idx] += intensities[stencil_idx]


frame_diffs = 21 * np.arange(-1, 2)
scan_diffs = np.arange(-2, 3)
min_tof_diff = -2
max_tof_diff = 2


# this is way too long.
event_counts = np.zeros(dtype=np.uint32, shape=tofs.shape)

frames, scans = counts.nonzero()
ms1_mask = np.isin(frames, raw_data_handler.ms1_frames)
frames = frames[ms1_mask]
scans = scans[ms1_mask]

MS1_events_cnt = (
    pd.DataFrame(raw_data_handler.frames).query("MsMsType==0").NumPeaks.sum()
)

with ProgressBar(total=MS1_events_cnt, desc="test") as progress:
    map_on_all_points(
        progress,
        frames,
        scans,
        frame_scan_to_idx,
        visit_neighbors,
        frame_scan_to_idx,
        frame_diffs,
        scan_diffs,
        tofs,
        min_tof_diff,
        max_tof_diff,
        count_points,
        event_counts,
        intensities,
    )
