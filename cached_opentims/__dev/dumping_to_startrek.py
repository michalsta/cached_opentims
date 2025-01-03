from collections import Counter

from numba_progress import ProgressBar
from tqdm import tqdm as progressbar

import matplotlib.pyplot as plt
import mmapped_df
import numba
import numpy as np
import numpy.typing as npt
import opentimspy
import pandas as pd
from cached_opentims.misc import array_size_in_mb
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

path = "spectra/G8602.d"
_progressbar_message = "test"


_columns: list[str] = [
    "frame",
    "scan",
    "tof",
    "intensity",
    "mz",
    "inv_ion_mobility",
    "retention_time",
]

folder_startrek = "test.startrek"

# raw_data_handler = opentimspy.OpenTIMS(path)
# with mmapped_df.DatasetWriter(folder_startrek, overwrite_dir=True) as DW:
#     for frame in progressbar(
#         raw_data_handler.frames["Id"],
#         desc=_progressbar_message,
#     ):
#         raw_frame = pd.DataFrame(
#             raw_data_handler.query(frame, columns=_columns),
#             columns=_columns,
#             copy=False,
#         )
#         DW.append_df(raw_frame)
# # 2'03"
# df = mmapped_df.open_dataset(folder_startrek)


# _columns: list[str] = [
#     "frame",
#     "scan",
#     "tof",
#     "intensity",
# ]
# folder_startrek = "test2.startrek"

# raw_data_handler = opentimspy.OpenTIMS(path)
# with mmapped_df.DatasetWriter(folder_startrek, overwrite_dir=True) as DW:
#     for frame in progressbar(
#         raw_data_handler.frames["Id"],
#         desc=_progressbar_message,
#     ):
#         raw_frame = pd.DataFrame(
#             raw_data_handler.query(frame, columns=_columns),
#             columns=_columns,
#             copy=False,
#         )
#         DW.append_df(raw_frame)

# df = mmapped_df.open_dataset(folder_startrek)


# for frame in progressbar(
#     raw_data_handler.frames["Id"],
#     desc=_progressbar_message,
# ):
#     raw_frame = pd.DataFrame(
#         raw_data_handler.query(frame, columns=_columns),
#         columns=_columns,
#         copy=False,
#     )
# 24"
# folder_startrek = "test/test.startrek"

# df = mmapped_df.open_new_dataset_dct(
#     folder_startrek,
#     scheme=raw_frame,
#     nrows=len(raw_data_handler),
# )

# _columns: list[str] = [
#     "frame",
#     "scan",
#     "tof",
#     "intensity",
#     "mz",
#     "inv_ion_mobility",
#     "retention_time",
# ]


# i = 0
# for frame in progressbar(
#     raw_data_handler.frames["Id"],
#     desc=_progressbar_message,
# ):
#     raw_frame = raw_data_handler.query(frame, columns=_columns)
#     n = len(raw_frame["frame"])
#     for dim, vals in raw_frame.items():
#         df[dim][i : i + n] = vals
#     i += n

# len(raw_data_handler)


@numba.njit(boundscheck=True)
def update_count(xx, store):
    for x in xx:
        store[x] += 1


raw_data_handler = opentimspy.OpenTIMS(path)
scheme = pd.DataFrame(
    {
        "tof": pd.Series(dtype=np.uint32),
        "intensity": pd.Series(dtype=np.uint16),
    }
)
folder_startrek = "test.startrek"
df = mmapped_df.open_new_dataset_dct(
    folder_startrek,
    scheme=scheme,
    nrows=len(raw_data_handler),
)

counts = np.zeros(
    dtype=np.uint16,
    shape=(raw_data_handler.max_frame + 1, raw_data_handler.max_scan + 2),
)
i = 0
for frame in progressbar(
    raw_data_handler.frames["Id"],
    desc=_progressbar_message,
):
    raw_frame = raw_data_handler.query(frame, columns=["scan", "tof", "intensity"])
    n = len(raw_frame["intensity"])
    update_count(raw_frame["scan"], counts[frame])
    df["tof"][i : i + n] = raw_frame["tof"]
    df["intensity"][i : i + n] = raw_frame["intensity"]
    i += n


np.save("test_counts.npy", counts)

folder_startrek = "test.startrek"
counts = np.load("test_counts.npy")
df = mmapped_df.open_dataset_simple_namespace(folder_startrek)
tofs = df.tof
intensities = df.intensity


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


res = np.zeros(dtype=np.uint32, shape=tofs.shape)


@numba.njit
def sumrealgood(idx, middle_frame, middle_scan, middle_tof, results):
    results[idx] = middle_frame + middle_scan + middle_tof


frames, scans = counts.nonzero()
with ProgressBar(total=len(tofs), desc="test") as progress:
    map_on_all_points(progress, frames, scans, frame_scan_to_idx, sumrealgood, res)


# and now, using adapters.


@numba.njit
def sumrealgood2(middle_frame, middle_scan, middle_tof):
    return middle_frame + middle_scan + middle_tof


@numba.njit
def adapter(idx, middle_frame, middle_scan, middle_tof, foo, foo_results):
    foo_results[idx] = foo(middle_frame, middle_scan, middle_tof)


res2 = np.zeros(dtype=np.uint32, shape=tofs.shape)

frames, scans = counts.nonzero()
with ProgressBar(total=len(tofs), desc="test") as progress:
    map_on_all_points(
        progress, frames, scans, frame_scan_to_idx, adapter, sumrealgood2, res2
    )
# this above is experimental, but possible.


# OK, this is a problem, if the results are updated here, cause it might be different.


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


frame_diffs = 21 * np.arange(-2, 3)
scan_diffs = np.arange(-10, 11)
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


# perhaps it is possible to use function with data.
# OK, the above is not doing the right thing: it's not working on ms1 only.

res = Counter(interesting_event_counts)


Frames = pd.DataFrame(raw_data_handler.frames)
Frames


FramesMS1 = Frames.query("MsMsType == 0")

ids = FramesMS1.Id.to_numpy()
cnts = FramesMS1.NumPeaks.to_numpy()


def get_min_unsign_int_data_type(x):
    exps = [8, 16, 32, 64]
    exp = exps[np.searchsorted(exps, np.log2(x))]
    return eval(f"np.uint{exp}")


# So: we don't need to have it here, can very quickly make in RAM.

all_frames = np.empty(dtype=get_min_unsign_int_data_type(ids.max()), shape=cnts.sum())
# so the lesson is learned.


@numba.njit
def better_tile(results, ids, cnts):
    i = 0
    for _id, _cnt in zip(ids, cnts):
        for _ in range(_cnt):
            results[i] = _id
            i += 1


better_tile(all_frames, ids, cnts)


# how to turn scans into an array?
frames, scans = counts.nonzero()
ms1_mask = np.isin(frames, raw_data_handler.ms1_frames)

ms1frames = frames[ms1_mask]
ms1scans = scans[ms1_mask]

all_scans = np.empty(
    dtype=get_min_unsign_int_data_type(ms1scans.max()), shape=cnts.sum()
)
better_tile(all_scans, ms1scans, counts[ms1frames, ms1scans])


array_size_in_mb(all_scans)
array_size_in_mb(all_frames)

# now, should we not preselect intensities / tofs according to what we do?
# cool thing now: we exactly know how big the result should be. we don't need to dump frames and scans to SSD.
# and simpler map_on_all_points.


@numba.njit(parallel=True, boundscheck=True)
def map_on_all_points(
    progress_proxy,
    all_frames,
    all_scans,
    all_tofs,
    foo,
    *foo_args,
):
    assert len(all_frames) == len(all_tofs)
    assert len(all_frames) == len(all_scans)
    for idx in numba.prange(len(all_frames)):
        frame = all_frames[idx]
        scan = all_scans[idx]
        tof = all_tofs[idx]
        foo(idx, frame, scan, tof, *foo_args)
        progress_proxy.update(1)


# w sumie to po co ja to w ogóle dumpuje??? nie będę dumpował, dwie ścieżki.
# w RAMIE potrzebujemy tylko
ram_intensities = intensities.copy()

array_size_in_mb(ram_intensities)
# simply not do it!

raw_data_handler = opentimspy.OpenTIMS(path)
Frames = pd.DataFrame(
    {
        col: raw_data_handler.frames[col]
        for col in ["Id", "NumPeaks", "MsMsType", "MaxIntensity", "NumScans"]
    }
)
MS1Frames = Frames.query("MsMsType==0")

peaks_cnt = MS1Frames.NumPeaks.sum()
intensities = np.zeros(
    shape=peaks_cnt,
    dtype=get_min_unsign_int_data_type(MS1Frames.MaxIntensity.max()),
)
frames = np.zeros(
    shape=peaks_cnt,
    dtype=get_min_unsign_int_data_type(MS1Frames.Id.max()),
)
scans = np.zeros(
    shape=peaks_cnt,
    dtype=get_min_unsign_int_data_type(MS1Frames.NumScans.max()),
)
tofs = np.zeros(
    shape=peaks_cnt,
    dtype=np.uint32,
)

i = 0
for frame in tqdm(raw_data_handler.ms1_frames):
    Frame = raw_data_handler.query(frame, columns=["scan", "tof", "intensity"])
    n = len(Frame["intensity"])
    frames[i : i + n] = frame
    scans[i : i + n] = Frame["scan"]
    tofs[i : i + n] = Frame["tof"]
    intensities[i : i + n] = Frame["intensity"]
    i += n

list(map(array_size_in_mb, [frames, scans, tofs, intensities]))


@numba.njit
def count_events_per_frame_and_scan(frames, scans):
    counts = np.zeros(
        dtype=np.uint16,
        shape=(frames.max() + 1, scans.max() + 2),
    )
    for frame, scan in zip(frames, scans):
        counts[frame, scan] += 1
    return counts


frame_scan_to_count = count_events_per_frame_and_scan(frames, scans)
frame_scan_to_index = counts_to_index(frame_scan_to_count)

# now: still under 10GB on a big dataset.
# and much simpler to do things.


@numba.njit(parallel=True, boundscheck=True)
def map_on_all_points(
    progress_proxy,
    frames,
    scans,
    tofs,
    intensities,
    foo,
    *foo_args,
):
    assert len(frames) == len(scans)
    assert len(frames) == len(tofs)
    assert len(frames) == len(intensities)
    for i in numba.prange(len(frames)):
        frame = frames[i]
        scan = scans[i]
        tof = tofs[i]
        intensity = intensities[i]
        progress_proxy.update(1)
        foo(i, frame, scan, tof, intensity, *foo_args)


# now, each function know where to save things.
