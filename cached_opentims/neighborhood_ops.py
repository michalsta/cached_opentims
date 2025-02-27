import functools
import typing

from numba_progress import ProgressBar

import numba
import numpy as np
import numpy.typing as npt
import opentimspy
import pandas as pd

from .stats import count2D, minmax


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


@numba.njit(parallel=True)
def funny_map(progress_proxy, N, foo, *foo_args):
    for idx in numba.prange(N):
        foo(idx, *foo_args)
        progress_proxy.update(1)


@numba.njit
def binary_search(sorted_list, value):
    """np.searchsorted equivalent. but a bit slower."""
    left = 0
    right = len(sorted_list)
    while left < right:
        mid = (left + right) // 2
        if sorted_list[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


@numba.njit
def binary_search2(sorted_list, left, right, value):
    """np.searchsorted equivalent. but a bit slower."""
    while left < right:
        mid = (left + right) // 2
        if sorted_list[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


@numba.njit(boundscheck=True)
def map_3D_box(IDX, XX, YY, ZZ, XY2IDX, X_tol, Y_tol, Z_tol, foo, *foo_args):
    """
    Iterate over all points around a 3D box represented sparsely by integer based coordinates.

    Box parametrization: [WW[IDX] - W_tol, WW[IDX] - W_tol] for W in [X,Y,Z].

    Arguments:
        IDX: the index of the current point.
        XX: all sparse x coordinates
        YY: all sparse y coordinates
        ZZ: all sparse z coordinates
        XY2IDX: (x,y) -> 1st occurence of (x,y) in the sparse format.
        X_tol: Tollerance in the x dim.
        Y_tol: Tollerance in the y dim.
        Z_tol: Tollerance in the z dim.
        foo: function to apply to each point in the box.
        *foo_args: other foo's arguments.
    """
    MIN_X = np.intp(0)  # casting to proper type to avoid the uint shannanigans...
    MAX_X = np.intp(XY2IDX.shape[0] - 1)
    MIN_Y = np.intp(0)
    MAX_Y = np.intp(XY2IDX.shape[1] - 2)
    X = np.intp(XX[IDX])
    Y = np.intp(YY[IDX])
    Z = np.intp(ZZ[IDX])
    for x in range(max(X - X_tol, MIN_X), min(X + X_tol, MAX_X) + 1):
        for y in range(max(Y - Y_tol, MIN_Y), min(Y + Y_tol, MAX_Y) + 1):
            idx_start = np.intp(XY2IDX[x, y])
            idx_end = np.intp(XY2IDX[x, y + 1])
            if idx_start < idx_end:
                idx_start += np.searchsorted(
                    ZZ[idx_start:idx_end], Z - Z_tol
                )  # get on right stripe.
                for idx in range(idx_start, idx_end):  # at most one look-up too far
                    z = np.intp(ZZ[idx])
                    if abs(z - Z) > Z_tol:
                        break
                    foo(IDX, idx, X, Y, Z, x, y, z, *foo_args)


@numba.njit(boundscheck=True)
def update_neighbor_intensity(
    IDX, idx, X, Y, Z, x, y, z, intensities, neighbor_intensity
):
    neighbor_intensity[IDX] += intensities[idx]


def extract_box_inefficiently(
    raw_data_handler, cycle, scan, tof, cycle_tol, scan_tol, tof_tol
):
    """Extract data-box inefficiently for testing purposes."""
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


@numba.njit(parallel=True, boundscheck=True)
def map_on_events(
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


# obsolete
@numba.njit(parallel=True)
def map3Dsubsets(
    foo,
    frames,
    scans,
    tofs,
    intensities,
    indices,
    io,
    progress_proxy,
    *args,
):
    for idx_in_results in numba.prange(len(indices)):
        idx = indices[idx_in_results]
        foo(
            io[idx_in_results],
            frames[idx],
            scans[idx],
            tofs[idx],
            intensities[idx],
            *args,
        )
        progress_proxy.update(1)


@numba.njit(parallel=True)
def map_onto_intervals_for_noncontiguous_sparse_data(
    foo,
    frames,
    scans,
    tofs,
    intensities,
    left_right_idxs,
    io,
    progress_proxy,
    *args,
):
    visited = 0
    for left_idx, right_idx in left_right_idxs:
        event_cnt = right_idx - left_idx
        for i in numba.prange(event_cnt):
            idx_in_sparse = i + left_idx
            middle_frame = frames[idx_in_sparse]
            middle_scan = scans[idx_in_sparse]
            middle_tof = tofs[idx_in_sparse]
            middle_intensity = intensities[idx_in_sparse]
            idx_in_results = i + visited
            foo(
                io[idx_in_results],
                middle_frame,
                middle_scan,
                middle_tof,
                middle_intensity,
                *args,
            )
            progress_proxy.update(1)
        visited += event_cnt


def extract_points(sparse, indices) -> npt.NDArray:
    point_cnt = 0
    for left_idx, right_idx in indices:
        for _ in range(left_idx, right_idx):
            point_cnt += 1
    points = np.empty(
        shape=(point_cnt, sparse.shape[1]),
        dtype=sparse.dtype,
    )
    i = 0
    for left_idx, right_idx in indices:
        for idx in range(left_idx, right_idx):
            points[i] = sparse[idx]
            i += 1
    return points


def extract_points_inefficiently(
    raw_data_handler: opentimspy.OpenTIMS,
    frames_to_chosen_frames: npt.NDArray,
    chosen_frames_to_frames: npt.NDArray,
    frame_start: int,
    frame_end: int,
    scan_start: int,
    scan_end: int,
    tof_start: int,
    tof_end: int,
    columns: list[str, ...] = ["frame", "scan", "tof", "intensity"],
) -> pd.DataFrame:
    frame_start = max(frame_start, 0)
    scan_start = max(scan_start, 0)
    frame_selection = pd.DataFrame(
        raw_data_handler[frames_to_chosen_frames[np.r_[frame_start:frame_end]]],
        columns=columns,
    )
    scan_selection = frame_selection.query("scan >= @scan_start and scan < @scan_end")
    tof_selection = scan_selection.query("tof >= @tof_start and tof < @tof_end")
    tof_selection_using_frames = tof_selection.copy()
    tof_selection_using_frames["frame"] = chosen_frames_to_frames[
        tof_selection_using_frames.frame
    ]
    return tof_selection_using_frames, tof_selection, scan_selection, frame_selection


def check_box_extractions(
    idx,
    cumsum_idx,
    sparse,
    tofs,
    raw_data_handler,
    frames_to_chosen_frames,
    chosen_frames_to_frames,
    frame_offset=2,
    scan_offset=10,
    tof_offset=2,
):
    """For debugging only."""
    frame, scan, tof, intensity = sparse[idx]
    frame_start = frame - frame_offset
    frame_end = frame + frame_offset + 1
    scan_start = scan - scan_offset
    scan_end = scan + scan_offset + 1
    tof_start = tof - tof_offset
    tof_end = tof + tof_offset + 1
    indices = np.zeros(
        shape=((frame_end - frame_start) * (scan_end - scan_start), 2), dtype=np.uint64
    )
    fill_indices_3D(
        indices,
        cumsum_idx,
        tofs,
        frame_start,
        frame_end,
        scan_start,
        scan_end,
        tof_start,
        tof_end,
    )
    Y = extract_points(sparse, indices)
    X, _, _, _ = extract_points_inefficiently(
        raw_data_handler,
        frames_to_chosen_frames,
        chosen_frames_to_frames,
        frame_start,
        frame_end,
        scan_start,
        scan_end,
        tof_start,
        tof_end,
    )
    assert np.all(X.to_numpy() == Y)
    return X.reset_index(drop=True), pd.DataFrame(Y, columns=X.columns)


def get_extract_neighbor_events_inefficiently(
    ms1_frame_spacing: int,
    frame_offset: int,
    scan_offset: int,
    tof_offset: int,
    raw_data_handler: opentimspy.OpenTIMS,
    _columns: list[str, ...] = ["frame", "scan", "tof", "intensity"],
) -> typing.Callable[[tuple[np.uint32, np.uint32, np.uint32, np.uint32]], pd.DataFrame]:
    def extract_neighbor_events_inefficiently(
        event: tuple[np.uint32, np.uint32, np.uint32, np.uint32],
    ) -> pd.DataFrame:
        middle_frame, middle_scan, middle_tof, middle_intensity = event
        min_frame = max(
            middle_frame - frame_offset * ms1_frame_spacing,
            raw_data_handler.ms1_frames[0],
        )
        max_frame = (
            min(
                middle_frame + frame_offset * ms1_frame_spacing,
                raw_data_handler.ms1_frames[-1],
            )
            + 1
        )
        frames = np.arange(min_frame, max_frame, ms1_frame_spacing)
        scan_start = middle_scan - scan_offset
        scan_end = middle_scan + scan_offset + 1
        tof_start = middle_tof - tof_offset
        tof_end = middle_tof + tof_offset + 1
        raw = pd.DataFrame(raw_data_handler[frames], columns=_columns, copy=False)
        return raw.query(
            "scan >= @scan_start and scan < @scan_end and tof >= @tof_start and tof < @tof_end"
        )

    return extract_neighbor_events_inefficiently


def get_triangular_1D_kernel(offset):
    N = offset * 2 + 1
    res = np.ones(shape=(N,), dtype=float)
    for i in range(1, offset + 1):
        res[offset - i] = res[offset + i] = 1 - i / (offset + 1)
    return res


def multiply_marginals(*marginals: npt.NDArray):
    return functools.reduce(np.multiply.outer, marginals)


# Do not ornate with numba.njit: leave it to the user
def visit_neighbors_frame_diffs(
    # required
    event_io: npt.NDArray,
    middle_frame,
    middle_scan,
    middle_tof,
    middle_intensity,
    # foo(..., *args)
    frame_diffs,
    scan_offset,
    tof_offset,
    tofs,
    cumsum_idx,
    # results_updater & its *args
    results_updater,
    *args,
):
    MAX_FRAME = cumsum_idx.shape[0]
    MAX_SCAN = cumsum_idx.shape[1]
    min_frame_diff = frame_diffs[0]
    min_scan_diff = -scan_offset
    for frame_diff in frame_diffs:
        if middle_frame >= -frame_diff:  # avoiding cycling with unsigned ints
            frame = middle_frame + frame_diff
            if frame < MAX_FRAME:
                for scan_diff in range(-scan_offset, scan_offset + 1):
                    if middle_scan >= -scan_diff:  # avoiding cycling with unsigned ints
                        scan = middle_scan + scan_diff
                        if scan < MAX_SCAN:
                            left_idx, right_idx = cumsum_idx[frame, scan]
                            if left_idx < right_idx:  # no events in (frame,scan) slice
                                tofs_to_seach_in = tofs[left_idx:right_idx]
                                tof_start = middle_tof - tof_offset
                                tof_end = middle_tof + tof_offset + 1
                                L = left_idx + np.searchsorted(
                                    tofs_to_seach_in, tof_start
                                )
                                R = left_idx + np.searchsorted(
                                    tofs_to_seach_in, tof_end
                                )
                                for idx in range(L, R):
                                    relative_frame_idx = frame_diff - min_frame_diff
                                    relative_scan_idx = scan_diff - min_scan_diff
                                    relative_tof_idx = tofs[idx] - tof_start
                                    results_updater(
                                        event_io,
                                        idx,
                                        relative_frame_idx,
                                        relative_scan_idx,
                                        relative_tof_idx,
                                        *args,
                                    )


# Do not ornate with numba.njit: leave it to the user
def visit_neighbors(
    # required
    event_io: npt.NDArray,
    middle_frame: np.uint32,
    middle_scan: np.uint32,
    middle_tof: np.uint32,
    middle_intensity: np.uint32,
    # foo(..., *args)
    frame_offset: int,
    frame_step: int,
    scan_offset: int,
    tof_offset: int,
    # where to look into
    tofs: npt.NDArray,
    cumsum_idx: npt.NDArray,
    # results_updater & its *args
    results_updater,
    *args,
):
    MAX_FRAME = cumsum_idx.shape[0]
    MAX_SCAN = cumsum_idx.shape[1]
    min_frame_diff = -frame_offset
    min_scan_diff = -scan_offset
    for frame_diff in range(-frame_offset, frame_offset + 1, frame_step):
        if middle_frame >= -frame_diff:  # avoiding cycling with unsigned ints
            frame = middle_frame + frame_diff
            if frame < MAX_FRAME:
                for scan_diff in range(-scan_offset, scan_offset + 1):
                    if middle_scan >= -scan_diff:  # avoiding cycling with unsigned ints
                        scan = middle_scan + scan_diff
                        if scan < MAX_SCAN:
                            left_idx, right_idx = cumsum_idx[frame, scan]
                            if left_idx < right_idx:  # no events in (frame,scan) slice
                                tofs_to_seach_in = tofs[left_idx:right_idx]
                                tof_start = middle_tof - tof_offset
                                tof_end = middle_tof + tof_offset + 1
                                L = left_idx + np.searchsorted(
                                    tofs_to_seach_in, tof_start
                                )
                                R = left_idx + np.searchsorted(
                                    tofs_to_seach_in, tof_end
                                )
                                for idx in range(L, R):
                                    relative_frame_idx = frame_diff - min_frame_diff
                                    relative_scan_idx = scan_diff - min_scan_diff
                                    relative_tof_idx = tofs[idx] - tof_start
                                    results_updater(
                                        event_io,
                                        idx,
                                        relative_frame_idx,
                                        relative_scan_idx,
                                        relative_tof_idx,
                                        *args,
                                    )


@numba.njit(parallel=True)
def ParallelMap(
    foo: numba.core.registry.CPUDispatcher,
    indices: npt.NDArray,
    progress_proxy: ProgressBar,
    *foo_args,
):
    for idx in numba.prange(len(indices)):
        foo(idx, *foo_args)
        progress_proxy.update(1)


# all this would be more elegant if events were bundled together in one struct
def visit_neighbors_using_sparse_diffs(
    idx: int,
    event_idxs: npt.NDArray,
    middle_frame: np.uint32,
    middle_scan: np.uint32,
    middle_tof: np.uint32,
    middle_intensity: np.uint32,
    sparse_diffs: npt.NDArray,
):
    event_id = event_idxs[idx]  # the real number of event in the sparse data tables.


def report_neighbor_idxs(
    left_frame: np.uint32,
    right_frame: np.uint32,
    left_scan: np.uint32,
    right_scan: np.uint32,
    left_tof: np.uint32,
    right_tof: np.uint32,
    # where to look into
    tofs: npt.NDArray,
    cumsum_idx: npt.NDArray,
    # results_updater & its *args
    frame_step: np.uint32,
) -> list[np.uint32]:
    res = []
    MAX_FRAME = cumsum_idx.shape[0]
    MAX_SCAN = cumsum_idx.shape[1]
    min_scan_diff = -scan_offset
    min_frame_diff = -frame_offset
    for frame in range(left_frame, right_frame, frame_step):
        if frame < MAX_FRAME:
            for scan in range(left_scan, right_scan):
                if scan < MAX_SCAN:
                    left_idx, right_idx = cumsum_idx[frame, scan]
                    if left_idx < right_idx:  # no events in (frame,scan) slice
                        tofs_to_seach_in = tofs[left_idx:right_idx]
                        L = left_idx + np.searchsorted(tofs_to_seach_in, left_tof)
                        R = left_idx + np.searchsorted(tofs_to_seach_in, right_tof)
                        for idx in range(L, R):
                            res.append(idx)
    return res


# Do not ornate with numba.njit: leave it to the user
def update_count_maxIntensity_TIC(
    event_io,
    idx,
    relative_frame_idx,
    relative_scan_idx,
    relative_tof_idx,
    intensities,
    *args,
) -> None:
    intensity = intensities[idx]
    event_io[0] += 1  # count
    event_io[1] = max(event_io[1], intensity)  # max
    event_io[2] += intensity  # TIC


# Do not ornate with numba.njit: leave it to the user
def update_count_maxIntensity_TIC_kernelScore(
    event_io,
    idx,
    relative_frame_idx,
    relative_scan_idx,
    relative_tof_idx,
    intensities,
    kernel,
    *args,
) -> None:
    intensity = intensities[idx]
    event_io[0] += 1  # count
    event_io[1] = max(event_io[1], intensity)  # max
    event_io[2] += intensity  # TIC
    weight = kernel[
        relative_frame_idx,
        relative_scan_idx,
        relative_tof_idx,
    ]
    event_io[3] += weight * intensity


def write_neighbor(
    event_io,
    idx,
    relative_frame_idx,
    relative_scan_idx,
    relative_tof_idx,
    # additional ones
    retentions_times,
    frames,
    inv_ion_mobilities,
    scans,
    mzs,
    tofs,
    intensities,
    ClusterIDs,
    clustered_retention_times,
    clustered_frames,
    clustered_inv_ion_mobilities,
    clustered_scans,
    clustered_mzs,
    clustered_tofs,
    clustered_intensities,
) -> None:
    cluster_ID = event_io[0]
    res_idx = event_io[1]
    event_io[1] += 1
    ClusterIDs[res_idx] = cluster_ID
    clustered_retention_times[res_idx] = retentions_times[idx]
    clustered_frames[res_idx] = frames[idx]
    clustered_inv_ion_mobilities[res_idx] = inv_ion_mobilities[idx]
    clustered_scans[res_idx] = scans[idx]
    clustered_mzs[res_idx] = mzs[idx]
    clustered_tofs[res_idx] = tofs[idx]
    clustered_intensities[res_idx] = intensities[idx]


def get_ms1_spacing(ms1frames: npt.NDArray) -> int:
    unique_diffs = np.unique(np.diff(ms1frames))
    assert len(unique_diffs) == 1, "ms1 frames not equally spaced."
    return int(unique_diffs[0])
