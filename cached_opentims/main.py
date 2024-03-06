import sys
from pathlib import Path

from tqdm import tqdm

import mmapped_df
import numba
import numpy as np
import pandas as pd
from cached_opentims.io import create_and_open_cached_tdf
from opentimspy import OpenTIMS


@numba.njit()
def count(frames, min_scan, max_scan, counts):
    ret = np.uint64(0)
    for frame in frames:
        for scan in range(min_scan, max_scan + 1):
            ret += counts[frame, scan]
    return ret


# @numba.njit()
# def fill_data(frames, scan_min, scan_max, mz_min, starts, counts, src, T):
#     t_idx = 0
#     for frame in frames:
#         for scan in range(scan_min, scan_max + 1):
#             d_start = starts[frame, scan]
#             for d_idx in range(0, counts[frame, scan]):
#                 T[t_idx] = src[d_start + d_idx]
#                 t_idx += 1
#                 d_idx += 1


@numba.njit()
def fill_data(frames, scan_min, scan_max, starts, counts, src, T):
    t_idx = 0
    for frame in frames:
        for scan in range(scan_min, scan_max + 1):
            d_start = starts[frame, scan]
            for d_idx in range(0, counts[frame, scan]):
                T[t_idx] = src[d_start + d_idx]
                t_idx += 1


@numba.njit()
def mz_filtered_indices(frames, scan_min, scan_max, starts, counts, src, T):
    ret = np.empty(shape=count(frames, min_scan, max_scan, counts), dtype=uint64)
    t_idx = 0
    for frame in frames:
        for scan in range(scan_min, scan_max + 1):
            d_start = starts[frame, scan]
            for d_idx in range(0, counts[frame, scan]):
                if mz_min <= masses[d_start + d_idx] <= mz + max:
                    T[t_idx] = src[d_start + d_idx]
                    t_idx += 1
    return ret[:t_idx]


class CachedOpenTIMS:
    def __init__(self, path, tmpdir=None, dont_recalculate=True):
        path = Path(path)
        assert tmpdir is None
        cache_dir = Path(str(path) + ".cache")

        (
            self.backend,
            self.starts,
            self.counts,
            _maxes,
            _Frames,
        ) = create_and_open_cached_tdf(
            folder_startrek=cache_dir,
            folder_d=path,
            _progressbar_message="Uncompressing TIMS dataset into *.d.cache",
        )

    def query(self, frames, colnames=None):
        return box_query(frames, self.OT.min_scan, self.OT.max_scan, colnames)

    def box_query(self, frames, min_scan, max_scan, colnames):
        if not isinstance(frames, np.ndarray):
            frames = np.array(frames, dtype=np.uint64)
        arr_size = count(frames, min_scan, max_scan, self.counts)
        acc = {}
        for colname in self.backend if colnames is None else colnames:
            T = np.empty(shape=(arr_size,), dtype=self.backend.dtypes[colname])
            fill_data(
                frames,
                min_scan,
                max_scan,
                self.starts,
                self.counts,
                self.backend[colname].values,
                T,
            )
            acc[colname] = T
        return pd.DataFrame(acc, copy=False)


#    def mz_box_query(self, frames, min_scan, max_scan, mz_min, mz_max):
#        if not isinstance(frames, np.ndarray):
#            frames = np.array(frames, dtype=np.uint64)
#        indices =


if __name__ == "__main__":
    OT = CachedOpenTIMS("/mnt/storage/science/midia_rawdata/8027.d")
    print(OT.box_query([345, 645, 746], 100, 500))
