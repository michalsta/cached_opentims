import sys
from pathlib import Path
from warnings import warn

import duckdb
import mmapped_df
import numba
import numpy as np
import pandas as pd
from cached_opentims.io import create_and_open_cached_tdf
from opentimspy import OpenTIMS
from tqdm import tqdm


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
    def __init__(self, path, cache_dir=None, dont_recalculate=True):
        path = Path(path)
        if cache_dir is None:
            cache_dir = Path(str(path) + ".cache")
        else:
            cache_dir = Path(cache_dir)

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


class MmappedOpenTIMS:
    def __init__(self, folder_startrek: Path | str, **kwargs):
        folder_startrek = Path(folder_startrek)
        if folder_startrek.suffix != ".startrek":
            warn("Expected .startrek extension.")

        (
            self.dataset_df,
            self.frame_scan_starts,
            self.frame_scan_counts,
            self.maxes,
            self.analysis_tdf,
        ) = create_and_open_cached_tdf(folder_startrek=folder_startrek, **kwargs)
        self.dataset = {
            col: self.dataset_df[col].to_numpy() for col in self.dataset_df.columns
        }

        self.duckcon = duckdb.connect()
        self.duckcon.execute(
            "ATTACH DATABASE '{}' AS analysis_tdf; ".format(self.analysis_tdf)
        )
        self.Frames = self.duckcon.execute("SELECT * FROM analysis_tdf.Frames;").df()


if __name__ == "__main__":
    OT = CachedOpenTIMS("/mnt/storage/science/midia_rawdata/8027.d")
    print(OT.box_query([345, 645, 746], 100, 500))
