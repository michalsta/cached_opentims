from pathlib import Path

import numpy as np
import pandas as pd
import numba
from opentimspy import OpenTIMS
import mmapped_df



@numba.njit
def index(frames, scans, starts, counts):
    for idx in range(len(frames)):
        frame = frames[idx]
        scan = scans[idx]
        if starts[frame, scan] == 0:
            starts[frame, scan] = idx
        counts[frame,scan] += 1

    last_frame = 0
    for frame in range(starts.shape[0]):
        for scan in range(starts.shape[1]):
            if starts[frame, scan] == 0:
                starts[frame, scan] = last_frame
            last_frame = starts[frame, scan]

@numba.njit()
def count(frames, min_scan, max_scan, counts):
    ret = np.uint64(0)
    for frame in frames:
        for scan in range(min_scan, max_scan+1):
            ret += counts[frame, scan]
    return ret

@numba.njit()
def fill_data(frames, scan_min, scan_max, starts, counts, src, T):
    t_idx = 0
    for frame in frames:
        for scan in range(scan_min, scan_max+1):
            d_start = starts[frame, scan]
            for d_idx in range(0, counts[frame, scan]):
                T[t_idx] = src[d_start+d_idx]
                t_idx += 1
                d_idx += 1

class CachedOpenTIMS:
    def __init__(self, path, tmpdir = None):
        path = Path(path)
        assert tmpdir is None
        cache_dir = Path(str(path)+".cache")
        self.OT = OpenTIMS(path)
        if not cache_dir.exists():
            with mmapped_df.DatasetWriter(cache_dir) as DW:
                for frame in self.OT:
                    #print(frame)
                    DW.append_df(pd.DataFrame(frame))

        self.backend = mmapped_df.open_dataset(cache_dir)

        self.starts = np.zeros(shape=(self.OT.max_frame+1, self.OT.max_scan+1), dtype=np.uint64)
        self.counts = np.zeros(shape=(self.OT.max_frame + 1, self.OT.max_scan + 1), dtype=np.uint64)
        self.index = index(self.backend.frame.values, self.backend.scan.values, self.starts, self.counts)

    def query(self, frames):
        return box_query(frames, self.OT.min_scan, self.OT.max_scan)

    def box_query(self, frames, min_scan, max_scan):
        if not isinstance(frames, np.ndarray):
            frames = np.array(frames, dtype=np.uint64)
        arr_size = count(frames, min_scan, max_scan, self.counts)
        acc = {}
        for colname in self.backend:
            T = np.empty(shape=(arr_size,), dtype=self.backend.dtypes[colname])
            fill_data(frames, min_scan, max_scan,self.starts,self.counts,self.backend[colname].values,T)
            acc[colname] = T
        return pd.DataFrame(acc)

if __name__ == '__main__':
    OT = CachedOpenTIMS("/mnt/storage/science/midia_rawdata/8027.d")
    print(OT.box_query([345,645,746], 100, 500))