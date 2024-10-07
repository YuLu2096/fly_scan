import io
import glob
import os

import pandas as pd
import numpy as np


class SpectroscopyDataset:

    def __init__(self, path, *args, **kwargs):
        self.path = path
        self.data = None
        self.energies_ev = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class ColumnMajorCSVSpectroscopyDataset(SpectroscopyDataset):
    def __init__(self, *args, column_name="YBCO_epararb.0001", **kwargs):
        super().__init__(*args, **kwargs)
        table = pd.read_csv(self.path, header=0)
        self.data = table[column_name].to_numpy()
        self.data = self.data.reshape(1, -1)
        self.energies_ev = table['energy'].to_numpy()


class YBCORawDataset(SpectroscopyDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        table = pd.read_table(self.path, comment='#', header=None, sep='\s+')
        self.energies_ev = table[0].to_numpy()
        self.data = np.log(table[4] / table[5]).to_numpy()
        self.data = self.data.reshape(1, -1)
        

class LTORawDataset(SpectroscopyDataset):
    def __init__(self, *args, filename_pattern='*', **kwargs):
        super().__init__(*args, **kwargs)
        filelist = glob.glob(os.path.join(self.path, filename_pattern))
        filelist.sort()
        self.data = []
        for i, fname in enumerate(filelist):
            table = pd.read_table(fname, comment='#', header=None, sep='\s+')
            if i == 0:
                self.energies_ev = table[0].to_numpy()
            else:
                assert len(table[0].to_numpy()) == len(self.energies_ev), \
                        "Inconsistent number of points at {} ({} vs {}).".format(
                            i, len(table[0].to_numpy()), len(self.energies_ev))
            data = np.log(table[2] / table[3]).to_numpy()
            self.data.append(data)
        self.data = np.stack(self.data)


class RowMajorCSVSpectroscopyDataset(SpectroscopyDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        table = pd.read_csv(self.path, header=None)
        self.data = table.iloc[1:].to_numpy()
        self.energies_ev = table.iloc[0].to_numpy()


class NORSpectroscopyDataset(SpectroscopyDataset):
    def __init__(self, *args, file_pattern='*.nor', data_column='norm', **kwargs):
        super().__init__(*args, **kwargs)
        self.data_column = data_column
        flist = glob.glob(os.path.join(self.path, file_pattern))
        flist.sort()
        self.data = []
        for f in flist:
            d, self.energies_ev = self.load_data_from_nor(f)
            self.data.append(d)
        self.data = np.array(self.data)

    def load_data_from_nor(self, path):
        table = read_nor(path)
        data = table[self.data_column].to_numpy()
        energies = table['e'].to_numpy()
        _, unique_inds = np.unique(energies, return_index=True)
        unique_inds = np.sort(unique_inds)
        data = data[unique_inds]
        energies = energies[unique_inds]
        return data, energies


def read_nor(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        columns = []
        for i, l in enumerate(lines):
            if not lines[i + 1].startswith('#'):
                columns = l[1:].strip().split()
                break
    table = pd.read_table(fname, comment='#', header=None, sep='\s+')
    table.columns = columns
    return table

