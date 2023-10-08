'''
modified from https://github.com/GraphPKU/I2GNN/blob/master/data_processing.py
'''
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
import pickle
import os
from typing import Callable, List, Optional
from torch_geometric.data import Data, InMemoryDataset
import scipy.io as scio


class dataset_random_graph(InMemoryDataset):
    def __init__(self, dataname='count_cycle', root='dataset', processed_name='processed', split='train', yidx: int=0, ymean: float=0, ystd: float=1):
        self.root = root
        self.dataname = dataname
        self.raw = os.path.join(root, dataname)
        self.processed = os.path.join(root, dataname, processed_name)
        super(dataset_random_graph, self).__init__(root=root, transform=None, pre_transform=None,
                                            pre_filter=None)
        split_id = 0 if split == 'train' else 1 if split == 'val' else 2
        data, slices = torch.load(self.processed_paths[split_id])
        data.y = (data.y[:, [yidx]]-ymean)/ystd
        self.data, self.slices = data, slices
        self.y_dim = self.data.y.size(-1)

    @property
    def raw_dir(self):
        name = 'raw'
        return os.path.join(self.root, self.dataname, name)

    @property
    def processed_dir(self):
        return self.processed
    @property
    def raw_file_names(self):
        names = ["data"]
        return ['{}.mat'.format(name) for name in names]

    @property
    def processed_file_names(self):
        return ['data_tr.pt', 'data_val.pt', 'data_te.pt']

    def adj2data(self, A, y):
        # x: (n, d), A: (e, n, n)
        # begin, end = np.where(np.sum(A, axis=0) == 1.)
        begin, end = np.where(A == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        num_nodes = A.shape[0]
        if y.ndim == 1:
            y = y.reshape([1, -1])
        x = torch.ones((num_nodes, 1), dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=torch.tensor(y), num_nodes=torch.tensor([num_nodes]))

    @staticmethod
    def wrap2data(d):
        # x: (n, d), A: (e, n, n)
        x, A, y = d['x'], d['A'], d['y']
        x = torch.tensor(x)
        begin, end = np.where(np.sum(A, axis=0) == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        y = torch.tensor(y[-1:])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def process(self):
        # process npy data into pyg.Data
        print('Processing data from ' + self.raw_dir + '...')
        raw_data = scio.loadmat(self.raw_paths[0])
        if raw_data['F'].shape[0] == 1:
            data_list_all = [[self.adj2data(raw_data['A'][0][i], raw_data['F'][0][i]) for i in idx]
                             for idx in [raw_data['train_idx'][0], raw_data['val_idx'][0], raw_data['test_idx'][0]]]
        else:
            data_list_all = [[self.adj2data(A, y) for A, y in zip(raw_data['A'][0][idx][0], raw_data['F'][idx][0])]
                        for idx in [raw_data['train_idx'], raw_data['val_idx'], raw_data['test_idx']]]
        for save_path, data_list in zip(self.processed_paths, data_list_all):
            print('pre-transforming for data at'+save_path)
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                temp = []
                for i, data in enumerate(data_list):
                    if i % 100 == 0:
                        print('Pre-processing %d/%d' % (i, len(data_list)))
                    temp.append(self.pre_transform(data))
                data_list = temp
                # data_list = [self.pre_transform(data) for data in data_list]
            data, slices = self.collate(data_list)
            torch.save((data, slices), save_path)



def create_one_hot_label(d, max_num_rings):
    # please manually define this function replying on the labels you want
    num_labels = 2 + (1 + max_num_rings) + 2 # 1-bit for HAS RING, 1-bit for HAS tricycles
    labels = []
    # if has ring
    flag = [1., 0] if d['has_rings'] == 'True' else [0, 1.]
    labels.append(np.array(flag).astype(np.float32))
    # how many rings
    flag = np.eye(max_num_rings + 1)[int(d['nring'])]
    labels.append(flag.astype(np.float32))
    # if has 3-ring
    # flag = [1., 0] if int(d['natom_in_3_rings']) > 0 else [0, 1.]
    # mol = Chem.MolFromSmiles(Chem.CanonSmiles(d['smiles']))
    # flag = utils.detect_triple_ring(mol)
    flag = [1., 0] if d['has_triple_ring'] == 'True' else [0, 1.]
    labels.append(np.array(flag).astype(np.float32))

    return labels


class Chembl(InMemoryDataset):

    def __init__(self, root: str = "dataset/count_chembl", processed_name: str = 'processed'):
        self.processed_name = processed_name
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['chembl.pkl']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.processed_name)

    @property
    def processed_file_names(self) -> str:
        return 'data_processed.pt'

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        with open(self.raw_paths[0], 'rb') as f:
            smiles_list = pickle.load(f)


        data_list = []
        for i, sm in enumerate(smiles_list):
            if i % 500 == 0:
                print('Pre-processing: %d/%d' %(i, len(smiles_list)))
            mol = Chem.MolFromSmiles(sm)
            N = mol.GetNumAtoms()

            # x
            x = torch.zeros([N, ], dtype=torch.long)

            # edge
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [1]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = torch.reshape(edge_type, [-1, 1]).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            data = Data(x=x, edge_index=edge_index,
                        edge_attr=edge_attr, name=sm)

            # calculate rings
            size_list = [3, 4, 5, 6, 7]
            ssr = Chem.GetSymmSSSR(mol)
            ssr = [list(s) for s in ssr]
            n_kring_graph = np.zeros([1, len(size_list)], dtype=np.int)
            n_kring_node = np.zeros((N, len(size_list)), dtype=np.int)
            for ring in ssr:
                size = len(ring)
                if size not in size_list:
                    continue
                # node level
                for atom in ring:
                    n_kring_node[atom, size_list.index(size)] += 1
                # graph level
                n_kring_graph[0, size_list.index(size)] += 1
            n_kring_graph = torch.tensor(n_kring_graph, dtype=torch.int)
            n_kring_node = torch.tensor(n_kring_node, dtype=torch.int)
            data.n_kring_graph = n_kring_graph
            # data.n_kring_node = n_kring_node
            data.y = n_kring_node.float()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

if __name__ == "__main__":
    ds = dataset_random_graph("count_cycle", split="train")
    dataset_random_graph("count_cycle", split="valid")
    dataset_random_graph("count_cycle", split="test")
    dataset_random_graph("count_graphlet", split="train")
    dataset_random_graph("count_graphlet", split="valid")
    dataset_random_graph("count_graphlet", split="test")
    print(ds[0])
    Chembl()