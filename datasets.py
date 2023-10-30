'''
Copied from https://github.com/JiaruiFeng/KP-GNN/tree/a127847ed8aa2955f758476225bc27c6697e7733
'''
from sklearn.metrics import accuracy_score
import torch
import pickle
import numpy as np
import scipy.io as sio
from scipy.special import comb
import networkx as nx
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected, degree
from torch_geometric.datasets import TUDataset, ZINC, GNNBenchmarkDataset, QM9
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from data_I2GNN import dataset_random_graph


class PlanarSATPairsDataset(InMemoryDataset):

    def __init__(self,
                 root="dataset/EXP",
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(root, transform,
                                                    pre_transform, pre_filter)
        data, slices = torch.load(self.processed_paths[0])
        if data.x.dim() == 1:
            data.x = data.x.unsqueeze(-1)
        data.x += 1
        data.y = data.y.to(torch.float).reshape(-1, 1)
        self.data, self.slices = data, slices

    @property
    def raw_file_names(self):
        return ["GRAPHSAT" + ".pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        with open("dataset/EXP/raw/" + "newGRAPHSAT" + ".pkl", "rb") as f:
            data_list = pickle.load(f)
        data_list = [Data.from_dict(_) for _ in data_list]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def EXP_node_feature_transform(data):
    data.x = data.x[:, 0].to(torch.long)
    return data


class GraphCountDataset(InMemoryDataset):

    def __init__(self,
                 root="dataset/subgraphcount",
                 transform=None,
                 pre_transform=None):
        super(GraphCountDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        a = sio.loadmat(self.raw_paths[0])
        self.train_idx = torch.from_numpy(a['train_idx'][0])
        self.val_idx = torch.from_numpy(a['val_idx'][0])
        self.test_idx = torch.from_numpy(a['test_idx'][0])

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        b = self.processed_paths[0]
        a = sio.loadmat(self.raw_paths[0])  # 'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A = a['A'][0]
        # list of output
        Y = a['F']

        data_list = []
        for i in range(len(A)):
            a = A[i]
            A2 = a.dot(a)
            A3 = A2.dot(a)
            tri = np.trace(A3) / 6
            tailed = ((np.diag(A3) / 2) * (a.sum(0) - 2)).sum()
            cyc4 = 1 / 8 * (np.trace(A3.dot(a)) + np.trace(A2) - 2 * A2.sum())
            cus = a.dot(np.diag(np.exp(-a.dot(a).sum(1)))).dot(a).sum()

            deg = a.sum(0)
            star = 0
            for j in range(a.shape[0]):
                star += comb(int(deg[j]), 3)

            expy = torch.tensor([[tri, tailed, star, cyc4, cus]])

            E = np.where(A[i] > 0)
            edge_index = torch.Tensor(np.vstack(
                (E[0], E[1]))).type(torch.int64)
            x = torch.ones(A[i].shape[0], 1).long()  # change to category
            # y=torch.tensor(Y[i:i+1,:])
            data_list.append(Data(edge_index=edge_index, x=x, y=expy))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SRDataset(InMemoryDataset):

    def __init__(self,
                 root="dataset/sr25",
                 transform=None,
                 pre_transform=None):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        data, slices = torch.load(self.processed_paths[0])
        data.x = data.x.long() + 1
        data.y = torch.arange(data.y.shape[0])
        self.data, self.slices = data, slices

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  # sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i, datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(), 1)
            edge_index = to_undirected(
                torch.tensor(list(datum.edges())).transpose(1, 0))
            data_list.append(Data(edge_index=edge_index, x=x, y=0))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class myEvaluator(Evaluator):

    def __init__(self, name):
        super().__init__(name=name)

    def __call__(self, y_pred, y_true):
        ret = super().eval({"y_pred": y_pred, "y_true": y_true})
        assert len(ret) == 1
        return list(ret.values())[0]


from torchmetrics import Accuracy, MeanAbsoluteError, F1Score
from torchmetrics.classification import MultilabelAveragePrecision
from typing import Iterable, Callable, Optional, Tuple
from torch_geometric.data import Dataset
from torch_geometric.datasets import LRGBDataset


def loaddataset(name: str,
                **kwargs):  #-> Iterable[Dataset], str, Callable, str
    if name == "sr":
        dataset = SRDataset(**kwargs)
        dataset.num_tasks = torch.max(dataset.data.y).item() + 1
        return (dataset, dataset, dataset), "fixed", Accuracy(
            "multiclass",
            num_classes=dataset.num_tasks), "cls"  # full training/valid/test??
    elif name == "EXP":
        dataset = PlanarSATPairsDataset(
            pre_transform=EXP_node_feature_transform, **kwargs)
        dataset.num_tasks = 1
        return (dataset, ), "fold-8-1-1", Accuracy("binary"), "bincls"
    elif name == "CSL":

        def CSL_node_feature_transform(data):
            if "x" not in data:
                data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)
            return data

        dataset = GNNBenchmarkDataset("dataset",
                                      "CSL",
                                      pre_transform=CSL_node_feature_transform,
                                      **kwargs)
        dataset.num_tasks = torch.max(dataset.y).item() + 1
        return (dataset, ), "fold-8-1-1", Accuracy("multiclass",
                                                   num_classes=10), "cls"
    elif name.startswith("subgcount"):
        y_slice = int(name[len("subgcount"):])
        dataset = GraphCountDataset(**kwargs)
        dataset.data.y = dataset.data.y - dataset.data.y.mean(dim=0)
        dataset.data.y = dataset.data.y / dataset.data.y.std(dim=0)
        dataset.data.y = dataset.data.y[:, [y_slice]]
        # degree feature
        # dataset.data.x.copy_(torch.cat([degree(dat.edge_index[0], num_nodes=dat.num_nodes, dtype=torch.long) for dat in dataset]).reshape(-1, 1))
        dataset.num_tasks = 1
        dataset.data.y = dataset.data.y.to(torch.float)
        return (dataset[dataset.train_idx], dataset[dataset.val_idx],
                dataset[dataset.test_idx]
                ), "fixed", MeanAbsoluteError(), "l1reg"  #
    elif name.startswith("count_cycle"):
        sname = name[len("count_cycle"):]
        yidx = int(sname)
        trn_ds = dataset_random_graph("count_cycle", split="train", yidx=yidx)
        val_ds = dataset_random_graph("count_cycle", split="val", yidx=yidx)
        y_train_val = torch.cat([trn_ds.y, val_ds.y], dim=0)
        mean = y_train_val.mean()
        std = y_train_val.std()
        trn_ds = dataset_random_graph("count_cycle",
                                      split="train",
                                      yidx=yidx,
                                      ymean=mean,
                                      ystd=std)
        val_ds = dataset_random_graph("count_cycle",
                                      split="val",
                                      yidx=yidx,
                                      ymean=mean,
                                      ystd=std)
        tst_ds = dataset_random_graph("count_cycle",
                                      split="test",
                                      yidx=yidx,
                                      ymean=mean,
                                      ystd=std)

        trn_ds.num_tasks = 1
        val_ds.num_tasks = 1
        tst_ds.num_tasks = 1

        # print(trn_ds.data.y.shape, trn_ds[0])
        return (trn_ds, val_ds,
                tst_ds), "fixed", MeanAbsoluteError(), "nodesmoothl1reg"  #
    elif name.startswith("count_graphlet"):
        sname = name[len("count_graphlet"):]
        yidx = int(sname)
        trn_ds = dataset_random_graph("count_graphlet",
                                      split="train",
                                      yidx=yidx)
        val_ds = dataset_random_graph("count_graphlet", split="val", yidx=yidx)
        y_train_val = torch.cat([trn_ds.y, val_ds.y], dim=0)
        mean = y_train_val.mean()
        std = y_train_val.std()
        trn_ds = dataset_random_graph("count_graphlet",
                                      split="train",
                                      yidx=yidx,
                                      ymean=mean,
                                      ystd=std)
        val_ds = dataset_random_graph("count_graphlet",
                                      split="val",
                                      yidx=yidx,
                                      ymean=mean,
                                      ystd=std)
        tst_ds = dataset_random_graph("count_graphlet",
                                      split="test",
                                      yidx=yidx,
                                      ymean=mean,
                                      ystd=std)

        trn_ds.num_tasks = 1
        val_ds.num_tasks = 1
        tst_ds.num_tasks = 1
        return (trn_ds, val_ds,
                tst_ds), "fixed", MeanAbsoluteError(), "nodel1reg"  #
    elif name == "pascalvocsp":
        xmean = torch.tensor([
            4.2845e-01, 3.7611e-01, 1.4307e-01, 2.6746e-02, 3.0037e-02,
            2.7267e-02, 5.0544e-01, 4.6751e-01, 2.3912e-01, 3.5321e-01,
            2.8807e-01, 7.9090e-02, 1.9030e+02, 2.4771e+02
        ])  # mean of training data's x
        xstd = torch.tensor([
            2.5953e-01, 2.5717e-01, 2.7131e-01, 5.4823e-02, 5.4429e-02,
            5.4475e-02, 2.6238e-01, 2.6601e-01, 2.7751e-01, 2.5197e-01,
            2.4986e-01, 2.6070e-01, 1.1768e+02, 1.4007e+02
        ])  # std of training data's x
        eamean = torch.tensor([0.0764,
                               33.7348])  # mean of training data's edge_attr
        eastd = torch.tensor([0.0869,
                              20.9451])  # std of training data's edge_attr

        def pascalvocsp_pre_transform(data):
            data.x = (data.x - xmean) / xstd
            data.edge_attr = (data.edge_attr - eamean) / eastd
            return data

        trn_ds = LRGBDataset("./dataset",
                             "PascalVOC-SP",
                             "train",
                             pre_transform=pascalvocsp_pre_transform)
        val_ds = LRGBDataset("./dataset",
                             "PascalVOC-SP",
                             "val",
                             pre_transform=pascalvocsp_pre_transform)
        tst_ds = LRGBDataset("./dataset",
                             "PascalVOC-SP",
                             "test",
                             pre_transform=pascalvocsp_pre_transform)
        trn_ds.num_tasks = 21
        val_ds.num_tasks = 21
        tst_ds.num_tasks = 21
        return (trn_ds, val_ds,
                tst_ds), "fixed", F1Score("multiclass",
                                          num_classes=21,
                                          average="macro"), "nodecls"  #
    elif name == "pepfunc":

        def pepfunc_pre_process(data):
            data.x = data.x + 1
            data.edge_attr = data.edge_attr + 1
            return data

        trn_ds = LRGBDataset("./dataset",
                             "Peptides-func",
                             "train",
                             pre_transform=pepfunc_pre_process).shuffle()
        val_ds = LRGBDataset("./dataset",
                             "Peptides-func",
                             "val",
                             pre_transform=pepfunc_pre_process).shuffle()
        tst_ds = LRGBDataset("./dataset",
                             "Peptides-func",
                             "test",
                             pre_transform=pepfunc_pre_process).shuffle()
        trn_ds.num_tasks = 10
        val_ds.num_tasks = 10
        tst_ds.num_tasks = 10
        return (trn_ds, val_ds,
                tst_ds), "fixed", lambda x, y: MultilabelAveragePrecision(
                    10, average="macro")(x, (y > 0.5).to(torch.long)), "bincls"
    elif name == "pepstruct":

        def pepstruct_pre_process(data):
            data.x = data.x + 1
            data.edge_attr = data.edge_attr + 1
            return data

        trn_ds = LRGBDataset("./dataset",
                             "Peptides-struct",
                             "train",
                             pre_transform=pepstruct_pre_process)
        val_ds = LRGBDataset("./dataset",
                             "Peptides-struct",
                             "val",
                             pre_transform=pepstruct_pre_process)
        tst_ds = LRGBDataset("./dataset",
                             "Peptides-struct",
                             "test",
                             pre_transform=pepstruct_pre_process)
        trn_ds.num_tasks = 11
        val_ds.num_tasks = 11
        tst_ds.num_tasks = 11
        return (trn_ds, val_ds, tst_ds), "fixed", MeanAbsoluteError(), "l1reg"
    elif name in ["MUTAG", "DD", "PROTEINS", "PTC-MR", "IMDB-BINARY"]:
        dataset = TUDataset("dataset", name=name, **kwargs)
        dataset.num_tasks = 1
        dataset.y = dataset.y.to(torch.float)
        print(dataset.data)
        return (dataset, ), "fold-9-0-1", Accuracy("binary"), "bincls"
    elif name == "zinc":

        def zincpretransform(data):
            data.x = data.x + 1
            data.y = data.y.reshape(-1, 1)
            data.edge_attr = (data.edge_attr + 1).to(torch.long).reshape(-1, 1)
            return data

        trn_d = ZINC("dataset/ZINC",
                     subset=True,
                     split="train",
                     pre_transform=zincpretransform,
                     **kwargs)
        val_d = ZINC("dataset/ZINC",
                     subset=True,
                     split="val",
                     pre_transform=zincpretransform)
        tst_d = ZINC("dataset/ZINC",
                     subset=True,
                     split="test",
                     pre_transform=zincpretransform)
        trn_d.num_tasks = 1
        val_d.num_tasks = 1
        tst_d.num_tasks = 1
        return (trn_d, val_d,
                tst_d), "fixed", MeanAbsoluteError(), "smoothl1reg"  #"reg"
    
    elif name == "zinc-full":

        def zincpretransform(data):
            data.x = data.x + 1
            data.y = data.y.reshape(-1, 1)
            data.edge_attr = (data.edge_attr + 1).to(torch.long).reshape(-1, 1)
            return data

        trn_d = ZINC("dataset/ZINC",
                     subset=False,
                     split="train",
                     pre_transform=zincpretransform,
                     **kwargs)
        val_d = ZINC("dataset/ZINC",
                     subset=False,
                     split="val",
                     pre_transform=zincpretransform)
        tst_d = ZINC("dataset/ZINC",
                     subset=False,
                     split="test",
                     pre_transform=zincpretransform)
        trn_d.num_tasks = 1
        val_d.num_tasks = 1
        tst_d.num_tasks = 1
        return (trn_d, val_d,
                tst_d), "fixed", MeanAbsoluteError(), "smoothl1reg"  #"reg"
    elif name.startswith("qm9"):
        y_slice = int(name[3:])

        def qm9pretrans(data):
            data.x = (data.x + 1)# .to(torch.long)
            data.edge_attr = (data.edge_attr + 1)# .to(torch.long)
            return data
        
        def qm9trans(data):
            data.y = data.y[:, [y_slice]]
            return data

        dataset = QM9("dataset/qm9", transform=qm9trans, pre_transform=qm9pretrans, **kwargs)
        tenpercent = int(len(dataset) * 0.1)
        dataset.num_tasks = 1
        dataset = dataset.shuffle()
        test_dataset = dataset[:tenpercent]
        val_dataset = dataset[tenpercent:2 * tenpercent]
        train_dataset = dataset[2 * tenpercent:]
        return (train_dataset, val_dataset,
                test_dataset), "8-1-1", MeanAbsoluteError(), "l1reg"
    elif name.startswith("ogbg"):

        def pretransform(data):
            data.edge_attr = data.edge_attr + 1
            return data

        dataset = PygGraphPropPredDataset(name=name,
                                          pre_transform=pretransform)
        split_idx = dataset.get_idx_split()
        if "molhiv" in name:
            task = "bincls"
        elif "pcba" in name:
            task = "bincls"
        else:
            raise NotImplementedError
        dataset.y = dataset.y.to(torch.float)
        dataset.x = dataset.x + 1
        return (dataset[split_idx["train"]], dataset[split_idx["valid"]],
                dataset[split_idx["test"]]), "fixed", myEvaluator(name), task
    else:
        raise NotImplementedError(name)


if __name__ == "__main__":
    datalist = [
        "sr", "EXP", "CSL", "subgcount0", "zinc", "MUTAG", "DD", "PROTEINS",
        "ogbg-molhiv", "ogbg-molpcba"
    ]  # "QM9",  "IMDB-BINARY",
    for ds in datalist:
        datasets = loaddataset(ds)[0]
        dataset = datasets[0]
        data = dataset[0]
        print(ds, dataset.num_tasks, data, data.x.dtype, data.y.dtype)