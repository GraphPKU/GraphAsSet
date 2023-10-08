from torch_geometric.datasets import LRGBDataset
import torch
import numpy as np

for i in ["Peptides-func",]: #  "Peptides-struct"
    ds = LRGBDataset("./dataset", i, "train")
    data = ds[0]
    num_nodes = [d.num_nodes for d in ds]
    print(data, ds.x.max(dim=0)[0], np.max(num_nodes),  ds.x.min(dim=0)[0], ds.edge_attr.max(dim=0)[0], ds.edge_attr.min(dim=0)[0], ds.y.unique())
    print(torch.mean((ds.y>0.5).to(torch.float), dim=0))
    