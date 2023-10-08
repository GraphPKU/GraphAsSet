import torch_geometric.data as pygData
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.utils import is_undirected
import torch


def batch2dense(batch: pygData.Batch,
                batch_size: int = None,
                max_num_nodes: int = None,
                aligned_size: int = 1):  #?? 32 if pep
    max_num_nodes = torch.max(torch.diff(batch.ptr))
    max_num_nodes = (
        (max_num_nodes + aligned_size - 1) // aligned_size) * aligned_size
    x, nodemask = to_dense_batch(x=batch.x,
                                 batch=batch.batch,
                                 batch_size=batch_size,
                                 max_num_nodes=max_num_nodes)
    nodemask = torch.logical_not(nodemask)  # true means not node
    max_num_nodes = x.shape[1]
    batch_size = x.shape[0]
    A = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr,
                     max_num_nodes).contiguous()
    if getattr(batch, "pos", None) is not None:
        pos, _ = to_dense_batch(batch.pos,
                                batch.batch,
                                batch_size=batch_size,
                                max_num_nodes=max_num_nodes)
        return A, x, nodemask, pos, max_num_nodes
    else:
        return A, x, nodemask, max_num_nodes
