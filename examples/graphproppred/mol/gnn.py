import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from laf import ScatterAggregationLayer, ExponentialLAF
from conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean", laf_fun='mean', laf_layers='false', device='cuda', lafgrad=True):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, laf_fun=laf_fun, laf_layers=laf_layers, device=device)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, laf_fun=laf_fun, laf_layers=laf_layers, device=device)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        elif self.graph_pooling == "laf":
            if laf_fun =='exp':
                self.pool = ExponentialLAF()
            else:
                self.pool = ScatterAggregationLayer(function=laf_fun, grad=lafgrad, device=device)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        if self.graph_pooling == 'laf' and isinstance(self.pool, ScatterAggregationLayer):
            x_min = torch.min(h_node, dim=0, keepdim=True)[0]
            s = torch.ones_like(h_node) * x_min
            s = F.relu(-s)
            h_node = h_node + s
            s = F.relu(-x_min)
            out = self.pool(h_node, batched_data.batch)
            s_out = self.pool(s, torch.tensor([0], device=self.pool.device))
            s_out = torch.ones_like(out) * s_out
            h_graph = out - s_out
        else:
            h_graph = self.pool(h_node, batched_data.batch)
        return self.graph_pred_linear(h_graph)


if __name__ == '__main__':
    GNN(num_tasks = 10)
