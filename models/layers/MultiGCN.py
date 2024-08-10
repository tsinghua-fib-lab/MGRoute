import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import mlp, mlp_res

class MultiGCNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(self.__class__, self).__init__()
        self.hidden_dim = hidden_dim
        # self.w_dist = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.w_rel = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.w_dist = nn.Sequential(mlp_res(hidden_dim*2, hidden_dim*2), nn.Linear(hidden_dim*2, hidden_dim, bias=False))


        self.w_dist = mlp_res(hidden_dim, hidden_dim)
        # 多头
        # self.num_heads = 8
        # self.w_dist = mlp(hidden_dim, [hidden_dim, int(hidden_dim/self.num_heads)])
        # self.final = mlp_res(hidden_dim, hidden_dim)
        # self.attention = nn.Parameter(torch.Tensor(self.num_heads, hidden_dim*1))

        self.w_rel = mlp_res(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)
        self.attention = nn.Parameter(torch.Tensor(1, hidden_dim))
        self.node_map = mlp_res(hidden_dim, hidden_dim)
        self.edge_map = mlp_res(hidden_dim, hidden_dim)

    def propagation(self, A, mat):
        '''
        A : (B, N, N)
        mat: (B, N, N, H)
        '''
        B, N, H = mat.shape[0], mat.shape[1], mat.shape[3]
        A = A.reshape(B*N, 1, N)
        mat = mat.reshape(B*N, N, H)
        res = torch.bmm(A, mat)
        return res.reshape(B, N, H)
    
    def forward(self, node_embed, edge_fea, A_dist):
        A_dist = A_dist.float()

        node_new = self.node_map(node_embed)
        edge_new = self.edge_map(edge_fea)

        # Attention mechanism
        edge_and_node = node_new.unsqueeze(1) * edge_new #节点和自己的每条边做点积
        attention_coef = F.softmax(torch.matmul(edge_and_node, self.attention.transpose(0, 1)), dim=-1).squeeze(-1)

        node_embed = self.propagation(attention_coef * A_dist, self.w_dist(edge_and_node))
        node_embed = node_embed.transpose(1, 2).contiguous()
        node_embed_bn = self.batch_norm(node_embed)
        node_embed = node_embed_bn.transpose(1,2).contiguous()

        return self.act(node_embed), self.w_rel(edge_fea)
    

class PointEmbGCN(nn.Module):

    def __init__(self, num_gcn_layers, output_emb_size, output_emb_size_out):

        super(PointEmbGCN, self).__init__()

        self.num_gcn_layers = num_gcn_layers
        # self.num_gcn_layers = 1
        self.output_emb_size = output_emb_size


        # self.w_dep_r = nn.Linear(1, self.output_emb_size, bias=False)
        # self.w_dist_r = DistEmb_GCN(dist_emb_size=self.output_emb_size)
        self.w_dist_r = nn.Linear(2, self.output_emb_size, bias=True)

        self.node_emb_transform = nn.Linear(output_emb_size_out, output_emb_size)

        # self.gcn_layers = nn.ModuleList([MultiGCNLayer(hidden_dim=output_emb_size) for _ in range(self.num_gcn_layers)])

        self.gcn_dist = MultiGCNLayer(hidden_dim=output_emb_size, num_layers=self.num_gcn_layers)
        # self.gcn_de = MultiGCNLayer(hidden_dim=output_emb_size, num_layers=self.num_gcn_layers)
        # self.gcn_pi = MultiGCNLayer(hidden_dim=output_emb_size, num_layers=self.num_gcn_layers)

        self.final_layer = nn.Linear(output_emb_size, output_emb_size_out)
        self.rider_map = nn.Linear(output_emb_size_out, 3)

    def get_adjacency(self, E_dist, point_masks):

        batch_size, graph_size = E_dist.shape[0], E_dist.shape[1]

        A_dist = torch.zeros([batch_size, graph_size, graph_size]).to(E_dist.device)
        node_masks = ~point_masks
        edge_masks = node_masks.unsqueeze(1) * node_masks.unsqueeze(-1)
        B_indices = torch.arange(batch_size, device=E_dist.device)[:, None]
        N_indices = torch.arange(graph_size, device=E_dist.device)
        # 找出最小和倒数第二非零值
        min_dist_idx = torch.argsort(E_dist + 1e9 * (~edge_masks), dim=-1)[:, :, 0]
        min_dist_idx_2 = torch.argsort(E_dist + 1e9 * (~edge_masks), dim=-1)[:, :, 1]
        A_dist[B_indices, N_indices, min_dist_idx] = -1
        A_dist[B_indices, N_indices, min_dist_idx_2] = 1
        A_dist[B_indices, min_dist_idx, N_indices] = -1
        A_dist[B_indices, min_dist_idx_2, N_indices] = 1
        A_dist = A_dist * edge_masks

        # A_dist所有不为0的值都为1
        A_adj = torch.where(A_dist != 0, torch.ones_like(A_dist), torch.zeros_like(A_dist))
        return A_dist, A_adj