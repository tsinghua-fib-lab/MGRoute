import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedGraphConv(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(GatedGraphConv, self).__init__()
        self.U = nn.Linear(node_dim, hidden_dim, bias=False)
        self.V = nn.Linear(node_dim, hidden_dim, bias=False)
        self.A = nn.Linear(node_dim, hidden_dim, bias=False)
        self.B = nn.Linear(node_dim, hidden_dim, bias=False)
        self.C = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.bn_node = nn.BatchNorm1d(hidden_dim)
        self.bn_edge = nn.BatchNorm1d(hidden_dim)

    def forward(self, h, e, adj):

        # 边特征更新
        # 注意：这里仅仅是示例，实际中可能会根据图结构有所不同
        e = self.C(e) + self.A(h.unsqueeze(2)) + self.B(h.unsqueeze(1))
        e = self.bn_edge(e.view(-1, e.size(-1))).view(e.size())  # 批量归一化
        e = F.relu(e)
        adj_augmented = adj.unsqueeze(3)
        e = e * adj_augmented
        # 边特征的门控权重
        e_weight = torch.sigmoid(e)
        # 计算门控后的边权重
        e_norm = F.softmax(e_weight, dim=2)

        # 门控权重的维度是 [batch_size, num_nodes, num_nodes, hidden_dim]
        # 节点特征的维度是 [batch_size, num_nodes, node_dim]
        # 我们需要将节点特征投影到 hidden_dim 维度
        h_in = self.U(h)
        h_neighbors = self.V(h)
        
        # 使用门控权重更新节点特征
        # 逐元素乘法用于应用门控权重

        m = torch.einsum('bmnd,bnd->bmnd', e_norm, h_neighbors)
        m = m.sum(dim=2)  # 对所有邻居特征求和
        m = self.bn_node(m.view(-1, m.size(-1))).view(m.size())  # 批量归一化要求将批次和节点维度合并
        h = h_in + F.relu(m)
        
        return h, e_norm

class GatedGCNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(self.__class__, self).__init__()
        self.hidden_dim = hidden_dim
        self.gatedgcn = GatedGraphConv(hidden_dim, hidden_dim, hidden_dim)
    
    def forward(self, node_embed, res_edge, Adj):
        node_embed, res_edge_new = self.gatedgcn(node_embed, res_edge, Adj)
        return node_embed, res_edge_new