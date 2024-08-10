import torch
import torch.nn as nn
import torch.nn.functional as F

class PointAttention(nn.Module):

    def __init__(self, hidden_size=256):
        super(PointAttention, self).__init__()
        self.q_dense = nn.Linear(hidden_size*2, hidden_size)  # Replaces tf.layers.dense for query
        self.e_dense = nn.Linear(hidden_size, hidden_size)  # Replaces tf.layers.dense for reference
        self.v = nn.Parameter(torch.randn(hidden_size,1))  # Trainable vector v
        # nn.init.xavier_uniform_(self.v)
        self.hidden_size = hidden_size

    def forward(self, query, ref):
        batch_size, graph_size = ref.size(0), ref.size(1)

        q = self.q_dense(query)
        q = q.view(batch_size, self.hidden_size, 1)
        q = q.expand(-1, -1, graph_size)

        e = self.e_dense(ref)
        e = e.transpose(1, 2)

        expanded_v = self.v.view(1, 1, -1).expand(batch_size, -1, -1)

        return torch.matmul(expanded_v, torch.tanh(q + e))

class PointAttention2(nn.Module):

    def __init__(self, hidden_size=256, num_heads=8):
        super(PointAttention2, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.q_dense = nn.Linear(hidden_size*2, hidden_size)
        self.e_dense = nn.Linear(hidden_size, hidden_size)
        self.v_dense = nn.Linear(self.head_dim, 1)

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, query, ref):
        batch_size, graph_size = ref.size(0), ref.size(1)

        q = self.q_dense(query).view(batch_size, self.num_heads, self.head_dim)
        e = self.e_dense(ref).view(batch_size, graph_size, self.num_heads, self.head_dim)

        q = q.unsqueeze(2).expand(-1, -1, graph_size, -1)
        e = e.transpose(1, 2)

        v = torch.tanh(q + e)
        v = self.v_dense(v)

        out = v.sum(dim=1).transpose(1, 2)

        return out
    
class PointAttention3(nn.Module):

    def __init__(self, hidden_size=256, num_heads=8):
        super(PointAttention3, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        # 分别对query和reference的线性变换
        self.q_dense = nn.Linear(hidden_size*2, hidden_size)
        self.e_dense = nn.Linear(hidden_size, hidden_size)

        # 用于多头注意力合并后的输出
        self.out_dense = nn.Linear(self.num_heads, 1)

        # self.norm = nn.LayerNorm(hidden_size)

    def forward(self, query, ref):
        batch_size, seq_len, _ = ref.size()

        # Expand query to match the number of heads
        q = self.q_dense(query)
        q = F.gelu(q).view(batch_size, self.num_heads, self.head_dim).unsqueeze(2)

        e = self.e_dense(ref)
        e = F.gelu(e).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Transpose to bring heads to the front for matrix multiplication
        q = q.transpose(0, 1)  # HxBxD
        e = e.transpose(1, 2).transpose(0, 1)  # HxBxNxD

        # Compute attention scores
        scores = torch.einsum('hbld,hbnd->hbn', q, e).transpose(0, 1).transpose(1, 2)  # HxBxN
        # scores = scores.mean(dim=0, keepdim=True)  # BxNxH

        # Apply linear transformation and layer normalization
        out = self.out_dense(scores).transpose(1, 2)
        # out = self.norm(out)

        return out