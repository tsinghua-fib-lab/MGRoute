import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import mlp
from .utility_funcs import normalize_adjacency_matrix
from .MultiHeadAttn import MultiHeadAttention, FeedForward
from .MultiGCN import MultiGCNLayer


    
class TransformerEncoder(nn.Module):
    def __init__(self, hp):
        super(TransformerEncoder, self).__init__()
        self.hp = hp
        hp_dropout_rate = 0.1
        self.dropout = nn.Dropout(hp_dropout_rate)

        # Encoder layers
        self.enc_layers = nn.ModuleList([TransformerEncoderLayer(hp) for _ in range(hp.num_blocks)])
        # Positional Encoding is usually implemented as a separate function or layer
        self.edge_layer = nn.Sequential(nn.Linear(8, hp.d_model), nn.ReLU())

    def positional_encoding(self, inputs, maxlen, masking=torch.tensor([1])):
        maxlen = maxlen.item()
        """
        Sinusoidal Positional_Encoding. See 3.5
        inputs: 3d tensor. (N, T, E)
        maxlen: scalar. Must be >= T
        masking: Boolean. If True, padding positions are set to zeros.
        scope: Optional scope for `variable_scope`.
        returns
        3d tensor that has the same shape as inputs.

        实现了正弦波位置编码（Sinusoidal Positional Encoding）。
        位置编码是在自然语言处理中，特别是在Transformer模型中使用的一种技术，用于给模型提供单词在序列中位置的信息。
        inputs: 输入张量，其形状为 [N, T, E]，其中 N 是批次大小，T 是序列长度，E 是嵌入维度。
        maxlen: 序列的最大长度，必须大于等于 T。
        masking: 布尔值，用于决定是否对填充位置应用掩码（设置为零）。
        scope: TensorFlow中变量作用域的名称。

        计算位置编码:

        使用一个形状为 [maxlen, E] 的矩阵来计算位置编码。对于每个位置 pos 和每个维度 i，使用正弦和余弦函数的变换来生成位置编码。这种方法可以让模型更容易学习到位置的相对关系。
        应用正弦和余弦函数:

        对于偶数维度（2i），应用正弦函数。
        对于奇数维度（2i+1），应用余弦函数。
        查找位置编码:

        使用 tf.nn.embedding_lookup 根据位置索引 position_ind 在预先计算的位置编码矩阵中查找相应的编码。
        掩码处理 (如果 masking 为 True):

        将输入中值为零的位置在输出位置编码中也设置为零。这通常用于处理填充（padding）
        """
        E = inputs.size(-1)
        N, T = inputs.size(0), inputs.size(1)

        # position indices
        position_ind = torch.arange(T).unsqueeze(0).repeat(N, 1)

        # First part of the PE function: sin and cos argument
#         E_t = torch.tensor([E]).item()
#         maxlen_t =torch.tensor([maxlen]).item()
#         C = torch.tensor([10000]).item()
        E_t = int(E)  # 如果 E 本来就是 float 或 int，直接使用即可
        maxlen_t = int(maxlen)  # 同上
        C = 10000  # 如果是常量，直接赋值即可
        position_enc_values = [
            [pos / (C**((i - i % 2) / E_t)) for i in range(E_t)]
            for pos in range(maxlen_t)
        ]
#         print("position_enc_values:",position_enc_values)
        position_enc = torch.tensor(position_enc_values).to(inputs.device)
        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = torch.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2])
        position_enc = position_enc.float()
        # position_enc = torch.tensor(position_enc, dtype=torch.float32)

        # lookup
        outputs = position_enc[position_ind]

        # masks
        if masking:
            outputs = torch.where(inputs == 0, inputs, outputs)

        return outputs.float()

    def forward(self, embeddings, src_masks, edge_fea, causality = torch.tensor([0])):
        """
        embeddings: Float tensor of shape (N, T, d_model)
        src_masks: Byte tensor of shape (N, T), where padding positions are marked with True
        """
        # Embedding scaling and positional encoding
        enc = embeddings * (self.hp.d_model ** 0.5)

        enc = enc + self.positional_encoding(enc, torch.tensor([self.hp.maxlen1]))  # Assuming positional_encoding is defined elsewhere
        enc = self.dropout(enc)

        # 计算语义相似度
        semantic_sim = (normalize_adjacency_matrix(torch.matmul(enc, enc.transpose(1, 2)) / (self.hp.d_model ** 0.5))*(~src_masks.unsqueeze(1).repeat(1, enc.size(1), 1))).unsqueeze(-1)

        edge_fea = torch.cat([edge_fea, semantic_sim], dim=-1)

        edge_fea = self.edge_layer(edge_fea)
        # Pass through each encoder layer
        count = 0
        for layer in self.enc_layers:
            enc, edge_fea = layer(enc, src_masks, edge_fea, causality, count) # Residual connection
            count += 1

        return enc

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hp):
        super(TransformerEncoderLayer, self).__init__()
        hp_dropout_rate = torch.tensor([0.1])
        self.self_attn = MultiHeadAttention(hp.d_model, hp.num_heads, hp_dropout_rate)  # Assuming MultiHeadAttention is defined elsewhere
        self.feed_forward = FeedForward(hp.d_model, hp.d_ff, hp_dropout_rate)  # Assuming FeedForward is defined elsewhere
        # self.gatedgcn = GatedGCNLayer(hp.d_model)
        self.gcn = MultiGCNLayer(hp.d_model)
        # self.gcn = MultiGATLayer(hp.d_model, hp.d_model, hp.d_model)
    def forward(self, x, src_mask, edge_fea, causality, count):
        # 根据src_mask生成adj_mask
        adj = ~src_mask.unsqueeze(1).repeat(1, x.size(1), 1)
        # 用src_mask对adj_mask进行掩码
        adj = ~src_mask.unsqueeze(-1)*adj
        # 归一化adj
        adj = normalize_adjacency_matrix(adj.float())

        if count >0:
            # x, edge_fea = self.gatedgcn(x, edge_fea, adj)
            x, edge_fea = self.gcn(x, edge_fea, adj)
        # Apply self-attention
        x = self.self_attn(q=x, k=x, v=x, mask=src_mask, causality= causality, edge_fea = edge_fea) + x

        # Apply feed forward network
        x = self.feed_forward(x) 
        return x, edge_fea