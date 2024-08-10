import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """
    Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    """
    def __init__(self, features, epsilon=1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # 最后一维度取平均
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=torch.tensor([0.1])):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.dropout = nn.Dropout(dropout_rate.item())

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)
        self.dropout_rate = dropout_rate
        self.layer_norm = nn.LayerNorm(d_model)
        # self.edge_fea = mlp(7,[64,1])

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def scaled_dot_product_attention(self, Q, K, V, key_masks, edge_fea, causality=torch.tensor([0]), dropout_rate=torch.tensor([0])):

        d_k = Q.size(-1)

        # dot product
        outputs = torch.matmul(Q, K.transpose(-2, -1))  # (N, T_q, T_k)

        # edge_a = self.edge_fea(edge_fea).squeeze(-1)
        # # 第一维度扩展self.num_heads倍
        # edge_a = edge_a.repeat(self.num_heads, 1, 1)
        # # scale
        # outputs /= d_k ** 0.5 + edge_a
        outputs /= d_k ** 0.5

        # key masking
        outputs = self.mask(outputs, key_masks=key_masks, type=torch.tensor([1]))

        # causality or future blinding masking
        if causality:
            outputs = self.mask(outputs, key_masks=torch.tensor([0]), type=torch.tensor([0]))

        # softmax
        outputs = F.softmax(outputs, dim=-1)
        if self.training:
            outputs = F.dropout(outputs, p=dropout_rate.item())

        # weighted sum (context vectors)
        outputs = torch.matmul(outputs, V)  # (N, T_q, d_v)

        return outputs

    def mask(self, inputs, key_masks, type=torch.tensor([1])):
        """
        Masks paddings on keys or queries to inputs
        inputs: 3d tensor. (h*N, T_q, T_k)
        key_masks: 3d tensor. (N, 1, T_k)
        type: string. "key" | "future"
        e.g.,
        >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
        >> key_masks = tf.constant([[0., 0., 1.],
                                    [0., 1., 1.]])
        >> mask(inputs, key_masks=key_masks, type="key")
        array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
            [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
        [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
            [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],
        [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
            [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
        [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
            [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
        """
        # if inputs.dim() == 4:
        #     s1, s2, s3, s4 = inputs.size()
        #     inputs = inputs.reshape(s1 * s2, s3, s4)
        # key_masks = key_masks.unsqueeze(1)
        padding_num = -2 ** 32 + 1
        # if type in ("k", "key", "keys"):
        if type:
            # Key Masking
            key_masks = key_masks.to(torch.float32)
            # key_masks = key_masks.repeat_interleave(inputs.size(0) // key_masks.size(0), dim=1)  # (h*N, seqlen)
            # #  repeat_interleave(..., dim=0): repeat_interleave方法沿着指定的维度（在这里是第0维，即批次维度）重复每个元素指定次数。
            # #  在这个例子中，每个元素在key_masks中被重复inputs.size(0) // key_masks.size(0)次。
            # key_masks = key_masks.reshape(-1, key_masks.size(-1))  # (h*N, seq_len_q, seq_len_k)
            # key_masks = key_masks.unsqueeze(1)  # (h*N, 1, seqlen)

            # 计算需要重复的次数
            repeat_times = inputs.size(0) // key_masks.size(0)

            # 使用 repeat_interleave 进行复制
            key_masks = key_masks.repeat_interleave(repeat_times, dim=0)  # (h*N, seqlen)

            # 使用 unsqueeze 增加一个维度
            key_masks = key_masks.unsqueeze(1) 

            outputs = inputs + key_masks * padding_num
            
        # elif type in ("f", "future", "right"):
        else:
            # # Future Masking
            # T_q, T_k = inputs.size(1), inputs.size(2)
            # mask = torch.tril(torch.ones(T_q, T_k, device=inputs.device)).unsqueeze(0)
            # outputs = inputs.masked_fill(mask == 0, padding_num)
            # 创建一个与 inputs 第一个元素形状相同的全1张量
            diag_vals = torch.ones_like(inputs[0, :, :]).to(inputs.device)  # (T_q, T_k)

            # 使用 torch.tril 创建一个下三角矩阵
            tril = torch.tril(diag_vals)  # (T_q, T_k)

            # 使用 unsqueeze 和 repeat 扩展并复制下三角矩阵
            future_masks = tril.unsqueeze(0).repeat(inputs.size(0), 1, 1)  # (N, T_q, T_k)

            # 创建一个与 future_masks 形状相同的、填充了 padding_num 的张量
            paddings = torch.ones_like(future_masks).to(inputs.device) * padding_num

            # 使用 torch.where 实现条件选择
            outputs = torch.where(future_masks == 0, paddings, inputs)
        # else:
        #     raise ValueError("Check if you entered type correctly!")
        # if inputs.dim() == 4:
        #     outputs = outputs.reshape(s1, s2, s3, s4)
        return outputs

    def forward(self, q, k, v, mask, causality, edge_fea):
        # batch_size = torch.tensor([q.size(0)])

        # Linear layers
        q_ = self.wq(q)  # (batch_size, seq_len, d_model)
        k_ = self.wk(k)  # (batch_size, seq_len, d_model)
        v_ = self.wv(v)  # (batch_size, seq_len, d_model)

        # Split heads
        q_ = torch.cat(torch.chunk(q_, self.num_heads, dim=2), dim=0)  # (batch_size, num_heads, seq_len_q, depth)
        k_ = torch.cat(torch.chunk(k_, self.num_heads, dim=2), dim=0)   # (batch_size, num_heads, seq_len_k, depth)
        v_ = torch.cat(torch.chunk(v_, self.num_heads, dim=2), dim=0)   # (batch_size, num_heads, seq_len_v, depth)

        # Scaled dot-product attention
        scaled_attention = self.scaled_dot_product_attention(q_, k_, v_, mask, edge_fea, causality, self.dropout_rate)

        # Concat attention heads
        outputs = torch.cat(torch.chunk(scaled_attention, self.num_heads, dim=0), dim=2)  # (batch_size, seq_len_v, num_heads, depth)

        # Reshape to (batch_size, seq_len_v, d_model)
        # concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        

        # Final linear layer
        # outputs = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)
        outputs = self.dropout(outputs)
        # Residual connection and normalization
        outputs += q
        outputs = self.layer_norm(outputs)  # (batch_size, seq_len_v, d_model)

        return outputs

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=torch.tensor([0.1])):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate.item())
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        outputs = self.w_1(inputs)
        outputs = F.relu(outputs)
        outputs = self.w_2(outputs)
        outputs = self.dropout(outputs)
        outputs += inputs
        outputs = self.layer_norm(outputs)
        return outputs