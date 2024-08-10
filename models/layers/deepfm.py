import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import mlp

class DeepFM(nn.Module):
    def __init__(self, c_sizes, fea_n, fea_c, c_feature_emb_size=16, output_size=128):
        super(DeepFM, self).__init__()
        self.c_feature_emb_size = c_feature_emb_size
        self.c_feature_1d_embs = nn.ModuleList([
            nn.Embedding(dict_size, 1) for dict_size in c_sizes
        ])
        self.c_feature_2d_embs = nn.ModuleList([
            nn.Embedding(dict_size, c_feature_emb_size) for dict_size in c_sizes
        ])
        self.dense_1d = nn.Linear(fea_n, 1) # Assuming n_features is a single dimensional feature

        # Initialize embeddings
        for emb in self.c_feature_1d_embs:
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.c_feature_2d_embs:
            nn.init.xavier_uniform_(emb.weight)

        # Assuming dnn and output mlp are defined elsewhere as PyTorch modules
        
        self.dnn = mlp(input_size = fea_c * c_feature_emb_size + fea_n, layer_sizes=[256, 128])       # Define with appropriate parameters
        self.output_mlp = mlp(input_size = 130, layer_sizes=[output_size]) # Define with appropriate parameters

    def forward(self, n_features, c_features):
        batch_size, graph_size = n_features.shape[0], n_features.shape[1]

        # FM 1st part
        fm_1st_sparse_res = [emb(c_features[:, :, i]) for i, emb in enumerate(self.c_feature_1d_embs)]
        fm_1st_sparse_res = torch.sum(torch.cat(fm_1st_sparse_res, dim=2), dim=2, keepdim=True) # [2048, 12, 13]->[2048, 12, 1]
        fm_1st_dense_res = self.dense_1d(n_features) #([2048, 12, 29])
        fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res #([2048, 12, 1]) + ([2048, 12, 1])

        # FM 2nd part
        fm_2nd_sparse_res = [emb(c_features[:, :, i]).view(batch_size, graph_size, 1, self.c_feature_emb_size)
                             for i, emb in enumerate(self.c_feature_2d_embs)]
        fm_2nd_concat_1d = torch.cat(fm_2nd_sparse_res, dim=2)# torch.Size([2048, 12, 13, 16])
        sum_embed = torch.sum(fm_2nd_concat_1d, dim=2)# torch.Size([2048, 12, 16])  [bs, gs, emb_size]
        square_sum_embed = sum_embed * sum_embed # [bs, gs, emb_size]
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d # [bs, gs, n, emb_size] ([2048, 12, 13, 16])
        sum_square_embed = torch.sum(square_embed, dim=2) # [bs, gs, emb_size]
        sub = (square_sum_embed - sum_square_embed) * 0.5
        fm_2nd_part = torch.sum(sub, dim=2, keepdim=True) # [bs, gs, 1]

        # DNN part
        sparse_dense_output = fm_2nd_concat_1d.view(batch_size, graph_size, -1)
        dnn_output = self.dnn(torch.cat([sparse_dense_output, n_features], dim=2))# 13*16+29 = 237 [2048, 12, 237]->([2048, 12, 128])
        point_embeddings = torch.cat([fm_1st_part, fm_2nd_part, dnn_output], dim=-1) #torch.Size([2048, 12, 130])
        final_embeddings = self.output_mlp(point_embeddings)

        return final_embeddings