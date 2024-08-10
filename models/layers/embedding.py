import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import mlp, mlp_res
from .MultiHeadAttn import MultiHeadAttention
from .DCN import DCNV2Layer
from .Transformers import TransformerEncoder
from .deepfm import DeepFM
from .utility_funcs import pad_roll_3


class HP(nn.Module):
    def __init__(self,d_model=512, d_ff=2048, num_blocks=3, num_heads=8, maxlen1=14, maxlen2=14):
        super(HP, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.maxlen1 = maxlen1
        self.maxlen2 = maxlen2


etr_transformer_paras = HP(d_model=64, d_ff=64, num_blocks=3, num_heads=8, maxlen1=12, maxlen2=12)

class DistEmb(nn.Module):
    def __init__(self, dist_emb_size=16, DIST_BIN_SIZE=50):
        super(DistEmb, self).__init__()
        self.mat_emb_lookup = nn.Embedding(DIST_BIN_SIZE + 1, dist_emb_size)

    def forward(self, point_dist_mat, extend_point_masks):
        DIST_BIN_SIZE = 50
        IGNORE_BIN_IDX = DIST_BIN_SIZE
        boundaries = torch.tensor([i * 50 for i in range(DIST_BIN_SIZE - 1)], dtype=torch.float32).to(point_dist_mat.device)
        batch_size, graph_size = extend_point_masks.shape[0], extend_point_masks.shape[1]

        # Bucketize
        cat_dist_mat = torch.bucketize(point_dist_mat, boundaries, right=True)

        # Apply mask
        dist_mat_mask = torch.logical_or(
            extend_point_masks.view(batch_size, 1, graph_size).repeat(1, graph_size, 1),
            extend_point_masks.view(batch_size, graph_size, 1).repeat(1, 1, graph_size)
        )
        cat_dist_mat = torch.where(dist_mat_mask, torch.full_like(cat_dist_mat, IGNORE_BIN_IDX), cat_dist_mat)

        # Embedding lookup and reshape
        cat_dist_emb = self.mat_emb_lookup(cat_dist_mat)
        cat_dist_emb = cat_dist_emb.view(batch_size, graph_size, -1)

        return cat_dist_emb

class DenseMatEmb(nn.Module):
    def __init__(self, hidden_size, dist_emb_size=8, DIST_BIN_SIZE=50):
        super(DenseMatEmb, self).__init__()
        self.hidden_size = hidden_size
        # self.mat_emb_lookup = nn.Parameter(torch.Tensor(DIST_BIN_SIZE + 1, dist_emb_size))
        self.mat_emb_lookup = nn.Embedding(DIST_BIN_SIZE + 1, dist_emb_size)
        # nn.init.xavier_uniform_(self.mat_emb_lookup)
        self.mlp = mlp(input_size =dist_emb_size*196, layer_sizes=[self.hidden_size])  # Define MLP with appropriate layer sizes
        self.dist_emb_size = dist_emb_size
    def forward(self, point_dist_mat, point_masks):
        DIST_BIN_SIZE = 50
        IGNORE_BIN_IDX = DIST_BIN_SIZE
        boundaries = torch.tensor([i * 50 for i in range(DIST_BIN_SIZE - 1)], dtype=torch.float32).to(point_dist_mat.device)
        # extend_point_masks = F.pad(point_masks, (2, 0), mode='constant', value=True)
        extend_point_masks = F.pad(point_masks, (2, 0), mode='constant', value=1.0)
        batch_size, graph_size = extend_point_masks.size()
        cat_dist_mat = torch.bucketize(point_dist_mat, boundaries, right=True)
        dist_mat_mask = torch.logical_or(
            extend_point_masks.unsqueeze(1).repeat(1, graph_size, 1),
            extend_point_masks.unsqueeze(2).repeat(1, 1, graph_size))
        cat_dist_mat = torch.where(dist_mat_mask, torch.full_like(cat_dist_mat, IGNORE_BIN_IDX), cat_dist_mat)
        # cat_dist_emb = F.embedding(cat_dist_mat, self.mat_emb_lookup)
        cat_dist_emb = self.mat_emb_lookup(cat_dist_mat)
        cat_dist_emb = cat_dist_emb.view(batch_size, self.dist_emb_size * 196)
        cat_dist_emb = self.mlp(cat_dist_emb)
        cat_dist_emb = cat_dist_emb.view(batch_size, 1, self.hidden_size).repeat(1, graph_size - 2, 1)
        return cat_dist_emb

class DenseMatEmb2(nn.Module):
    def __init__(self, hidden_size, dist_emb_size=8, DIST_BIN_SIZE=50):
        super(DenseMatEmb2, self).__init__()
        self.begein = nn.Linear(14, hidden_size)
        self.res_mlp = mlp_res(hidden_size, hidden_size)
        # self.final = nn.Linear(hidden_size, 14)

    def forward(self, point_dist_mat, point_masks):
        dist_emb = self.begein(point_dist_mat)
        dist_emb = self.res_mlp(dist_emb)
        # cat_dist_emb = self.final(dist_emb)
        return dist_emb[:, 2:, :]

class DenseMatEmb3(nn.Module):
    def __init__(self, hidden_size, dist_emb_size=8, DIST_BIN_SIZE=50):
        super(DenseMatEmb3, self).__init__()
        self.begein = nn.Linear(14, hidden_size)
        self.self_attn = MultiHeadAttention(hidden_size, 8) 
        # self.final = nn.Linear(hidden_size, 14)
    def forward(self, point_dist_mat, point_masks):
        dist_emb = self.begein(point_dist_mat)
        extend_point_masks = F.pad(point_masks, (2, 0), mode='constant', value=1.0)
        dist_emb = self.self_attn(dist_emb, dist_emb, dist_emb, extend_point_masks, torch.tensor([0]), torch.tensor([0]))
        # cat_dist_emb = self.final(dist_emb)
        return dist_emb[:, 2:, :]

class SimpleConcatEmb(nn.Module):
    def __init__(self, c_sizes, c_feature_emb_size=16):
        super(SimpleConcatEmb, self).__init__()
        self.c_sizes = c_sizes
        self.embeddings = nn.ModuleList([
            nn.Embedding(dict_size, c_feature_emb_size)
            for dict_size in c_sizes
        ])

        # Initialize embeddings similar to tf.glorot_uniform_initializer
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, n_features, c_features):
        c_feature_emb_list = []
        for i, emb in enumerate(self.embeddings):
            feature = c_features[:, :, i].long()  # Ensure indices are long type
            tmp_c_feature_emb = emb(feature)
            c_feature_emb_list.append(tmp_c_feature_emb)
        concatenated_c = torch.cat(c_feature_emb_list, dim=-1).float()
        n_features = n_features.float()
        concatenated = torch.cat([concatenated_c,n_features], dim=-1)
        return concatenated
    
class ODEmb(nn.Module):
    def __init__(self, poi_id_dict_size, poi_emb_size, aoi_id_dict_size, aoi_emb_size,
                 geohash7_id_dict_size, geohash7_emb_size, output_emb_size, name=''):
        super(ODEmb, self).__init__()
        self.poi_id_emb_table = nn.Embedding(poi_id_dict_size, poi_emb_size)
        self.aoi_id_emb_table = nn.Embedding(aoi_id_dict_size, aoi_emb_size)
        self.geohash7_id_emb_table = nn.Embedding(geohash7_id_dict_size, geohash7_emb_size)
        self.output_emb_size = output_emb_size
        # Initialize embeddings
        nn.init.xavier_uniform_(self.poi_id_emb_table.weight)
        nn.init.xavier_uniform_(self.aoi_id_emb_table.weight)
        nn.init.xavier_uniform_(self.geohash7_id_emb_table.weight)

        # Assuming dcn_v2 and mlp are defined elsewhere as PyTorch modules
        self.pickup_cross = DCNV2Layer(input_dim = 128)
        self.pickup_mlp = mlp(input_size = 128, layer_sizes = [256, 128, 64])      # Define with appropriate parameters
        self.deliver_cross = DCNV2Layer(input_dim = 128) # Define with appropriate parameters
        self.deliver_mlp = mlp(input_size = 128, layer_sizes = [256, 128, 64])      # Define with appropriate parameters
        self.pickup_comb_mlp = mlp(input_size = 128, layer_sizes = [256, 128, 64])  # Define with appropriate parameters
        self.deliver_comb_mlp = mlp(input_size = 128, layer_sizes = [256, 128, 64]) # Define with appropriate parameters

    def forward(self, poi_id_features, aoi_id_features, geohash7_id_features, point_types):
        poi_id_embeddings = self.poi_id_emb_table(poi_id_features)
        aoi_id_embeddings = self.aoi_id_emb_table(aoi_id_features)
        geohash7_id_embeddings = self.geohash7_id_emb_table(geohash7_id_features)

        pickup_embeddings = torch.cat([poi_id_embeddings, aoi_id_embeddings, geohash7_id_embeddings], dim=-1)
        pickup_cross_embeddings = self.pickup_cross(pickup_embeddings)
        pickup_dnn_embeddings = self.pickup_mlp(pickup_embeddings)
        pickup_embeddings = self.pickup_comb_mlp(torch.cat([pickup_cross_embeddings, pickup_dnn_embeddings], dim=-1))

        deliver_embeddings = torch.cat([aoi_id_embeddings, geohash7_id_embeddings], dim=-1)
        deliver_cross_embeddings = self.deliver_cross(deliver_embeddings)
        deliver_dnn_embeddings = self.deliver_mlp(deliver_embeddings)
        deliver_embeddings = self.deliver_comb_mlp(torch.cat([deliver_cross_embeddings, deliver_dnn_embeddings], dim=-1))

        selection_masks = torch.repeat_interleave(point_types.unsqueeze(-1), repeats=self.output_emb_size, dim=-1)
        od_point_embeddings = torch.where(selection_masks == -1, pickup_embeddings, deliver_embeddings)

        return od_point_embeddings
    
class RPosEmb(nn.Module):
    def __init__(self, emb_size=16, dist_bin_size=101, dist_per_bin=50):
        super(RPosEmb, self).__init__()
        self.dist_bin_size = dist_bin_size
        self.boundaries = nn.Parameter(torch.tensor(
            [dist_per_bin * (i - 0.5) for i in range(-dist_bin_size // 2, dist_bin_size // 2)]
        ), requires_grad=False)
        self.r_pos_emb_table = nn.Embedding((dist_bin_size+1) * (dist_bin_size+1), emb_size)
        nn.init.xavier_uniform_(self.r_pos_emb_table.weight)  # Xavier/Glorot Uniform Initialization

    def forward(self, line_dist_mat, angle_mat):
        angle_vec = angle_mat[:, 1, :]
        dist_vec = line_dist_mat[:, 1, :]
        u_vec = dist_vec * torch.cos(angle_vec)
        v_vec = dist_vec * torch.sin(angle_vec)
        u_index_vec = torch.bucketize(u_vec, self.boundaries)
        v_index_vec = torch.bucketize(v_vec, self.boundaries)
        uv_index_vec = v_index_vec * self.dist_bin_size + u_index_vec
        uv_emb_vec = self.r_pos_emb_table(uv_index_vec)
        return uv_emb_vec
    
class RpPointEmb(nn.Module):
    def __init__(self, pickup_point_c_sizes, deliver_point_c_sizes, output_emb_size, args):
        super(RpPointEmb, self).__init__()
        self.output_emb_size = output_emb_size
        self.pickup_point_emb = DeepFM(c_sizes=pickup_point_c_sizes, fea_n = 29, fea_c = 13, output_size=output_emb_size)
        self.deliver_point_emb = DeepFM(c_sizes=deliver_point_c_sizes, fea_n = 22, fea_c = 8, output_size=output_emb_size)
        self.rp_p_mlp_emb = mlp(input_size = output_emb_size*2, layer_sizes=[output_emb_size])
        self.rp_geohash = DistEmb()
        # self.rp_line = DistEmb()
        # self.rp_rpos = RPosEmb()
        # self.rp_pre_rider_emb_mlp = mlp(input_size = 464, layer_sizes=[output_emb_size])
        # self.rp_point_unified_emb_mlp = mlp(input_size = 720, layer_sizes=[output_emb_size])
        self.rp_pre_rider_emb_mlp = mlp(input_size = 224, layer_sizes=[output_emb_size,output_emb_size])
        self.rp_point_unified_emb_mlp = mlp(input_size = 224+256, layer_sizes=[output_emb_size,output_emb_size])
        rp_transformer_paras = HP(d_model=64, d_ff=64, num_blocks=3, num_heads=8, maxlen1=14, maxlen2=14)
        rp_transformer_paras.d_model = output_emb_size
        self.args = args

        if args.ab_gcn:
            encoder_layers = nn.TransformerEncoderLayer(d_model=output_emb_size, nhead=rp_transformer_paras.num_heads, dim_feedforward=rp_transformer_paras.d_ff)
            self.rp_transformer = nn.TransformerEncoder(encoder_layers, num_layers=rp_transformer_paras.num_blocks)
        else:
            self.rp_transformer = TransformerEncoder(rp_transformer_paras)


    def forward(self, pickup_point_n_features, pickup_point_c_features, deliver_point_n_features,
                    deliver_point_c_features, geohash_dist_mat, line_dist_mat, angle_mat, point_masks, point_types, edge_fea):
        
        # Point Feature Embedding
        pickup_point_emb = self.pickup_point_emb(pickup_point_n_features, pickup_point_c_features)
        deliver_point_emb = self.deliver_point_emb(deliver_point_n_features, deliver_point_c_features)
        pad_roll_dpe = pad_roll_3(deliver_point_emb, input_dim=3, shift=-1)
        pickup_point_emb = torch.cat([pickup_point_emb, pad_roll_dpe], dim=-1)
        pickup_point_emb = self.rp_p_mlp_emb(pickup_point_emb)
        selection_masks = torch.repeat_interleave(point_types.unsqueeze(-1), repeats=self.output_emb_size, dim=-1)
        task_point_embeddings = torch.where(selection_masks == -1, pickup_point_emb, deliver_point_emb)
        
        # # Point Position Embedding
        # geohash_dist_embeddings = self.rp_geohash(geohash_dist_mat, point_masks)
        # line_dist_embeddings = self.rp_line(line_dist_mat, point_masks)
        # r_pos_embeddings = self.rp_rpos(line_dist_mat, angle_mat)

        # # Concatenation
        # # 骑手节点的embedding 是三个距离embedding的拼接
        # pre_rider_point_embeddings = self.rp_pre_rider_emb_mlp(
        #     torch.cat([geohash_dist_embeddings[:, :2], line_dist_embeddings[:, :2], r_pos_embeddings[:, :2]], dim=-1))
        # # 任务节点的embedding 是任务节点的embedding和三个距离embedding的拼接
        # task_point_embeddings = self.rp_point_unified_emb_mlp(torch.cat(
        #     [task_point_embeddings, geohash_dist_embeddings[:, 2:], line_dist_embeddings[:, 2:],
        #         r_pos_embeddings[:, 2:]], dim=-1))
        
        geohash_dist_embeddings = self.rp_geohash(geohash_dist_mat, point_masks)
        pre_rider_point_embeddings = self.rp_pre_rider_emb_mlp(geohash_dist_embeddings[:, :2])

        if self.args.ab_distance:
            geohash_dist_embeddings = torch.zeros_like(geohash_dist_embeddings).to(geohash_dist_embeddings.device)
            pre_rider_point_embeddings = torch.zeros_like(pre_rider_point_embeddings).to(pre_rider_point_embeddings.device)
        
        task_point_embeddings = self.rp_point_unified_emb_mlp(torch.cat([task_point_embeddings, geohash_dist_embeddings[:, 2:]], dim=-1))

        route_point_embeddings = torch.cat([pre_rider_point_embeddings, task_point_embeddings], dim=1)
        if self.args.ab_gcn:
            route_point_embeddings = self.rp_transformer(route_point_embeddings)
        else:
            route_point_embeddings = self.rp_transformer(route_point_embeddings, point_masks, edge_fea)
        return route_point_embeddings

class EtrPointEmb(nn.Module):
    def __init__(self, pickup_point_c_sizes, deliver_point_c_sizes, output_emb_size, c_feature_emb_size=16):
        super(EtrPointEmb, self).__init__()
        self.output_emb_size = output_emb_size
        mlp_layer_sizes = [2 * output_emb_size, output_emb_size]

        self.etr_p_emb = SimpleConcatEmb(c_sizes=pickup_point_c_sizes,c_feature_emb_size=c_feature_emb_size) # 13*16+29 = 237
        #torch.Size([2048, 12, 237])
        self.etr_p_cross = DCNV2Layer(input_dim = 237)
        self.etr_p_mlp = mlp(input_size = 237, layer_sizes = mlp_layer_sizes)      # Define with appropriate parameters
        self.etr_d_emb = SimpleConcatEmb(c_sizes=deliver_point_c_sizes, c_feature_emb_size=c_feature_emb_size) # 8*16+22 = 150
        self.etr_d_cross = DCNV2Layer(input_dim = 150) # Define with appropriate parameters
        self.etr_d_mlp = mlp(input_size = 150, layer_sizes = mlp_layer_sizes)
        
    def forward(self, pickup_point_n_features, pickup_point_c_features, deliver_point_n_features,
                    deliver_point_c_features, point_types):
        
        # Point Feature Embedding
        pickup_point_emb = self.etr_p_emb(pickup_point_n_features, pickup_point_c_features)
        pickup_point_emb = self.etr_p_cross(pickup_point_emb) #torch.Size([2048, 12, 237])
        pickup_point_emb = self.etr_p_mlp(pickup_point_emb) #torch.Size([2048, 12, 256])
        deliver_point_emb = self.etr_d_emb(deliver_point_n_features, deliver_point_c_features)
        deliver_point_emb = self.etr_d_cross(deliver_point_emb) #torch.Size([2048, 12, 150])
        deliver_point_emb = self.etr_d_mlp(deliver_point_emb)
        selection_masks = torch.repeat_interleave(point_types.unsqueeze(-1), repeats=self.output_emb_size, dim=-1)
        task_point_embeddings = torch.where(selection_masks == -1, pickup_point_emb, deliver_point_emb)

        return task_point_embeddings
    
class PointEmb(nn.Module):
    def __init__(self, context_c_sizes, wb_c_sizes, pickup_c_sizes, output_emb_size, args):
        super(PointEmb, self).__init__()
        self.pickup_point_c_sizes = context_c_sizes + wb_c_sizes + pickup_c_sizes
        self.deliver_point_c_sizes = context_c_sizes + wb_c_sizes
        self.etr = EtrPointEmb(self.pickup_point_c_sizes, self.deliver_point_c_sizes, output_emb_size)
        self.rp = RpPointEmb(self.pickup_point_c_sizes, self.deliver_point_c_sizes, output_emb_size, args)

    def forward(self, context_n_features, context_c_features, wb_c_features,
                  pickup_n_features, pickup_c_features, deliver_n_features, da_n_features,
                  geohash_dist_mat, line_dist_mat, angle_mat, point_masks, point_types, edge_fea):
        
        batch_size = wb_c_features.size(0)
        graph_size = wb_c_features.size(1)
        # Feature Concatenation
        context_n_features = torch.repeat_interleave(context_n_features.unsqueeze(1), repeats=graph_size, dim=1)
        context_c_features = torch.repeat_interleave(context_c_features.unsqueeze(1), repeats=graph_size, dim=1)
        pickup_point_n_features = torch.cat(
            [context_n_features, pickup_n_features, da_n_features], dim=-1)
        pickup_point_c_features = torch.cat([context_c_features, wb_c_features, pickup_c_features],
                                            dim=-1)
        
        deliver_point_n_features = torch.cat(
            [context_n_features, deliver_n_features, da_n_features], dim=-1)
        deliver_point_c_features = torch.cat([context_c_features, wb_c_features], dim=-1)
        etr_point_embedding = self.etr(pickup_point_n_features, pickup_point_c_features, deliver_point_n_features,
                    deliver_point_c_features, point_types) #torch.Size([2048, 12, 256])
        rp_point_embedding = self.rp(pickup_point_n_features, pickup_point_c_features, deliver_point_n_features,
                    deliver_point_c_features, geohash_dist_mat, line_dist_mat, angle_mat, point_masks, point_types, edge_fea)#torch.Size([2048, 14, 256])
        return etr_point_embedding, rp_point_embedding
    
