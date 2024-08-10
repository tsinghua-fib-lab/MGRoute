import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DCNV2Layer(nn.Module):
    def __init__(self, input_dim, layer_num=3):
        super(DCNV2Layer, self).__init__()
        self.layer_num = layer_num
        # self.cross_weights = nn.ParameterList([
        #     nn.Parameter(init.xavier_normal_(torch.empty(input_dim, input_dim)))
        #     for _ in range(layer_num)
        # ])
        # self.cross_bias = nn.ParameterList([
        #     nn.Parameter(init.xavier_normal_(torch.empty(1,input_dim)))
        #     for _ in range(layer_num)
        # ])
        self.cross_weight_1 = nn.Parameter(init.xavier_normal_(torch.empty(input_dim, input_dim)))
        self.cross_bias_1 = nn.Parameter(init.xavier_normal_(torch.empty(1,input_dim)))
        self.cross_weight_2 = nn.Parameter(init.xavier_normal_(torch.empty(input_dim, input_dim)))
        self.cross_bias_2 = nn.Parameter(init.xavier_normal_(torch.empty(1,input_dim)))
        self.cross_weight_3 = nn.Parameter(init.xavier_normal_(torch.empty(input_dim, input_dim)))
        self.cross_bias_3 = nn.Parameter(init.xavier_normal_(torch.empty(1,input_dim)))

    def forward(self, inputs):
        outputs = inputs #torch.Size([2048, 12, 237])
        # outputs = (torch.matmul(outputs, self.cross_weight_1) + self.cross_bias_1) * inputs + outputs
        # outputs = (torch.matmul(outputs, self.cross_weight_2) + self.cross_bias_2) * inputs + outputs
        # outputs = (torch.matmul(outputs, self.cross_weight_3) + self.cross_bias_3) * inputs + outputs
        out_1_1 = torch.matmul(outputs, self.cross_weight_1)
        out_1_2 = out_1_1 + self.cross_bias_1
        out_1_3 = out_1_2 * inputs
        outputs = out_1_3 + outputs
        out_2_1 = torch.matmul(outputs, self.cross_weight_2)
        out_2_2 = out_2_1 + self.cross_bias_2
        out_2_3 = out_2_2 * inputs
        outputs = out_2_3 + outputs        
        out_3_1 = torch.matmul(outputs, self.cross_weight_3)
        out_3_2 = out_3_1 + self.cross_bias_3
        out_3_3 = out_3_2 * inputs
        outputs = out_3_3 + outputs   
        return outputs