from typing import List

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch_geometric.nn import GraphConv


class GraphConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, act: bool = True):
        super().__init__()

        self.gc = GraphConv(in_features, out_features, aggr='mean', bias=False)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, features, edges):
        out = self.gc(features, edges)
        out = self.norm(out)
        return self.act(out)


class GraphResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, graph_conv_block_num: int):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.ModuleList()

        self.layers.append(GraphConvBlock(in_features, out_features, act=True))

        for i in range(1, graph_conv_block_num):
            if i == graph_conv_block_num - 1:
                self.layers.append(GraphConvBlock(out_features, out_features, act=True))
            else:
                self.layers.append(GraphConvBlock(out_features, out_features, act=False))

        if in_features != out_features:
            self.linear = nn.Linear(in_features, out_features)

    def forward(self, features: Tensor, edges: Tensor):
        if self.in_features != self.out_features:
            residual = self.linear(features)
        else:
            residual = features

        f_out = self.layers[0](features, edges)
        for i in range(1, len(self.layers)):
            f_out = self.layers[i](f_out, edges)

        f_out = F.relu(f_out + residual)
        return f_out


class GNNBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 residual_block_num: int, graph_conv_block_num: int):
        super(GNNBlock, self).__init__()

        self.res_blocks = nn.ModuleList()
        self.res_blocks.append(GraphResidualBlock(in_features, out_features, graph_conv_block_num))
        for i in range(residual_block_num - 1):
            self.res_blocks.append(GraphResidualBlock(out_features, out_features, graph_conv_block_num))

    def forward(self, features: Tensor, edges: Tensor):
        for block in self.res_blocks:
            features = block(features, edges)
        return features


class GNN(nn.Module):
    def __init__(self,
                 channels: List[int],
                 unet_channels: List[int],
                 aggregate_indices: List[List[int]],
                 residual_block_num: int,
                 graph_conv_block_num: int,
                 voxel_shape: List[int]):
        super(GNN, self).__init__()

        self.aggregate_indices = aggregate_indices
        self.unet_channels = unet_channels

        voxel_shape = torch.from_numpy(np.array(voxel_shape, dtype=np.float32))
        self.register_buffer('voxel_shape', voxel_shape)

        self.graph_conv_first = GraphConv(6, channels[0], aggr='mean')
        self.graph_conv_last = GraphConv(channels[-1], 3, aggr='mean', bias=False)
        self.gnn_blocks = nn.ModuleList()
        for i in range(1, len(channels)):
            agg_channels = sum([unet_channels[j] for j in aggregate_indices[i - 1]])
            self.gnn_blocks.append(GNNBlock(channels[i - 1] + agg_channels * 3 + 3 + 3, channels[i],
                                            residual_block_num, graph_conv_block_num))

    @staticmethod
    def _aggregate_voxel_features(vf: Tensor, vertices: Tensor) -> Tensor:
        """
        :param vf: CxDxHxW
        :param vertices: Vx3, [-1, 1]
        """
        vf_ = vf.unsqueeze(0)
        v_ = vertices.view(1, 1, 1, -1, 3)
        v_ = torch.flip(v_, [-1])

        f = F.grid_sample(vf_, v_, mode='bilinear', padding_mode='border', align_corners=True)
        f = f.squeeze().transpose(0, 1)
        return f

    def aggregate_voxel_features(self, voxel_features: List[Tensor], vertices: Tensor, idx: int) -> Tensor:
        vertices = (2 * vertices / (self.voxel_shape - 1)) - 1.

        agg_features = [self._aggregate_voxel_features(voxel_features[j], vertices)
                        for j in self.aggregate_indices[idx]]
        agg_features = torch.concatenate(agg_features, dim=-1)
        return agg_features

    def forward(self,
                vertices: Tensor,
                edges: Tensor,
                vertex_normals: Tensor,
                voxel_features: List[Tensor],
                vector_potential: Tensor):

        edges = edges.T
        input_features = torch.cat([vertices, vector_potential], dim=-1)
        features = self.graph_conv_first(input_features, edges)

        for i, block in enumerate(self.gnn_blocks):
            agg_vertices = self.aggregate_voxel_features(voxel_features, vertices, i)
            agg_normals = self.aggregate_voxel_features(voxel_features, vertices + vertex_normals, i)
            agg_normals2 = self.aggregate_voxel_features(voxel_features, vertices - vertex_normals, i)
            agg_features = torch.cat([agg_vertices, agg_normals, agg_normals2], dim=-1)

            features = torch.cat([features, agg_features, input_features], dim=-1)
            features = block(features, edges)

        features = self.graph_conv_last(features, edges)
        return features

    def pre_compute_agg(self,
                        vertices: Tensor,
                        vertex_normals: Tensor,
                        voxel_features: List[Tensor], ):
        aggs = []
        for i in range(len(self.gnn_blocks)):
            agg_vertices = self.aggregate_voxel_features(voxel_features, vertices, i)
            agg_normals = self.aggregate_voxel_features(voxel_features, vertices + vertex_normals, i)
            agg_normals2 = self.aggregate_voxel_features(voxel_features, vertices - vertex_normals, i)
            aggs.append(torch.cat([agg_vertices, agg_normals, agg_normals2], dim=-1))

        return aggs

    def inference(self,
                  vertices: Tensor,
                  edges: Tensor,
                  vector_potential: Tensor,
                  aggs):
        edges = edges.T
        input_features = torch.cat([vertices, vector_potential], dim=-1)
        features = self.graph_conv_first(input_features, edges)

        for i, block in enumerate(self.gnn_blocks):
            features = torch.cat([features, aggs[i], input_features], dim=-1)
            features = block(features, edges)

        features = self.graph_conv_last(features, edges)
        return features


class GNNInfer(nn.Module):
    def __init__(self, gnn_model):
        super().__init__()
        self.model = gnn_model
        self.blocks = self.model.gnn_blocks

    def pre_compute_agg(self,
                        vertices: Tensor,
                        vertex_normals: Tensor,
                        voxel_features: List[Tensor], ):
        # Todo: rename to z
        aggs = []
        for i in range(len(self.model.gnn_blocks)):
            agg_vertices = self.model.aggregate_voxel_features(voxel_features, vertices, i)
            agg_normals = self.model.aggregate_voxel_features(voxel_features, vertices + vertex_normals, i)
            agg_normals2 = self.model.aggregate_voxel_features(voxel_features, vertices - vertex_normals, i)
            aggs.append(torch.cat([agg_vertices, agg_normals, agg_normals2], dim=-1))

        return aggs

    def forward(self,
                vertices: Tensor,
                edges: Tensor,
                vector_potential: Tensor,
                z):
        edges = edges.T
        input_features = torch.cat([vertices, vector_potential], dim=-1)
        features = self.model.graph_conv_first(input_features, edges)

        features = torch.cat([features, z[0], input_features], dim=-1)
        features = self.blocks[0](features, edges)
        features = torch.cat([features, z[1], input_features], dim=-1)
        features = self.blocks[1](features, edges)
        features = torch.cat([features, z[2], input_features], dim=-1)
        features = self.blocks[2](features, edges)
        features = torch.cat([features, z[3], input_features], dim=-1)
        features = self.blocks[3](features, edges)

        # for i, block in enumerate(self.blocks):
        #    features = torch.cat([features, z[i], input_features], dim=-1)
        #    features = block(features, edges)

        features = self.model.graph_conv_last(features, edges)
        return features
