# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
Modified by Yuanhui Huang.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import (
    NeRFEncoding,
    PeriodicVolumeEncoding,
    TensorVMEncoding,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, FieldConfig
from nerfstudio.fields import softsplat
try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass

# import .cuda_gridsample_grad2.cuda_gridsample as cudagrid
# from . import cuda_gridsample_grad2.cuda_gridsample as cudagrid
from .cuda_gridsample_grad2 import cuda_gridsample as cudagrid
from .mappings import GridMeterMapping
from .sh_render import SHRender
from .utils import sample_from_2d_img_feats
from mmcv.cnn.bricks.conv_module import ConvModule
# class GridMeterMapping:
#     def __init__(
#         self,
#         bev_inner=128,
#         bev_outer=32,
#         range_inner=51.2,
#         range_outer=51.2,
#         nonlinear_mode="linear_upscale",
#         z_inner=20,
#         z_outer=10,
#         z_ranges=[-5.0, 3.0, 11.0],
#     ) -> None:
#         self.bev_inner = bev_inner
#         self.bev_outer = bev_outer
#         self.range_inner = range_inner
#         self.range_outer = range_outer
#         assert nonlinear_mode == "linear_upscale"  # TODO
#         self.nonlinear_mode = nonlinear_mode
#         self.z_inner = z_inner
#         self.z_outer = z_outer
#         self.z_ranges = z_ranges

#         self.hw_unit = range_inner * 1.0 / bev_inner
#         self.increase_unit = (range_outer - bev_outer * self.hw_unit) * 2.0 / bev_outer / (bev_outer + 1)

#         self.z_unit = (z_ranges[1] - z_ranges[0]) * 1.0 / z_inner
#         self.z_increase_unit = (z_ranges[2] - z_ranges[1] - z_outer * self.z_unit) * 2.0 / z_outer / (z_outer + 1)

#     def grid2meter(self, grid):
#         hw = grid[..., :2]
#         hw_center = hw - (self.bev_inner + self.bev_outer)
#         hw_center_abs = torch.abs(hw_center)
#         yx_base_abs = hw_center_abs * self.hw_unit
#         hw_outer = torch.relu(hw_center_abs - self.bev_inner)
#         hw_outer_int = torch.floor(hw_outer)
#         yx_outer_base = hw_outer_int * (hw_outer_int + 1) / 2.0 * self.increase_unit
#         yx_outer_resi = (hw_outer - hw_outer_int) * (hw_outer_int + 1) * self.increase_unit
#         yx_abs = yx_base_abs + yx_outer_base + yx_outer_resi
#         yx = torch.sign(hw_center) * yx_abs

#         if grid.shape[-1] == 3:
#             d = grid[..., 2:3]
#             d_center = d
#             z_base = d_center * self.z_unit

#             d_outer = torch.relu(d_center - self.z_inner)
#             d_outer_int = torch.floor(d_outer)
#             z_outer_base = d_outer_int * (d_outer_int + 1) / 2.0 * self.z_increase_unit
#             z_outer_resi = (d_outer - d_outer_int) * (d_outer_int + 1) * self.z_increase_unit
#             z = z_base + z_outer_base + z_outer_resi + self.z_ranges[0]

#             return torch.cat([yx[..., 1:2], yx[..., 0:1], z], dim=-1)
#         else:
#             return yx[..., [1, 0]]

#     def meter2grid(self, meter):
#         xy = meter[..., :2]
#         xy_abs = torch.abs(xy)
#         wh_base_abs = xy_abs / self.hw_unit
#         wh_base_abs = wh_base_abs.clamp_(max=self.bev_inner)
#         xy_outer_abs = torch.relu(xy_abs - self.range_inner)

#         wh_outer_base = torch.sqrt(
#             (1.0 / 2 + self.hw_unit / self.increase_unit) ** 2 + 2 * xy_outer_abs / self.increase_unit
#         ) - (1.0 / 2 + self.hw_unit / self.increase_unit)
#         wh_outer_base = torch.floor(wh_outer_base)
#         xy_outer_resi = (
#             xy_outer_abs - wh_outer_base * self.hw_unit - self.increase_unit * wh_outer_base * (wh_outer_base + 1) / 2
#         )
#         wh_outer_resi = xy_outer_resi / (self.hw_unit + (wh_outer_base + 1) * self.increase_unit)
#         wh_center_abs = wh_base_abs + wh_outer_base + wh_outer_resi
#         wh_center = torch.sign(xy) * wh_center_abs
#         wh = wh_center + self.bev_inner + self.bev_outer

#         z = meter[..., 2:3]
#         z_abs = z - self.z_ranges[0]
#         d_base = z_abs / self.z_unit
#         d_base = d_base.clamp_(max=self.z_inner)
#         z_outer = torch.relu(z_abs - (self.z_ranges[1] - self.z_ranges[0]))

#         d_outer_base = torch.sqrt(
#             (1.0 / 2 + self.z_unit / self.z_increase_unit) ** 2 + 2 * z_outer / self.z_increase_unit
#         ) - (1.0 / 2 + self.z_unit / self.z_increase_unit)
#         d_outer_base = torch.floor(d_outer_base)
#         z_outer_resi = (
#             z_outer - d_outer_base * self.z_unit - self.z_increase_unit * d_outer_base * (d_outer_base + 1) / 2
#         )
#         d_outer_resi = z_outer_resi / (self.z_unit + (d_outer_base + 1) * self.z_increase_unit)
#         d = d_base + d_outer_base + d_outer_resi

#         return torch.cat([wh[..., 1:2], wh[..., 0:1], d], dim=-1)


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(
        self, sdf: TensorType["bs":...], beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SigmoidDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Sigmoid density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(
        self, sdf: TensorType["bs":...], beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta

        # negtive sdf will have large density
        return alpha * torch.sigmoid(-sdf * alpha)

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SingleVarianceNetwork(nn.Module):
    """Variance network in NeuS

    Args:
        nn (_type_): init value in NeuS variance network
    """

    def __init__(self, init_val, learnable=True):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter("variance", nn.Parameter(init_val * torch.ones(1), requires_grad=learnable))

    def forward(self, x):
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)

    def get_variance(self):
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


@dataclass
class SDFCustomFieldConfig(FieldConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: SDFCustomField)
    num_layers: int = 8
    """Number of layers for geometric network"""
    hidden_dim: int = 256
    """Number of hidden dimension of geometric network"""
    geo_feat_dim: int = 256
    """Dimension of geometric feature"""
    num_layers_color: int = 4
    """Number of layers for color network"""
    hidden_dim_color: int = 256
    """Number of hidden dimension of color network"""
    appearance_embedding_dim: int = 32
    """Dimension of appearance embedding"""
    use_appearance_embedding: bool = False
    """Dimension of appearance embedding"""
    bias: float = 0.8
    """sphere size of geometric initializaion"""
    geometric_init: bool = True
    """Whether to use geometric initialization"""
    inside_outside: bool = True
    """whether to revert signed distance value, set to True for indoor scene"""
    weight_norm: bool = True
    """Whether to use weight norm for linear laer"""
    use_grid_feature: bool = False
    """Whether to use multi-resolution feature grids"""
    divide_factor: float = 2.0
    """Normalization factor for multi-resolution grids"""
    beta_init: float = 0.1
    """Init learnable beta value for transformation of sdf to density"""
    encoding_type: Literal["hash", "periodic", "tensorf_vm"] = "hash"
    """feature grid encoding type"""
    position_encoding_max_degree: int = 6
    """positional encoding max degree"""
    use_diffuse_color: bool = False
    """whether to use diffuse color as in ref-nerf"""
    use_specular_tint: bool = False
    """whether to use specular tint as in ref-nerf"""
    use_reflections: bool = False
    """whether to use reflections as in ref-nerf"""
    use_n_dot_v: bool = False
    """whether to use n dot v as in ref-nerf"""
    rgb_padding: float = 0.001
    """Padding added to the RGB outputs"""
    off_axis: bool = False
    """whether to use off axis encoding from mipnerf360"""
    use_numerical_gradients: bool = False
    """whether to use numercial gradients"""
    num_levels: int = 16
    """number of levels for multi-resolution hash grids"""
    max_res: int = 2048
    """max resolution for multi-resolution hash grids"""
    base_res: int = 16
    """base resolution for multi-resolution hash grids"""
    log2_hashmap_size: int = 19
    """log2 hash map size for multi-resolution hash grids"""
    hash_features_per_level: int = 2
    """number of features per level for multi-resolution hash grids"""
    hash_smoothstep: bool = True
    """whether to use smoothstep for multi-resolution hash grids"""
    use_position_encoding: bool = True
    """whether to use positional encoding as input for geometric network"""

    mapping_args: Dict[str, Any] = to_immutable_dict(
        dict(
            nonlinear_mode="linear_upscale",
            h_size=[128, 32],
            h_range=[51.2, 28.8],
            h_half=False,
            w_size=[128, 32],
            w_range=[51.2, 28.8],
            w_half=False,
            d_size=[20, 10],
            d_range=[-4.0, 4.0, 12.0],
        )
    )

    # bev_inner: int = 128
    # bev_outer: int = 32
    # range_inner: float = 51.2
    # range_outer: float = 51.2
    # nonlinear_mode: str = "linear_upscale"
    # z_inner: int = 20
    # z_outer: int = 10
    # z_ranges: List[float] = field(default_factory=lambda: [-5.0, 3.0, 11.0])
    # mlp decoder
    embed_dims: int = 128
    feat_dims: int = 128
    color_dims: int = 0
    density_layers: int = 2
    sh_deg: int = 2
    sh_act: str = "relu"

    beta_learnable: bool = True
    second_derivative: bool = False
    representation: str = 'tpv'

    nbr_gradient_points: int = 128 * 128 * 16
    use_uniform_gradient: bool = False

    calculate_online: bool = False
    sample_gradient: bool = False
    use_compact_2nd_grad: bool = False

    using_2d_img_feats: bool = False
    estimate_flow: bool = False

    return_sem: bool = False


class SDFCustomField(Field):
    """_summary_

    Args:
        Field (_type_): _description_
    """

    config: SDFCustomFieldConfig

    def __init__(
        self,
        config: SDFCustomFieldConfig,
        aabb,
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()

        self.config = config

        # custom configs
        # self.z_size = self.config.z_inner + self.config.z_outer + 1
        # self.bev_size = 2 * (self.config.bev_inner + self.config.bev_outer) + 1
        self.mapping = GridMeterMapping(
            **self.config.mapping_args
        )
        self.z_size = self.mapping.size_d
        self.h_size = self.mapping.size_h
        self.w_size = self.mapping.size_w
        try:
            self.z_sdf, self.h_sdf, self.w_sdf = self.mapping.size_d_sdf, self.mapping.size_h_sdf, self.mapping.size_w_sdf
        except:
            self.z_sdf, self.h_sdf, self.w_sdf = self.z_size, self.h_size, self.w_size
        self.offset_x, self.offset_y, self.offset_z = (self.w_size-self.w_sdf)//2, (self.h_size-self.h_sdf)//2, (self.z_size-self.z_sdf)//2
        self.volume_pad = nn.ConstantPad3d((self.offset_z, self.offset_z, self.offset_x, self.offset_x, self.offset_y, self.offset_y), 0)
        # x = torch.linspace(0, self.w_size - 1, self.w_size)
        # y = torch.linspace(0, self.h_size - 1, self.h_size)
        # z = torch.linspace(0, self.z_size - 1, self.z_size)
        # Z, X, Y = torch.meshgrid(z, x, y)
        # fore_mask = (X>=self.offset_x) & (X<occ_X-self.offset_x) & (Y>=offset_y) & (Y<occ_Y-offset_y) & (Z>=offset_z) & (Z<occ_Z-offset_z)
        self.color_converter = SHRender
        self.sh_deg = self.config.sh_deg
        self.sh_act = self.config.sh_act
        self.density_color = None

        # TODO do we need aabb here?
        self.aabb = Parameter(aabb, requires_grad=False)

        self.spatial_distortion = spatial_distortion

        # MLP with geometric initialization
        self.density_layers = self.config.density_layers
        self.embed_dims = self.config.embed_dims
        self.feat_dims = self.config.feat_dims
        self.color_dims = self.config.color_dims
        density_net = []
        for i in range(self.density_layers - 1):
            density_net.extend([nn.Softplus(), nn.Linear(self.embed_dims, self.embed_dims)])
        if self.config.representation == 'bev' and not self.config.using_2d_img_feats:
            density_net.extend([nn.Softplus(), nn.Linear(self.embed_dims, (1 + self.color_dims) * self.z_size)])
            nn.init.normal_(
                density_net[-1].weight[range(0, (1 + self.color_dims) * self.z_size, 1 + self.color_dims)], 0, 0.0001
            )
            d_grids = torch.arange(self.z_size, dtype=torch.float)
            hwd_grids = torch.cat([torch.zeros(self.z_size, 2), d_grids.unsqueeze(-1)], dim=-1)
            meters = self.mapping.grid2meter(hwd_grids)
            z_meters = meters[:, 2]
            sdf_init = z_meters + 1.6
            # sdf_init = sdf_init / (self.config.z_ranges[2] - self.config.z_ranges[0]) * 3
            density_net[-1].bias[range(0, (1 + self.color_dims) * self.z_size, 1 + self.color_dims)].data = sdf_init
        else:
            density_net.extend([nn.Softplus(), nn.Linear(self.embed_dims, 1 + self.color_dims)])
        density_net = nn.Sequential(*density_net)
        self.density_net = density_net
        
        background_net = []
        for i in range(self.density_layers - 1):
            background_net.extend([nn.Softplus(), nn.Linear(self.embed_dims, self.embed_dims)])
        background_net.extend([nn.Softplus(), nn.Linear(self.embed_dims, 1 + self.color_dims)])
        background_net = nn.Sequential(*background_net)
        self.background_net = background_net
        
        self.conv = ConvModule(self.feat_dims ,self.embed_dims,kernel_size=1,stride=1,padding=1,bias=False,conv_cfg=dict(type='Conv2d'),
                                norm_cfg=dict(type='BN', ), act_cfg=dict(type='ReLU',inplace=True))
        # laplace function for transform sdf to density from VolSDF
        # self.laplace_density = LaplaceDensity(init_val=self.config.beta_init)
        self.laplace_density = nn.Identity()
        # self.laplace_density = SigmoidDensity(init_val=self.config.beta_init)

        # TODO use different name for beta_init for config
        # deviation_network to compute alpha from sdf from NeuS
        self.deviation_network = SingleVarianceNetwork(
            init_val=self.config.beta_init, learnable=self.config.beta_learnable
        )

        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self._cos_anneal_ratio = 1.0
        self.numerical_gradients_delta = 0.0001

        if self.config.sample_gradient:
            assert not self.config.calculate_online
            self.pad = nn.ReplicationPad3d(1)

        if self.config.estimate_flow:
            flow_net = []
            flow_net.extend([nn.Linear(self.embed_dims, self.embed_dims), nn.Softplus()])
            flow_net.extend([nn.Linear(self.embed_dims, self.embed_dims), nn.Softplus()])
            flow_net.extend([nn.Linear(self.embed_dims, 2)])
            nn.init.normal_(flow_net[-1].weight.data, 0., 1e-2)
            nn.init.constant_(flow_net[-1].bias.data, 0.)
            self.flow_net = nn.Sequential(*flow_net)
            
        if self.config.using_2d_img_feats:
            assert not self.config.calculate_online
            assert self.config.representation in ['bev', 'tpv', 'volume']
            grid_coords = torch.stack(
                [
                    torch.arange(self.h_size, dtype=torch.float)[:, None, None].expand(-1, self.w_size, self.z_size),
                    torch.arange(self.w_size, dtype=torch.float)[None, :, None].expand(self.h_size, -1, self.z_size),
                    torch.arange(self.z_size, dtype=torch.float)[None, None, :].expand(self.h_size, self.w_size, -1),
                ],
                dim=-1,
            ).flatten(0, 2)
            meter_coords = self.mapping.grid2meter(grid_coords)  # hwd, 3
            self.register_buffer("sampling_points_2d", meter_coords, False)

            ## build UpsampleFPN
            from mmdet3d.models.necks import SECONDFPN

            self.img_skip_model = SECONDFPN(
                in_channels=[self.feat_dims] * 4,
                out_channels=[self.embed_dims] * 4,
                upsample_strides=[1, 2, 4, 8],
                final_conv_feature_dim=self.embed_dims
            )

    def set_cos_anneal_ratio(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    # def update_mask(self, level: int):
    #     self.hash_encoding_mask[:] = 1.0
    #     self.hash_encoding_mask[level * self.features_per_level :] = 0

    def pre_compute_density_color(
        self,
        bev,
        dtype=torch.float,
        img_feats=None,
        img_metas=None,
    ):
        self.density_color_bg = None
        if self.config.using_2d_img_feats:
            assert img_feats is not None and img_metas is not None
            bs, num_cams = img_feats[0].shape[0:2]
            img_feats = [img_feat.flatten(0, 1) for img_feat in img_feats]
            img_feats = self.img_skip_model(img_feats)
            img_feats = img_feats[0].unflatten(0, [bs, num_cams])
            img_feats_3d = sample_from_2d_img_feats(img_feats, img_metas, self.sampling_points_2d)
            img_feats_3d = img_feats_3d.unflatten(1, [self.h_size, self.w_size, self.z_size])
        if self.config.calculate_online:
            self.density_color = torch.cat(bev, dim=1) if self.config.representation == 'tpv' else bev.clone()
            return
        if self.config.representation == 'bev':
            assert bev.dim() == 3
            device = bev.device
            bs = bev.size(0)
            bev = bev.unflatten(1, (self.h_sdf-1, self.w_sdf-1)).permute(0,3,1,2)
            # bev = self.conv(bev)
            x = torch.linspace(-1, 1, self.w_sdf)
            y = torch.linspace(-1, 1, self.h_sdf)
            Y, X = torch.meshgrid(y, x)
            vertices = torch.stack([X, Y], dim=-1).to(device)
            real_size = torch.FloatTensor([self.w_sdf-1, self.h_sdf-1]).to(device)
            verts_norm = vertices * real_size/(real_size-1)
            bev = cudagrid.grid_sample_2d(bev, verts_norm[None,...].repeat(bs,1,1,1), padding_mode='border', align_corners=True).permute(0,2,3,1)
            if self.config.using_2d_img_feats:
                volume = bev.unsqueeze(3) + img_feats_3d[:,self.offset_y:self.h_size-self.offset_y,self.offset_x:self.w_size-self.offset_x,self.offset_z:self.z_size-self.offset_z,:]
                self.density_color_bg = self.background_net(img_feats_3d).permute(0, 4, 1, 2, 3)
                density_color = self.density_net(volume).permute(0, 4, 1, 2, 3)
            else:
                density_color = self.density_net(bev).reshape(*bev.shape[:-1], self.z_size, -1)
                density_color = density_color.permute(0, 4, 1, 2, 3)  # bs, C, h, w, d
        elif self.config.representation == 'tpv':
            if isinstance(bev, list):
                tpv_hw, tpv_zh, tpv_wz = bev
                tpv_hw = tpv_hw.reshape(-1, self.h_size, self.w_size, 1, self.embed_dims)
                tpv_hw = tpv_hw.expand(-1, -1, -1, self.z_size, -1)

                tpv_zh = tpv_zh.reshape(-1, self.z_size, self.h_size, 1, self.embed_dims).permute(0, 2, 3, 1, 4)
                tpv_zh = tpv_zh.expand(-1, -1, self.w_size, -1, -1)

                tpv_wz = tpv_wz.reshape(-1, self.w_size, self.z_size, 1, self.embed_dims).permute(0, 3, 1, 2, 4)
                tpv_wz = tpv_wz.expand(-1, self.h_size, -1, -1, -1)

                tpv = tpv_hw + tpv_zh + tpv_wz
            else:
                tpv = bev
            bs, volume_h, volume_w, volume_z, dim = tpv.size()
            device = tpv.device
            if self.w_sdf==volume_w+1 and self.h_sdf==volume_h+1 and self.z_sdf==volume_z+1:
                x = torch.linspace(-1, 1, self.w_sdf)
                y = torch.linspace(-1, 1, self.h_sdf)
                z = torch.linspace(-1, 1, self.z_sdf)
                Y, X, Z = torch.meshgrid(y, x, z)
                vertices = torch.stack([Z, X, Y], dim=-1).to(device)
                real_size = torch.FloatTensor([volume_z, volume_w, volume_h]).to(device)
                verts_norm = vertices * real_size/(real_size-1)
                tpv = cudagrid.grid_sample_3d(tpv.permute(0,4,1,2,3), verts_norm[None,...].repeat(bs,1,1,1,1), padding_mode='border', align_corners=True).permute(0,2,3,4,1)
            ## TODO: temporarily using addition to fuse
            if self.config.using_2d_img_feats:
                tpv = tpv + img_feats_3d
            density_color = self.density_net(tpv).permute(0, 4, 1, 2, 3)
            if self.config.estimate_flow:
                # next_time = [img_metas[i].get('next_time', 0.5) for i in range(len(img_metas))] if self.training else [0.2]*len(img_metas)
                # next_time = torch.tensor(next_time,dtype=tpv.dtype, device=tpv.device)[:,None,None,None,None].expand(-1, self.h_sdf, self.w_sdf, self.z_sdf,-1)
                # # only support bs=1 now
                # self.fwd_flow = self.flow_net(torch.cat((tpv,next_time),dim=-1))
                # self.fwd_flow = self.fwd_flow.permute(0, 4, 1, 2, 3)
                # prev_time = torch.tensor([img_metas[i].get('prev_time', -0.5) for i in range(len(img_metas))],dtype=tpv.dtype, device=tpv.device)[:,None,None,None,None].expand(-1, self.h_sdf, self.w_sdf, self.z_sdf,-1)
                # self.bwd_flow = self.flow_net(torch.cat((tpv,prev_time),dim=-1))
                # self.bwd_flow = self.bwd_flow.permute(0, 4, 1, 2, 3)
                self.fwd_flow = self.flow_net(tpv).permute(0, 4, 1, 2, 3)
        elif self.config.representation == 'volume':
            bs, volume_h, volume_w, volume_z, dim = bev.size()
            device = bev.device
            x = torch.linspace(-1, 1, volume_w+1)
            y = torch.linspace(-1, 1, volume_h+1)
            z = torch.linspace(-1, 1, volume_z+1)
            Y, X, Z = torch.meshgrid(y, x, z)
            vertices = torch.stack([Z, X, Y], dim=-1).to(device)
            real_size = torch.FloatTensor([volume_z, volume_w, volume_h]).to(device)
            verts_norm = vertices * real_size/(real_size-1)
            volume = cudagrid.grid_sample_3d(bev.permute(0,4,1,2,3), verts_norm[None,...].repeat(bs,1,1,1,1), padding_mode='border', align_corners=True).permute(0,2,3,4,1)
            if self.config.using_2d_img_feats:
                volume = volume + img_feats_3d[:,self.offset_y:self.h_size-self.offset_y+1,self.offset_x:self.w_size-self.offset_x+1,self.offset_z:self.z_size-self.offset_z+1,:]
                self.density_color_bg = self.background_net(img_feats_3d).permute(0, 4, 1, 2, 3)
            density_color = self.density_net(volume).permute(0, 4, 1, 2, 3)    
            
        self.density_color = density_color  # .to(dtype)

        if self.config.sample_gradient:
            sdf = self.density_color[:, :1, ...].clone()
            self.gradients = self.first_order_derivative(sdf)  # bs, 3, h, w, d

            if self.config.second_derivative:
                g2_x = self.first_order_derivative(self.gradients[:, :1, ...].clone())
                g2_y = self.first_order_derivative(self.gradients[:, 1:2, ...].clone())
                g2_z = self.first_order_derivative(self.gradients[:, 2:, ...].clone())

                if self.config.use_compact_2nd_grad:
                    g2_xx, g2_yy, g2_zz = self.second_order_derivative(sdf=sdf)
                    g2_x[:, 0, ...] = g2_xx
                    g2_y[:, 1, ...] = g2_yy
                    g2_z[:, 2, ...] = g2_zz
                # else:
                #     g2_x = self.first_order_derivative(self.gradients[:, :1, ...].clone())
                #     g2_y = self.first_order_derivative(self.gradients[:, 1:2, ...].clone())
                #     g2_z = self.first_order_derivative(self.gradients[:, 2:, ...].clone())
                self.gradients2 = torch.stack([g2_x, g2_y, g2_z], dim=1)  # bs, 3, 3, h, w, d

    def first_order_derivative(self, sdf):
        # sdf: bs, 1, h, w, d
        sdf = self.pad(sdf)
        grad_x = sdf[:, :, 1:-1, 2:, 1:-1] - sdf[:, :, 1:-1, :-2, 1:-1]
        grad_x = grad_x / (self.aabb[1, 0] - self.aabb[0, 0]) * (self.w_size - 1) / 2

        grad_y = sdf[:, :, 2:, 1:-1, 1:-1] - sdf[:, :, :-2, 1:-1, 1:-1]
        grad_y = grad_y / (self.aabb[1, 1] - self.aabb[0, 1]) * (self.h_size - 1) / 2

        grad_z = sdf[:, :, 1:-1, 1:-1, 2:] - sdf[:, :, 1:-1, 1:-1, :-2]
        grad_z = grad_z / (self.aabb[1, 2] - self.aabb[0, 2]) * (self.z_size - 1) / 2
        gradients = torch.cat([grad_x, grad_y, grad_z], dim=1)  # bs, 3, h, w, d
        return gradients

    def second_order_derivative(self, sdf):
        # sdf: bs, 1, h, w, d
        # grad_2 = sdf.new_zeros(sdf.shape[0], 3, 3, *sdf.shape[2:])
        sdf = self.pad(sdf)
        x_1 = sdf[:, :, :, 1:, :] - sdf[:, :, :, :-1, :]
        x_1 = x_1 / ((self.aabb[1, 0] - self.aabb[0, 0]) / (self.w_size - 1))
        x_2 = x_1[:, :, :, 1:, :] - x_1[:, :, :, :-1, :]
        x_2 = x_2 / ((self.aabb[1, 0] - self.aabb[0, 0]) / (self.w_size - 1))
        y_1 = sdf[:, :, 1:, :, :] - sdf[:, :, :-1, :, :]
        y_1 = y_1 / ((self.aabb[1, 1] - self.aabb[0, 1]) / (self.h_size - 1))
        y_2 = y_1[:, :, 1:, :, :] - y_1[:, :, :-1, :, :]
        y_2 = y_2 / ((self.aabb[1, 1] - self.aabb[0, 1]) / (self.h_size - 1))
        z_1 = sdf[:, :, :, :, 1:] - sdf[:, :, :, :, :-1]
        z_1 = z_1 / ((self.aabb[1, 2] - self.aabb[0, 2]) / (self.z_size - 1))
        z_2 = z_1[:, :, :, :, 1:] - z_1[:, :, :, :, :-1]
        z_2 = z_2 / ((self.aabb[1, 2] - self.aabb[0, 2]) / (self.z_size - 1))
        # grad_2[:, 0, 0] = x_2[:, 0, 1:-1, :, 1:-1]
        # grad_2[:, 1, 1] = y_2[:, 0, :, 1:-1, 1:-1]
        # grad_2[:, 2, 2] = z_2[:, 0, 1:-1, 1:-1, :]
        return x_2[:, 0, 1:-1, :, 1:-1], y_2[:, 0, :, 1:-1, 1:-1], z_2[:, 0, 1:-1, 1:-1, :]

    def forward_geonetwork_online(self, inputs):
        grid = self.mapping.meter2grid(inputs, True)
        grid = grid[None, None, ...] * 2 - 1

        if self.config.representation == 'tpv':
            tpv_hw, tpv_zh, tpv_wz = self.density_color.split(
                [self.h_size * self.w_size, self.z_size * self.h_size, self.w_size * self.z_size], dim=1
            )
            tpv_hw = tpv_hw.reshape(-1, self.h_size, self.w_size, self.embed_dims).permute(0, 3, 1, 2)
            tpv_zh = tpv_zh.reshape(-1, self.z_size, self.h_size, self.embed_dims).permute(0, 3, 1, 2)
            tpv_wz = tpv_wz.reshape(-1, self.w_size, self.z_size, self.embed_dims).permute(0, 3, 1, 2)
            tpv_hw = (
                cudagrid.grid_sample_2d(tpv_hw, grid[..., [1, 0]], align_corners=True, padding_mode="border")
                .reshape(self.embed_dims, inputs.shape[0])
                .transpose(0, 1)
            )  # n, c
            tpv_zh = (
                cudagrid.grid_sample_2d(tpv_zh, grid[..., [0, 2]], align_corners=True, padding_mode="border")
                .reshape(self.embed_dims, inputs.shape[0])
                .transpose(0, 1)
            )  # n, c
            tpv_wz = (
                cudagrid.grid_sample_2d(tpv_wz, grid[..., [2, 1]], align_corners=True, padding_mode="border")
                .reshape(self.embed_dims, inputs.shape[0])
                .transpose(0, 1)
            )  # n, c
            tpv = tpv_hw + tpv_zh + tpv_wz
            density_color = self.density_net(tpv)
            return density_color
        elif self.config.representation == 'bev':
            bev = self.density_color
            bev = bev.unflatten(1, (self.h_size, self.w_size)).permute(0, 3, 1, 2)  # bs, c, h, w
            bev = (
                cudagrid.grid_sample_2d(bev, grid[..., [1, 0]], align_corners=True, padding_mode="border")
                .reshape(self.embed_dims, inputs.shape[0])
                .transpose(0, 1)
            )  # n, c
            density_color = self.density_net(bev).reshape(inputs.shape[0], 1, self.z_size, -1)
            density_color = density_color.permute(0, 3, 1, 2)
            sample_grid = torch.stack([grid[..., 2], torch.zeros_like(grid[..., 2])], dim=-1)  # 1, 1, n, 2
            sample_grid = sample_grid.permute(2, 0, 1, 3)  # n, 1, 1, 2
            density_color = cudagrid.grid_sample_2d(
                density_color, sample_grid, align_corners=True, padding_mode="border"
            ).reshape(
                inputs.shape[0], -1
            )  # n, c
        elif self.config.representation == 'volume':
            volume = self.density_color
            bs, volume_h, volume_w, volume_z, dim = bev.size()
            device = bev.device
            x = torch.linspace(-1, 1, self.w_size)
            y = torch.linspace(-1, 1, self.h_size)
            z = torch.linspace(-1, 1, self.z_size)
            Y, X, Z = torch.meshgrid(y, x, z)
            vertices = torch.stack([Z, X, Y], dim=-1).to(device)
            real_size = torch.FloatTensor([volume_z, volume_w, volume_h]).to(device)
            verts_norm = vertices * real_size/(real_size-1)
            volume = cudagrid.grid_sample_3d(bev.permute(0,4,1,2,3), verts_norm[None,...].repeat(bs,1,1,1,1), padding_mode='border', align_corners=True)
            
            volume_feats = cudagrid.grid_sample_3d(volume, grid[...,[2,1,0]], align_corners=True, padding_mode="border").permute(0,2,3,4,1)
            density_color = self.density_net(volume_feats).reshape(inputs.shape[0], -1)
            return density_color

    def forward_sdfnetwork_online(self, inputs):
        grid = self.mapping.meter2grid(inputs, True)
        grid = grid[None, None, ...] * 2 - 1  # 1, 1, n, 3

        if self.config.representation == 'tpv':
            tpv_hw, tpv_zh, tpv_wz = self.density_color.split(
                [self.h_size * self.w_size, self.z_size * self.h_size, self.w_size * self.z_size], dim=1
            )
            tpv_hw = tpv_hw.reshape(-1, self.h_size, self.w_size, self.embed_dims).permute(0, 3, 1, 2)
            tpv_zh = tpv_zh.reshape(-1, self.z_size, self.h_size, self.embed_dims).permute(0, 3, 1, 2)
            tpv_wz = tpv_wz.reshape(-1, self.w_size, self.z_size, self.embed_dims).permute(0, 3, 1, 2)
            tpv_hw = (
                cudagrid.grid_sample_2d(tpv_hw, grid[..., [1, 0]], align_corners=True, padding_mode="border")
                .reshape(self.embed_dims, inputs.shape[0])
                .transpose(0, 1)
            )  # n, c
            tpv_zh = (
                cudagrid.grid_sample_2d(tpv_zh, grid[..., [0, 2]], align_corners=True, padding_mode="border")
                .reshape(self.embed_dims, inputs.shape[0])
                .transpose(0, 1)
            )  # n, c
            tpv_wz = (
                cudagrid.grid_sample_2d(tpv_wz, grid[..., [2, 1]], align_corners=True, padding_mode="border")
                .reshape(self.embed_dims, inputs.shape[0])
                .transpose(0, 1)
            )  # n, c
            tpv = tpv_hw + tpv_zh + tpv_wz
            density_color = self.density_net(tpv)
            return density_color[:, :1]
        elif self.config.representation == 'bev':
            bev = self.density_color
            bev = bev.unflatten(1, (self.h_size, self.w_size)).permute(0, 3, 1, 2)  # bs, c, h, w
            bev = (
                cudagrid.grid_sample_2d(bev, grid[..., [1, 0]], align_corners=True, padding_mode="border")
                .reshape(self.embed_dims, inputs.shape[0])
                .transpose(0, 1)
            )  # n, c
            density_color = self.density_net(bev).reshape(inputs.shape[0], 1, self.z_size, -1)[..., :1]
            density_color = density_color.permute(0, 3, 1, 2)
            sample_grid = torch.stack([grid[..., 2], torch.zeros_like(grid[..., 2])], dim=-1)  # 1, 1, n, 2
            sample_grid = sample_grid.permute(2, 0, 1, 3)  # n, 1, 1, 2
            density_color = cudagrid.grid_sample_2d(
                density_color, sample_grid, align_corners=True, padding_mode="border"
            ).reshape(
                inputs.shape[0], -1
            )  # n, c
            return density_color
    
    def forward_geonetwork(self, inputs):
        """forward the geonetwork"""
        if self.config.calculate_online:
            return self.forward_geonetwork_online(inputs)
        if self.density_color_bg is not None:
            foreground_mask = (inputs[...,0]>self.aabb[0,0]) & (inputs[...,0]<self.aabb[1,0]) & (inputs[...,1]>self.aabb[0,1]) & (inputs[...,1]<self.aabb[1,1]) \
                                & (inputs[...,2]>self.aabb[0,2]) & (inputs[...,2]<self.aabb[1,2])
        else:
            foreground_mask = torch.ones_like(inputs[...,0], dtype=bool)
        foreground_output = self.sample_something(inputs[foreground_mask], self.volume_pad(self.density_color))
        background_output = self.sample_something(inputs[~foreground_mask], self.density_color_bg) if (~foreground_mask).sum()>0 else None
        return foreground_output, background_output, foreground_mask

    def forward_sdfnetwork(self, inputs):
        """forward the geonetwork"""
        if self.config.calculate_online:
            return self.forward_sdfnetwork_online(inputs)
        return self.sample_something(inputs, self.density_color[:, :1, ...])

    def sample_something(self, inputs, tensor):
        # tensor: bs, c, h, w, d
        bs = tensor.size(0)
        grid = self.mapping.meter2grid(inputs, True)

        grid = 2 * grid - 1
        grid = grid.reshape(bs, -1, 1, 1, 3).to(tensor.dtype)

        sampled = cudagrid.grid_sample_3d(
            tensor, grid[..., [2, 1, 0]], align_corners=True, padding_mode="border"
        )  # bs, c, n, 1, 1

        sampled = sampled.permute(0, 2, 3, 4, 1).flatten(0, 3)  # bs*n, c
        return sampled

    def get_sdf(self, ray_samples: RaySamples):
        """predict the sdf value for ray samples"""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        sdf = self.forward_sdfnetwork(positions_flat).view(*ray_samples.frustums.shape, 1)
        return sdf

    def set_numerical_gradients_delta(self, delta: float) -> None:
        """Set the delta value for numerical gradient."""
        self.numerical_gradients_delta = delta

    def gradient(self, x, skip_spatial_distortion=False, return_sdf=False):
        """compute the gradient of the ray"""
        if self.spatial_distortion is not None and not skip_spatial_distortion:
            x = self.spatial_distortion(x)

        # compute gradient in contracted space
        if self.config.use_numerical_gradients:
            # https://github.com/bennyguo/instant-nsr-pl/blob/main/models/geometry.py#L173
            delta = self.numerical_gradients_delta
            points = torch.stack(
                [
                    x + torch.as_tensor([delta, 0.0, 0.0]).to(x),
                    x + torch.as_tensor([-delta, 0.0, 0.0]).to(x),
                    x + torch.as_tensor([0.0, delta, 0.0]).to(x),
                    x + torch.as_tensor([0.0, -delta, 0.0]).to(x),
                    x + torch.as_tensor([0.0, 0.0, delta]).to(x),
                    x + torch.as_tensor([0.0, 0.0, -delta]).to(x),
                ],
                dim=0,
            )

            points_sdf = self.forward_sdfnetwork(points.view(-1, 3)).view(6, *x.shape[:-1])
            gradients = torch.stack(
                [
                    0.5 * (points_sdf[0] - points_sdf[1]) / delta,
                    0.5 * (points_sdf[2] - points_sdf[3]) / delta,
                    0.5 * (points_sdf[4] - points_sdf[5]) / delta,
                ],
                dim=-1,
            )
        else:
            x.requires_grad_(True)

            y = self.forward_sdfnetwork(x)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
        if not return_sdf:
            return gradients
        else:
            return gradients, points_sdf

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        h = self.forward_geonetwork(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, geo_feature = torch.split(h, [1, self.color_dims], dim=-1)
        density = self.laplace_density(sdf)
        return density, geo_feature

    def get_alpha(self, ray_samples: RaySamples, sdf=None, gradients=None):
        """compute alpha from sdf as in NeuS"""
        if sdf is None or gradients is None:
            inputs = ray_samples.frustums.get_start_positions()
            inputs.requires_grad_(True)
            with torch.enable_grad():
                sdf = self.forward_sdfnetwork(inputs)
                # sdf, _ = torch.split(h, [1, self.color_dims], dim=-1)
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

        inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions[...,0:sdf.shape[-2],:] * gradients).sum(-1, keepdim=True)

        # anneal as NeuS
        cos_anneal_ratio = self._cos_anneal_ratio

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas[...,0:sdf.shape[-2],:] * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas[...,0:sdf.shape[-2],:] * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        # HF-NeuS
        # # sigma
        # cdf = torch.sigmoid(sdf * inv_s)
        # e = inv_s * (1 - cdf) * (-iter_cos) * ray_samples.deltas
        # alpha = (1 - torch.exp(-e)).clip(0.0, 1.0)

        return alpha

    def get_occupancy(self, sdf):
        """compute occupancy as in UniSurf"""
        occupancy = self.sigmoid(-10.0 * sdf)
        return occupancy

    def get_colors(self, points, directions, gradients, geo_features):
        """compute colors"""
        condition = directions
        sample_colors = geo_features
        if self.color_dims > 0:
            sample_colors = self.color_converter(None, condition, sample_colors, self.sh_deg, self.sh_act)
            rgb = sample_colors.reshape(-1, 3)
        else:
            rgb = torch.empty((geo_features.shape[0], 0), device=geo_features.device, dtype=geo_features.dtype)
        return rgb

    def get_uniform_gradient(self):
        inputs = torch.rand(self.config.nbr_gradient_points, 3, device=self.aabb.device)
        inputs[:, 0] = inputs[:, 0] * (self.aabb[1, 0] - self.aabb[0, 0]) + self.aabb[0, 0]
        inputs[:, 1] = inputs[:, 1] * (self.aabb[1, 1] - self.aabb[0, 1]) + self.aabb[0, 1]
        inputs[:, 2] = inputs[:, 2] * (self.aabb[1, 2] - self.aabb[0, 2]) + self.aabb[0, 2]

        inputs.requires_grad_(True)
        self.density_color.requires_grad_(True)
        with torch.enable_grad():
            sdf = self.forward_sdfnetwork(inputs)

        if self.config.use_numerical_gradients:
            with torch.enable_grad():
                gradients = self.gradient(inputs, skip_spatial_distortion=True, return_sdf=False)
        else:
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

        second_grad = None
        if self.config.second_derivative:
            second_grads = []
            for idx in range(3):
                with torch.enable_grad():
                    g_slice = gradients[..., idx]
                d_output = torch.ones_like(gradients[..., idx], requires_grad=False, device=sdf.device)
                second_grad = torch.autograd.grad(
                    outputs=g_slice,
                    inputs=inputs,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                second_grads.append(second_grad)
            second_grad = torch.stack(second_grads, dim=-1)
        return gradients, second_grad

    def get_outputs(self, ray_samples: RaySamples, return_alphas=False, return_occupancy=False):
        """compute output of ray samples"""
        # if ray_samples.camera_indices is None:
        #     raise AttributeError("Camera indices are not provided.")

        outputs = {}

        # camera_indices = ray_samples.camera_indices.squeeze()

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        # assert self.spatial_distortion is None
        if self.spatial_distortion is not None:
            inputs = self.spatial_distortion(inputs)
        points_norm = inputs.norm(dim=-1)
        # compute gradient in constracted space
        inputs.requires_grad_(True)
        self.density_color.requires_grad_(True)
        with torch.enable_grad():
            h = self.forward_geonetwork(inputs)
            sdf, geo_feature = torch.split(h, [1, self.color_dims], dim=-1)

        if self.config.use_numerical_gradients:
            with torch.enable_grad():
                gradients = self.gradient(
                    inputs,
                    skip_spatial_distortion=True,
                    # return_sdf=True,
                    return_sdf=False,
                )
            # sampled_sdf = (
            #     sampled_sdf.view(-1, *ray_samples.frustums.directions.shape[:-1]).permute(1, 2, 0).contiguous()
            # )
        else:
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
        if self.config.second_derivative:
            second_grads = []
            for idx in range(3):
                with torch.enable_grad():
                    g_slice = gradients[..., idx]
                d_output = torch.ones_like(gradients[..., idx], requires_grad=False, device=sdf.device)
                second_grad = torch.autograd.grad(
                    outputs=g_slice,
                    inputs=inputs,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                second_grads.append(second_grad)
            second_grad = torch.stack(second_grads, dim=-1)
            # sampled_sdf = None

        rgb = self.get_colors(inputs, directions_flat, gradients, geo_feature)

        # density = self.laplace_density(sdf)

        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        # density = density.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        normals = F.normalize(gradients, p=2, dim=-1)
        points_norm = points_norm.view(*ray_samples.frustums.directions.shape[:-1], -1)

        outputs.update(
            {
                FieldHeadNames.RGB: rgb,
                # FieldHeadNames.DENSITY: density,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMAL: normals,
                FieldHeadNames.GRADIENT: gradients,
                "points_norm": points_norm,
                # "sampled_sdf": sampled_sdf,
            }
        )
        if self.config.second_derivative:
            outputs.update({"second_grad": second_grad})

        if return_alphas:
            # TODO use mid point sdf for NeuS
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        if return_occupancy:
            occupancy = self.get_occupancy(sdf)
            outputs.update({FieldHeadNames.OCCUPANCY: occupancy})

        return outputs

    def get_outputs_recompute_gradient(self, ray_samples: RaySamples, return_alphas=False, return_occupancy=False):
        """compute output of ray samples"""
        outputs = {}

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        # assert self.spatial_distortion is None
        if self.spatial_distortion is not None:
            inputs = self.spatial_distortion(inputs)
        points_norm = inputs.norm(dim=-1)
        # compute gradient in constracted space
        if self.config.use_numerical_gradients:
            h = self.forward_geonetwork(inputs)
            sdf, geo_feature = torch.split(h, [1, self.color_dims], dim=-1)
            gradients = self.gradient(inputs, skip_spatial_distortion=True, return_sdf=False)
            gradients = gradients.detach()
        else:
            inputs.requires_grad_(True)
            with torch.enable_grad():
                h = self.forward_geonetwork(inputs)
                sdf, geo_feature = torch.split(h, [1, self.color_dims], dim=-1)

            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(outputs=sdf, inputs=inputs, grad_outputs=d_output, retain_graph=True)[0]

        sampled_grad, second_grad = self.get_uniform_gradient()

        rgb = self.get_colors(inputs, directions_flat, gradients, geo_feature)
        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        normals = F.normalize(gradients, p=2, dim=-1)
        points_norm = points_norm.view(*ray_samples.frustums.directions.shape[:-1], -1)

        outputs.update(
            {
                FieldHeadNames.RGB: rgb,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMAL: normals,
                FieldHeadNames.GRADIENT: sampled_grad,
                "points_norm": points_norm,
            }
        )
        if self.config.second_derivative:
            outputs.update({"second_grad": second_grad})

        if return_alphas:
            # TODO use mid point sdf for NeuS
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        if return_occupancy:
            occupancy = self.get_occupancy(sdf)
            outputs.update({FieldHeadNames.OCCUPANCY: occupancy})

        return outputs

    def get_outputs_sample_gradient(self, ray_samples: RaySamples, return_alphas=False, return_occupancy=False):
        """compute output of ray samples"""
        outputs = {}

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)
        frames_id = ray_samples.times.reshape(-1)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        if self.spatial_distortion is not None:
            inputs = self.spatial_distortion(inputs)
        points_norm = inputs.norm(dim=-1)
        # compute gradient in constracted space
        next_mask = frames_id==1
        # import time
        if self.config.estimate_flow and next_mask.sum()>0:
            alpha = self.deviation_network.get_variance().detach()
            # next_volume = self.density_color.clone()
            next_density = alpha * (0.5 + 0.5 * self.density_color[:,0:1,...].sign() * torch.expm1(-self.density_color[:,0:1,...].abs() * alpha))
            next_volume = torch.cat([next_density, self.density_color[:,1:,...]], dim=1).permute(0,4,1,2,3).flatten(0,1)
            voxel_size = torch.FloatTensor([(self.aabb[1,0]-self.aabb[0,0])/(self.w_sdf-1), (self.aabb[1,1]-self.aabb[0,1])/(self.h_sdf-1)]).to(alpha.device)
            scene_flow = self.fwd_flow.permute(0,4,1,2,3).flatten(0,1) / voxel_size[None,:,None,None]
            # t1 = time.time()
            next_volume = softsplat.softsplat(tenIn=next_volume, tenFlow=scene_flow, tenMetric=None, strMode='sum').reshape(-1, self.z_sdf, 1+self.color_dims, self.h_sdf, self.w_sdf).permute(0,2,3,4,1)
            # t2 = time.time()
            h = torch.zeros([inputs.shape[0], self.color_dims+1], dtype=alpha.dtype, device=alpha.device)
            h[next_mask] = self.sample_something(inputs[next_mask], self.volume_pad(next_volume))
            h[~next_mask] = self.sample_something(inputs[~next_mask], self.volume_pad(self.density_color))
        else:
            h = self.sample_something(inputs, self.volume_pad(self.density_color))
        # h, h_bkgd, foreground_mask = self.forward_geonetwork(inputs)
        sdf, geo_feature = torch.split(h, [1, self.color_dims], dim=-1)
        # if h_bkgd != None:
        #     density, geo_feature_bkgd = torch.split(h_bkgd, [1, self.color_dims], dim=-1)
        #     density = F.softplus(density)
        #     geo_feature = torch.cat((geo_feature, geo_feature_bkgd), dim=0)
        # gradients = self.sample_something(inputs[foreground_mask], self.volume_pad(self.gradients))  # n, 3
        # t3 = time.time()
        gradients = self.sample_something(inputs, self.volume_pad(self.gradients))
        # t4 = time.time()
        # print(t2-t1, t4-t3)

        rgb = self.get_colors(inputs, directions_flat, gradients, geo_feature[..., :3])
        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        if self.config.return_sem:
            sem = geo_feature[..., 3:]
            sem = sem.softmax(dim=-1)
            sem = sem.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-2], -1, 1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-2], -1, 3)
        normals = F.normalize(gradients, p=2, dim=-1)
        points_norm = points_norm.view(*ray_samples.frustums.directions.shape[:-1], -1)

        outputs.update(
            {
                FieldHeadNames.RGB: rgb,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMAL: normals,
                FieldHeadNames.GRADIENT: torch.cat(
                    [gradients.reshape(-1, 3), self.gradients.permute(0,2,3,4,1).reshape(-1,3)], dim=0
                ),
                "points_norm": points_norm,
            }
        )
        if self.config.second_derivative:
            outputs.update({"second_grad": self.gradients2.flatten(1, 2).flatten(2).transpose(-1, -2)})

        if return_alphas:
            # TODO use mid point sdf for NeuS
            next_ray_mask = ray_samples.times[:, 0, 0] == 1
            alphas = torch.zeros_like(sdf, dtype=ray_samples.deltas.dtype)
            alphas[~next_ray_mask] = self.get_alpha(ray_samples[~next_ray_mask], sdf[~next_ray_mask], gradients[~next_ray_mask])
            if next_ray_mask.sum()>0:
                alphas[next_ray_mask] = 1 - torch.exp(- ray_samples.deltas[next_ray_mask]* sdf[next_ray_mask])
            # if h_bkgd != None:
            #     alphas_bkgd = 1 - torch.exp(- ray_samples.deltas[...,sdf.shape[-2]:,:]* density.reshape(*ray_samples.frustums.directions.shape[:-2],-1,1))
            #     alphas =  torch.cat((alphas, alphas_bkgd), dim=-2)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        if return_occupancy:
            occupancy = self.get_occupancy(sdf)
            outputs.update({FieldHeadNames.OCCUPANCY: occupancy})

        if self.config.return_sem:
            outputs.update({"sem": sem})

        return outputs

    def forward(self, ray_samples: RaySamples, return_alphas=False, return_occupancy=False):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if self.config.use_uniform_gradient:
            field_outputs = self.get_outputs_recompute_gradient(
                ray_samples, return_alphas=return_alphas, return_occupancy=return_occupancy
            )
        elif self.config.sample_gradient:
            field_outputs = self.get_outputs_sample_gradient(
                ray_samples, return_alphas=return_alphas, return_occupancy=return_occupancy
            )
        else:
            field_outputs = self.get_outputs(
                ray_samples, return_alphas=return_alphas, return_occupancy=return_occupancy
            )
        return field_outputs
