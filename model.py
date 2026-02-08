import torch
import torch.nn as nn
from timm.layers import DropPath
import torch.nn.functional as F
from timm.layers import DropPath


class Conv_Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class UpSample(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(UpSample, self).__init__()
        self.proj = nn.Sequential(
            LayerNorm(input_ch, eps=1e-6, data_format="channels_first"),
            nn.ConvTranspose2d(input_ch, output_ch, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class up_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DW_Block(nn.Module):
    def __init__(self, in_features, kernel_size, padding, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size, padding=padding, groups=in_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size, padding=padding, groups=in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MFFN(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super(MFFN, self).__init__()

        dim_reduce = dim // 2
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim_reduce, 1),
        )

        mlp_hidden_dim = int(dim_reduce * mlp_ratio)
        self.block1 = DW_Block(in_features=dim_reduce, kernel_size=1, padding=0, hidden_features=mlp_hidden_dim, drop=0.)
        self.block2 = DW_Block(in_features=dim_reduce, kernel_size=3, padding=1, hidden_features=mlp_hidden_dim, drop=0.)
        self.block3 = DW_Block(in_features=dim_reduce, kernel_size=5, padding=2, hidden_features=mlp_hidden_dim, drop=0.)

        self.gelu = nn.GELU()
        self.conv_final = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
        )

    def forward(self, x):
        x = self.conv_init(x)

        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)

        out = torch.cat([x, x1, x2, x3], dim=1)
        out = self.gelu(out)
        out = self.conv_final(out)

        return out


class SCA(nn.Module):
    def __init__(self, dim):
        super(SCA, self).__init__()
        self.dim = dim
        self.dim_sp = dim

        self.conv1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, padding=1, dilation=1, groups=self.dim_sp)
        self.conv2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, padding=2, dilation=2, groups=self.dim_sp)

        self.conv_fusion = nn.Sequential(
            nn.Conv2d(2 * self.dim, 2 * self.dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * self.dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.dim, self.dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        cd1 = self.conv1(x)
        cd2 = self.conv2(x)
        out = torch.cat([cd1, cd2], dim=1)
        out = self.conv_fusion(out)

        return out


class CFIA(nn.Module):
    def __init__(self, dim, num_heads=8, split_size=8):
        super(CFIA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = split_size
        self.win_trans = nn.MaxPool2d(kernel_size=split_size, stride=split_size)
        self.kv_h = nn.Conv2d(dim, self.dim * 2, 1)
        self.kv_l = nn.Conv2d(dim, self.dim * 2, 1)
        self.scale = (self.dim ** -0.5)

        self.q = nn.Conv2d(dim, self.dim, 1)

        self.fusion = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 1),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        low_feats = self.win_trans(x)
        high_feats = F.interpolate(low_feats, size=H, mode='nearest')
        high_feats = high_feats - x

        h_group, w_group = H // self.win_size, W // self.win_size
        total_groups = h_group * w_group

        h_kv = self.kv_h(high_feats).permute(0, 2, 3, 1).contiguous(). \
            reshape(B, h_group, self.win_size, w_group, self.win_size, 2 * self.dim). \
            transpose(2, 3).reshape(B, total_groups, -1, 2, self.num_heads,
                                    self.dim // self.num_heads).permute(3, 0, 1, 4, 2, 5).contiguous()
        h_k, h_v = h_kv[0], h_kv[1]

        x_q = self.q(x).permute(0, 2, 3, 1).contiguous(). \
            reshape(B, h_group, self.win_size, w_group, self.win_size, self.dim). \
            transpose(2, 3).reshape(B, total_groups, -1, self.num_heads,
                                    self.dim // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        x_attn = (x_q @ h_k.transpose(-2, -1)) * self.scale
        x_attn = x_attn.softmax(dim=-1)
        x_attn = (x_attn @ h_v).transpose(2, 3).reshape(B, h_group, w_group, self.win_size, self.win_size, self.dim)
        out_x = x_attn.transpose(2, 3).reshape(B, h_group * self.win_size, w_group * self.win_size, self.dim).permute(0, 3, 1, 2).contiguous()

        l_kv = self.kv_l(low_feats).permute(0, 2, 3, 1).reshape(B, -1, 2, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        l_k, l_v = l_kv[0], l_kv[1]
        x_q = x_q.transpose(1, 2).reshape(B, self.num_heads, -1, self.dim // self.num_heads)
        l_attn = (x_q @ l_k.transpose(-2, -1)) * self.scale
        l_attn = l_attn.softmax(dim=-1)
        out_l = (l_attn @ l_v).transpose(1, 2).reshape(B, H, W, self.dim).permute(0, 3, 1, 2).contiguous()

        out = out_x + out_l
        out = self.fusion(out)

        return out
    
class CFFM(nn.Module):
    def __init__(self, channels, reduction_ratio=4):

        super(CFFM, self).__init__()
        self.channels = channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(2 * channels, 2 * channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * channels // reduction_ratio, 2 * channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feat_a, feat_b):
        assert feat_a.size(1) == feat_b.size(1) == self.channels, "Channel dimensions must match!"

        batch_size = feat_a.size(0)

        feat_a_squeezed = self.gap(feat_a).view(batch_size, self.channels)
        feat_b_squeezed = self.gap(feat_b).view(batch_size, self.channels)

        squeezed_combined = torch.cat([feat_a_squeezed, feat_b_squeezed], dim=1) # [B, 2*C]

        excitation = self.fc(squeezed_combined) # [B, 2*C]
        weight_a, weight_b = torch.split(excitation, [self.channels, self.channels], dim=1)
        weight_a = weight_a.view(batch_size, self.channels, 1, 1)
        weight_b = weight_b.view(batch_size, self.channels, 1, 1)

        feat_a_weighted = feat_a * weight_a
        feat_b_weighted = feat_b * weight_b

        fused_feat = feat_a_weighted + feat_b_weighted

        return fused_feat


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class SFT(nn.Module):
    def __init__(self, embed_dim, num_heads, split_size, drop_path=0.):
        super(SFT, self).__init__()

        self.norm1 = LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")
        self.fatt = CFIA(embed_dim, num_heads, split_size)
        self.satt = SCA(embed_dim)
        self.fusion = nn.Conv2d(embed_dim, embed_dim, 1)

        self.ffn = MFFN(dim=embed_dim, mlp_ratio=4)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x_f = self.drop_path(self.fatt(x))
        x_s = self.drop_path(self.satt(x))
        x = self.fusion(x_f + x_s) + copy

        copy = x
        x = self.norm2(x)
        x = self.drop_path(self.ffn(x)) + copy

        return x

class DSFC(nn.Module):
    def __init__(self,
                 inch=3,
                 dims=[64, 128, 256, 512],
                 depths=[2, 2, 4, 2],
                 drop_path_rate=0.,
                 n_classes=1):
        super(DSFC, self).__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(inch, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        ) 
        self.layers1 = nn.Sequential(
            *[Conv_Block(dim=dims[0], drop_path=drop_path_rate) for _ in range(depths[0])]
        )
        self.former1 = nn.Sequential(
            *[SFT(embed_dim=dims[0], num_heads=8, split_size=16, drop_path=drop_path_rate)
              for _ in range(2)]
        )
        self.mixer1 = CFFM(dims[0])

        self.down2 = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
        ) 
        self.layers2 = nn.Sequential(
            *[Conv_Block(dim=dims[1], drop_path=drop_path_rate) for _ in range(depths[1])]
        )
        self.former2 = nn.Sequential(
            *[SFT(embed_dim=dims[1], num_heads=8, split_size=8, drop_path=drop_path_rate)
              for _ in range(2)]
        )
        self.mixer2 = CFFM(dims[1])

        self.down3 = nn.Sequential(
            LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)
        )
        self.layers3 = nn.Sequential(
            *[Conv_Block(dim=dims[2], drop_path=drop_path_rate) for _ in range(depths[2])]
        )
        self.former3 = nn.Sequential(
            *[SFT(embed_dim=dims[2], num_heads=8, split_size=4, drop_path=drop_path_rate)
              for _ in range(2)]
        )
        self.mixer3 = CFFM(dims[2])

        self.down4 = nn.Sequential(
            LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2)
        )
        self.layers4 = nn.Sequential(
            *[Conv_Block(dim=dims[3], drop_path=drop_path_rate) for _ in range(depths[3])]
        )
        self.former4 = nn.Sequential(
            *[SFT(embed_dim=dims[3], num_heads=8, split_size=2, drop_path=drop_path_rate)
              for _ in range(2)]
        )
        self.mixer4 = CFFM(dims[3])

        self.up1 = UpSample(dims[3], dims[2])
        self.up_conv1 = up_block(dims[3], dims[2])
        self.up2 = UpSample(dims[2], dims[1])
        self.up_conv2 = up_block(dims[2], dims[1])
        self.up3 = UpSample(dims[1], dims[0])
        self.up_conv3 = up_block(dims[1], dims[0])

        self.head = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.ConvTranspose2d(dims[0], dims[0] // 8, kernel_size=4, stride=4),
            nn.Conv2d(dims[0] // 8, n_classes, 1)
        )

    def forward(self, x):

        x = self.down1(x)
        c1 = self.layers1(x)
        f1 = self.former1(x)
        x = self.mixer1(c1, f1)
        skip1 = x

        x = self.down2(x)
        c2 = self.layers2(x)
        f2 = self.former2(x)
        x = self.mixer2(c2, f2)
        skip2 = x

        x = self.down3(x)
        c3 = self.layers3(x)
        f3 = self.former3(x)
        x = self.mixer3(c3, f3)
        skip3 = x

        x = self.down4(x)
        c4 = self.layers4(x)
        f4 = self.former4(x)
        x = self.mixer4(c4, f4)

        x = self.up1(x)
        x = self.up_conv1(torch.cat((x, skip3), dim=1))
        x = self.up2(x)
        x = self.up_conv2(torch.cat((x, skip2), dim=1))
        x = self.up3(x)
        x = self.up_conv3(torch.cat((x, skip1), dim=1))

        logits = self.head(x)

        return logits