import torch
import torch.nn as nn

from timm.layers import Mlp, DropPath, trunc_normal_
import math

class RunningStats:
    def __init__(self):
        self.mean = None
        self.var = None
        self.count = 0

    def push(self, data: torch.Tensor):
        batch_size = data.size(0)
        if self.mean is None:
            self.mean = data.mean(dim=0)
            self.var = data.var(dim=0, unbiased=False)
        else:
            new_mean = data.mean(dim=0)
            new_var = data.var(dim=0, unbiased=False)
            delta = new_mean - self.mean
            self.mean += delta * batch_size / (self.count + batch_size)
            self.var = (self.var * (self.count) + new_var * batch_size +
                        delta**2 * self.count * batch_size / (self.count + batch_size)) / (self.count + batch_size)
        self.count += batch_size

    def get_mean(self):
        return self.mean

    def get_std(self):
        return torch.sqrt(self.var)


# https://github.com/facebookresearch/ToMe/blob/main/tome/merge.py
def bipartite_soft_matching(
    metric,
    r,
    class_token = True,
    distill_token = False,
):
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return unm, dst

    def split(x):
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        return src, dst
    
    def gather(dst):
        n, t1, c = dst.shape
        return dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
    return merge, split, gather


def merge_wavg(
    merge, split, gather, x, size = None
):
    merge_flag = torch.ones_like(size)
    unm_flag, dst_flag = merge(merge_flag, mode="sum")
    merge_flag = torch.cat([unm_flag, dst_flag], dim=1)
    merge_flag = (merge_flag != 1).float()

    src0, dst0 = split(x.clone())
    x_unm, x_dst = merge(x * size, mode="sum")
    size_unm, size_dst = merge(size, mode="sum")
    dst1 = gather(x_dst/size_dst)

    x = torch.cat([x_unm, x_dst], dim=1)
    size = torch.cat([size_unm, size_dst], dim=1)

    x = x / size
    return x, size, src0, dst0, dst1, merge_flag


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., layer=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.layer = layer
        self.running_stats = RunningStats()

    def forward(self, x, size=None, merge_flag=None, distortion=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T1, T2)
        if merge_flag is not None and distortion is not None:
            attn_diag = attn.diagonal(dim1=-2, dim2=-1)
            token_mask = merge_flag.squeeze(-1)
            attn_diag += token_mask[:, None, :].to(attn.dtype) * distortion.to(attn.dtype).unsqueeze(-1)
        
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, k.mean(1)

    def measure_skew(self, src0, dst0, dst1):
        B, N, C = src0.shape
        qkv_src0 = self.qkv(src0).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_src0, k_src0, v_src0 = qkv_src0[0], qkv_src0[1], qkv_src0[2]

        qkv_dst0 = self.qkv(dst0).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_dst0, k_dst0, v_dst0 = qkv_dst0[0], qkv_dst0[1], qkv_dst0[2]
        qk0_0 = torch.sum(q_src0 * k_dst0, dim=-1) * self.scale
        qk0_1 = torch.sum(q_dst0 * k_src0, dim=-1) * self.scale

        qkv_dst1 = self.qkv(dst1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_dst1, k_dst1, v_dst1 = qkv_dst1[0], qkv_dst1[1], qkv_dst1[2]
        qk1 = torch.sum(q_dst1 * k_dst1, dim=-1) * self.scale

        attn_dist0 = (qk0_0-qk1).float()
        attn_dist1 = (qk0_1-qk1).float()
        attn_dist0 = attn_dist0.permute(0, 2, 1).reshape(B * N, self.num_heads)
        attn_dist1 = attn_dist1.permute(0, 2, 1).reshape(B * N, self.num_heads)
        attn_dist = torch.cat([attn_dist0, attn_dist0], dim=0).mean(dim=0, keepdim=True)
        self.running_stats.push(attn_dist)
        return attn_dist


class CustomBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 info = None, layer = 0, skip_lam=1., depth = 0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, layer=layer)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.info = info
        self.layer = layer
        self.kept_num = self.info["merge_schedule"][self.layer]
        self.skip_lam = skip_lam
        self.depth = depth

    def forward(self, x, prefix):
        if 1<=self.layer<=(self.depth-1):
            if self.training:
                x, src0, dst0, dst1, merge_flag = x
                distortion = self.attn.measure_skew(self.norm1(src0), self.norm1(dst0), self.norm1(dst1))
            else:
                x, merge_flag = x
                distortion = self.attn.running_stats.get_mean()
                if distortion is None:
                    distortion = torch.zeros((1, self.attn.num_heads), device=x.device)
        else:
            merge_flag = None
            distortion = None
        size = self.info[prefix + "size"]

        x_attn, metric = self.attn(self.norm1(x), size, merge_flag, distortion)
        x = x + self.drop_path1(x_attn)/self.skip_lam

        r = x.shape[1] - self.kept_num
        if r > 0:
            merge, split, gather = bipartite_soft_matching(metric, r)
            x, size, src0, dst0, dst1, merge_flag = merge_wavg(merge, split, gather, x, size)
        self.info[prefix + "size"] = size

        x = x + self.drop_path2(self.mlp(self.norm2(x)))/self.skip_lam
        if self.layer<=(self.depth-2):
            if self.training:
                src0 = src0 + self.drop_path2(self.mlp(self.norm2(src0)))/self.skip_lam
                dst0 = dst0 + self.drop_path2(self.mlp(self.norm2(dst0)))/self.skip_lam
                dst1 = dst1 + self.drop_path2(self.mlp(self.norm2(dst1)))/self.skip_lam
                return x, src0, dst0, dst1, merge_flag
            else:
                return x, merge_flag
        else:
            return x