"""
Patch Diffusion × JiT × MMDiT ハイブリッドモデル
- ピクセル空間 / ボトルネックパッチ埋め込み / 2D RoPE / x-prediction
- Patch Diffusionの位置情報はRoPEの位置パラメータで注入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = (int(dim * 8 / 3) + 63) // 64 * 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFFN(nn.Module):
    """Mixture of Experts SwiGLU FFN。各トークンがtop_k個のexpertのみ使う。"""
    def __init__(self, dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        hidden = (int(dim * 8 / 3) + 63) // 64 * 64
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                'w1': nn.Linear(dim, hidden, bias=False),
                'w2': nn.Linear(hidden, dim, bias=False),
                'w3': nn.Linear(dim, hidden, bias=False),
            })
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, N, D = x.shape
        x_flat = x.reshape(-1, D)

        logits = self.router(x_flat)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            idx = indices[:, k]
            w = weights[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = idx == e
                if not mask.any():
                    continue
                x_e = x_flat[mask]
                exp = self.experts[e]
                y_e = exp['w2'](F.silu(exp['w1'](x_e)) * exp['w3'](x_e))
                output[mask] += w[mask] * y_e

        return output.reshape(B, N, D)


class PatchEmbed(nn.Module):
    """
    bottleneck_dim=None: 直接射影 (patch_pixels → hidden_size)
    bottleneck_dim=int:  ボトルネック (patch_pixels → bottleneck → hidden_size)
    """
    def __init__(self, patch_size, in_channels, embed_dim, bottleneck_dim=None):
        super().__init__()
        self.patch_size = patch_size
        if bottleneck_dim is None:
            self.proj = nn.Conv2d(in_channels, embed_dim,
                                  kernel_size=patch_size, stride=patch_size)
            self.bottleneck = False
        else:
            self.proj1 = nn.Conv2d(in_channels, bottleneck_dim,
                                   kernel_size=patch_size, stride=patch_size)
            self.act = nn.GELU()
            self.proj2 = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1)
            self.bottleneck = True

    def forward(self, x):
        if self.bottleneck:
            x = self.act(self.proj1(x))
            x = self.proj2(x)
        else:
            x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim),
        )

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        emb = t.unsqueeze(-1).float() * freqs
        emb = torch.cat([emb.cos(), emb.sin()], dim=-1)
        return self.mlp(emb)


def compute_rope_2d(head_dim, pos_h, pos_w, theta=10000.0):
    """
    2D RoPEのcos/sinを計算。

    Args:
        head_dim: attention headの次元 (例: 64)
        pos_h: (B, N) 各トークンの行位置 (パッチグリッド座標)
        pos_w: (B, N) 各トークンの列位置
    Returns:
        cos, sin: (B, 1, N, head_dim)
    """
    half = head_dim // 2
    quarter = half // 2
    device = pos_h.device

    inv_freq_h = 1.0 / (theta ** (torch.arange(0, half, 2, device=device).float() / half))
    inv_freq_w = 1.0 / (theta ** (torch.arange(0, half, 2, device=device).float() / half))

    angles_h = pos_h.unsqueeze(-1).float() * inv_freq_h
    angles_w = pos_w.unsqueeze(-1).float() * inv_freq_w

    angles_h = angles_h.repeat_interleave(2, dim=-1)
    angles_w = angles_w.repeat_interleave(2, dim=-1)
    angles = torch.cat([angles_h, angles_w], dim=-1)

    return angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)


def apply_rope(x, cos, sin):
    """RoPE回転を適用。 x: (B, H, N, D)"""
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    x_rot = torch.stack([-x_pairs[..., 1], x_pairs[..., 0]], dim=-1)
    x_rot = x_rot.reshape(x.shape)
    return x * cos + x_rot * sin


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, num_experts=0, top_k=2):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        if num_experts > 0:
            self.ffn = MoEFFN(dim, num_experts=num_experts, top_k=top_k)
        else:
            self.ffn = SwiGLUFFN(dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, c, cos, sin):
        mod = self.adaLN(c).unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            mod.chunk(6, dim=-1)

        h = self.norm1(x) * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.drop(self.attn(h, cos, sin))

        h = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.drop(self.ffn(h))
        return x


class PatchDiffusionDiT(nn.Module):
    """
    Args:
        img_size:       学習画像の解像度。patch_sizeの倍数であること。
                        例: 256, 512, 1024
        patch_size:     パッチの辺ピクセル数。img_sizeの約数であること。
                        トークン数 = (img_size/patch_size)²
                        例: 16→1024tokens, 32→256tokens, 64→64tokens
        in_channels:    入力チャネル数。通常3(RGB)。
        depth:          Transformerブロック数。任意の正整数。
                        大きいほど高性能だが計算重。例: 8, 12, 16, 24
        hidden_size:    Transformerの隠れ次元。num_headsの倍数であること。
                        かつhead_dim(=hidden_size/num_heads)は偶数であること(RoPEの要件)。
                        例: 384, 512, 768, 1024, 1280
        num_heads:      Attention head数。hidden_sizeの約数であること。
                        例: 6, 8, 12, 16
        bottleneck_dim: パッチ埋め込みのボトルネック次元。
                        None: 直接射影 (patch_pixels → hidden_size、ボトルネックなし)
                        int:  ボトルネック経由 (patch_pixels → bottleneck → hidden_size)
                        例: None, 128, 256, 512
        dropout:        選択的ドロップアウト率。中間半分のブロックにのみ適用 (JiT準拠)。
                        0.0: ドロップアウトなし (JiT-B/L)
                        0.2: JiT-H (depth=32) の設定
        num_experts:    MoE expert数。0=MoEなし(通常のSwiGLU FFN)。
                        8: 8 experts (パラメータ~5倍、計算量はtop_k/num_experts)
                        例: num_experts=8, top_k=2 → FFN計算量1/4、パラメータ~5倍
        top_k:          各トークンが使うexpert数。num_experts > 0 の時のみ有効。
                        2: DiT-MoE準拠 (top-2 routing)

    制約まとめ:
        - img_size % patch_size == 0
        - hidden_size % num_heads == 0
        - (hidden_size // num_heads) % 2 == 0  (head_dimが偶数, RoPE用)
        - Patch Diffusion使用時: 各crop_sizeもpatch_sizeの倍数であること
    """
    def __init__(
        self,
        img_size=512,
        patch_size=32,
        in_channels=3,
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        dropout=0.0,
        num_experts=0,
        top_k=2,
    ):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size({img_size})はpatch_size({patch_size})の倍数である必要があります"
        assert hidden_size % num_heads == 0, \
            f"hidden_size({hidden_size})はnum_heads({num_heads})の倍数である必要があります"
        assert (hidden_size // num_heads) % 2 == 0, \
            f"head_dim({hidden_size // num_heads})は偶数である必要があります(RoPE)"

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid_size = img_size // patch_size
        self.head_dim = hidden_size // num_heads

        self.patch_embed = PatchEmbed(
            patch_size, in_channels, hidden_size, bottleneck_dim
        )
        self.t_embed = TimestepEmbedder(hidden_size)

        self.task_emb = nn.Embedding(2, hidden_size)
        self.modality_emb = nn.Embedding(2, hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads,
                     dropout=dropout if depth // 4 <= i < depth * 3 // 4 else 0.0,
                     num_experts=num_experts, top_k=top_k)
            for i in range(depth)
        ])

        self.final_norm = RMSNorm(hidden_size)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )
        self.final_proj = nn.Linear(
            hidden_size, patch_size * patch_size * in_channels
        )

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
        nn.init.zeros_(self.task_emb.weight)
        nn.init.zeros_(self.modality_emb.weight)

    def unpatchify(self, x, h, w):
        p = self.patch_size
        c = self.in_channels
        gh, gw = h // p, w // p
        x = x.reshape(x.shape[0], gh, gw, p, p, c)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], c, h, w)

    def _backbone(self, tokens, c, pos_h, pos_w):
        """compile対象のトランスフォーマー本体。"""
        cos, sin = compute_rope_2d(self.head_dim, pos_h, pos_w)
        for block in self.blocks:
            tokens = block(tokens, c, cos, sin)
        shift, scale = self.final_adaLN(c).unsqueeze(1).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + scale) + shift
        tokens = self.final_proj(tokens)
        return tokens

    def forward(self, x, t, pos_h, pos_w, cond_x=None, task_id=None):
        """
        Args:
            x: (B, 3, H, W) ノイズ付き画像/マスク [-1, 1]
            t: (B,) タイムステップ [0, 1]
            pos_h: (B, N) 各パッチの行位置 (グリッド座標)
            pos_w: (B, N) 各パッチの列位置
            cond_x: (B, 3, H, W) 条件画像 (セグメンテーション時)。Noneなら無条件生成。
            task_id: (B,) タスクID (0=顔生成, 1=セグメンテーション)。Noneなら0扱い。
        Returns:
            x_pred: (B, 3, H, W) クリーン画像/マスクの予測
        """
        B, C, H, W = x.shape

        c = self.t_embed(t)
        if task_id is not None:
            c = c + self.task_emb(task_id)

        target_tokens = self.patch_embed(x) + self.modality_emb.weight[1]

        if cond_x is not None:
            cond_tokens = self.patch_embed(cond_x) + self.modality_emb.weight[0]
            tokens = torch.cat([cond_tokens, target_tokens], dim=1)
            pos_h = torch.cat([pos_h, pos_h], dim=1)
            pos_w = torch.cat([pos_w, pos_w], dim=1)
            N_target = target_tokens.shape[1]

            out = self._backbone(tokens, c, pos_h, pos_w)
            out = out[:, -N_target:]
        else:
            out = self._backbone(target_tokens, c, pos_h, pos_w)

        return self.unpatchify(out, H, W)


def make_position_grid(grid_h, grid_w, offset_h=0, offset_w=0, device="cpu"):
    """
    パッチグリッドの位置座標を生成。

    Patch Diffusion用: offset_h/wでクロップ位置を指定。
    推論時: offset=0でフルグリッド。

    Returns:
        pos_h: (1, grid_h * grid_w)
        pos_w: (1, grid_h * grid_w)
    """
    rows = torch.arange(grid_h, device=device) + offset_h
    cols = torch.arange(grid_w, device=device) + offset_w
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
    pos_h = grid_r.reshape(1, -1)
    pos_w = grid_c.reshape(1, -1)
    return pos_h, pos_w


@torch.no_grad()
def sample(model, batch_size=8, steps=10, device="cpu", vae=None, vae_scaling_factor=None):
    """
    Heun法 (2次ODE solver) でサンプリング。
    vaeが指定されていれば潜在空間で生成後にデコードして画像を返す。
    """
    model.eval()
    img_size = model.img_size
    grid = model.grid_size
    channels = model.in_channels

    pos_h, pos_w = make_position_grid(grid, grid, device=device)
    pos_h = pos_h.expand(batch_size, -1)
    pos_w = pos_w.expand(batch_size, -1)

    dt = 1.0 / steps
    x = torch.randn(batch_size, channels, img_size, img_size, device=device)

    for i in range(steps):
        t_cur = i / steps
        t_next = (i + 1) / steps
        t_batch = torch.full((batch_size,), t_cur, device=device)

        # x_pred from model, then derive velocity
        x_pred = model(x, t_batch, pos_h, pos_w)
        v = (x_pred - x) / max(1 - t_cur, 0.05)

        # Heun step
        x_mid = x + v * dt
        if i < steps - 1:
            t_mid = torch.full((batch_size,), t_next, device=device)
            x_pred_mid = model(x_mid, t_mid, pos_h, pos_w)
            v_mid = (x_pred_mid - x_mid) / max(1 - t_next, 0.05)
            x = x + 0.5 * (v + v_mid) * dt
        else:
            x = x_mid

    if vae is not None:
        x = x / vae_scaling_factor
        x = vae.decode(x.to(vae.dtype)).sample.float()
        return x.clamp(-1, 1)
    return x.clamp(-1, 1)
