r"""
Standalone HiCFoundation encoder: a single self-contained file with no dependency on the rest of
this repo, for anyone who just wants "load HiCFoundation weights, get patch-level embeddings out."

Requirements
------------
    torch>=2.11.0
    numpy>=2.4.4

(Verified against exactly these versions; older torch/numpy will very likely also work -- nothing
here depends on a bleeding-edge API -- but hasn't been checked.)

Usage
-----
    model = HiCFoundationModel('/path/to/hicfoundation_pretrain.pth.tar')
    matrix = torch.rand(4, 256, 256)  # raw Hi-C contact counts, (B, 256, 256)
    patch_tokens = model(matrix)      # (B, num_patches, 1024) -- CLS/count token already dropped

    # barebone resolution-enhancement model: frozen HiCFoundation encoder + a small trainable
    # bridge/decoder/pixel-head on top, predicting a full (matrix_size, matrix_size) matrix
    resenh = HiCFoundationResEnhancement(
        '/path/to/hicfoundation_pretrain.pth.tar', decoder_dim=512, decoder_layers=4,
    )
    prediction = resenh(matrix)  # (4, 256, 256)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _1d_sincos(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    out = np.einsum('m,d->md', pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed_rectangle(embed_dim: int, grid_size: tuple[int, int], cls_token: bool = False) -> np.ndarray:
    # Faithful to HiCFoundation: w-axis varies first via np.meshgrid(grid_w, grid_h),
    # then reshape uses (grid_w_size, grid_h_size) which matches their published code.
    grid_h_size, grid_w_size = grid_size
    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape([2, 1, grid_w_size, grid_h_size])
    emb_h = _1d_sincos(embed_dim // 2, grid[0])
    emb_w = _1d_sincos(embed_dim // 2, grid[1])
    pos = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos = np.concatenate([np.zeros([1, embed_dim], dtype=np.float32), pos], axis=0)
    return pos


def _convert_count_to_pos_embed(count: torch.Tensor, embed_dim: int) -> torch.Tensor:
    # count: (B,) already log-formatted (i.e. log10(total_count)).
    # The `+1` shift is HiCFoundation's choice to distinguish the count token from positional embeddings.
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=count.dtype, device=count.device) / embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = torch.einsum('m,d->md', count, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1) + 1


class _PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, norm_eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HiCFoundationModel(nn.Module):
    r"""
    HiCFoundation pretrained encoder, patch-representations-only.

    Module/parameter names match the upstream `models_hicfoundation.Models_HiCFoundation` encoder
    portion exactly (`patch_embed.proj.{weight,bias}`, `blocks.{i}.{norm1,attn.qkv,attn.proj,norm2,
    mlp.fc1,mlp.fc2}.*`, `cls_token`, `pos_embed`, `norm.{weight,bias}`), so weights from
    `hicfoundation_pretrain.pth.tar` load with `strict=True` on the encoder-only key subset.

    Constructor takes the checkpoint path directly and loads weights immediately -- no separate
    `from_pretrained`/`load_state_dict` step needed.

    `forward` takes raw Hi-C contact counts `(B, 256, 256)`, handles the entire upstream input
    pipeline internally (`log10(x+1)` -> divide by per-sample max -> invert -> stack as fake-RGB
    `[ones, x, x]` -> ImageNet-normalize), and returns only the patch-level tokens -- CLS and the
    sinusoidal "count" token are computed internally (required for the model to run faithfully) but
    dropped from the return value.
    """

    def __init__(
        self,
        weights_path: str,
        img_size: tuple[int, int] = (256, 256),
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, (
            f'img_size {img_size} must be divisible by patch_size {patch_size}'
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.patch_embed = _PatchEmbed(in_chans, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([_Block(embed_dim, num_heads, mlp_ratio, norm_eps=norm_eps) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        # ImageNet stats used by the upstream pretraining pipeline.
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), persistent=False)

        nn.init.normal_(self.cls_token, std=0.02)
        self._init_pos_embed()
        self._load_weights(weights_path)

    def _init_pos_embed(self) -> None:
        # pos_embed is always re-derived from a sin-cos grid at the configured img_size, never
        # loaded from the checkpoint -- for the pretraining grid (14x14 @ patch_size=16, 224x224)
        # the regenerated values are bit-exact to the saved ones, and this lets img_size vary freely.
        pe = _get_2d_sincos_pos_embed_rectangle(self.embed_dim, self.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

    def _load_weights(self, weights_path: str) -> None:
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
        own_keys = set(self.state_dict().keys())
        # keys present in the checkpoint but not here (decoder, mask token, decoder pos embed,
        # count/pred heads) are silently skipped; pos_embed is skipped deliberately (see above)
        filtered = {k: v for k, v in state_dict.items() if k in own_keys and k != 'pos_embed'}
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        forbidden_missing = [k for k in missing if k != 'pos_embed']
        if forbidden_missing or unexpected:
            raise RuntimeError(f'unexpected state_dict mismatch: missing={forbidden_missing}, unexpected={unexpected}')

    def _hic_to_rgb(self, matrix: torch.Tensor) -> torch.Tensor:
        # (B, H, W) raw counts -> (B, 3, H, W) ImageNet-normalized RGB-fake image.
        x = torch.log10(matrix + 1)
        max_val = x.reshape(matrix.shape[0], -1).amax(dim=-1).clamp(min=1e-6).reshape(-1, 1, 1)
        x = (max_val - x) / max_val
        rgb = torch.stack([torch.ones_like(x), x, x], dim=1)
        return (rgb - self.imagenet_mean) / self.imagenet_std

    def forward(self, matrix: torch.Tensor, total_count: torch.Tensor | None = None) -> torch.Tensor:
        r"""
        Parameters
        ----------
        matrix: torch.Tensor
            `(B, 256, 256)` raw Hi-C contact counts.
        total_count: torch.Tensor | None
            `(B,)` raw total Hi-C contact count per sample. If `None`, the upstream placeholder of
            `1e9` is used for every sample.

        Returns
        -------
        torch.Tensor
            `(B, num_patches, embed_dim)` patch-level token embeddings, raster order. CLS and the
            count token are used internally but not included in the output.
        """
        B = matrix.shape[0]
        imgs = self._hic_to_rgb(matrix)
        x = self.patch_embed(imgs) + self.pos_embed[:, 1:, :]

        if total_count is None:
            total_count = torch.full((B,), 1e9, device=matrix.device, dtype=matrix.dtype)
        count_embed = _convert_count_to_pos_embed(torch.log10(total_count), self.embed_dim).unsqueeze(1)

        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(B, -1, -1)
        x = torch.cat([cls, count_embed, x], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x[:, 2:]  # drop CLS (0) and count token (1); patch tokens only


class HiCFoundationResEnhancement(nn.Module):
    r"""
    Barebone resolution-enhancement model built on top of `HiCFoundationModel`: a bridge
    (`embed_dim` -> `decoder_dim`, default 1024 -> 512), `decoder_layers` plain transformer blocks
    at `decoder_dim`, and a pixel-level head that predicts each patch's `(patch_size, patch_size)`
    block directly, assembled into the full `(matrix_size, matrix_size)` output.
    """

    def __init__(
        self,
        weights_path: str,
        decoder_dim: int = 512,
        decoder_layers: int = 0, # Set it to non-zero value when we want to try out bigger decoders
        decoder_heads: int = 8,
        mlp_ratio: float = 4.0,
        freeze_encoder: bool = True, # Do not change to false, it requires an absurd amount of VRAM
        img_size: tuple[int, int] = (256, 256),
        patch_size: int = 16,
    ):
        super().__init__()
        assert decoder_dim % decoder_heads == 0, f'decoder_dim {decoder_dim} must be divisible by decoder_heads {decoder_heads}'
        self.encoder = HiCFoundationModel(weights_path, img_size=img_size, patch_size=patch_size)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        self.matrix_size = img_size[0]
        self.patch_size = patch_size
        self.grid_size = img_size[0] // patch_size

        self.bridge = nn.Linear(self.encoder.embed_dim, decoder_dim)
        self.decoder_blocks = nn.ModuleList([_Block(decoder_dim, decoder_heads, mlp_ratio) for _ in range(decoder_layers)])
        self.decoder_norm = nn.LayerNorm(decoder_dim, eps=1e-6)
        self.pixel_head = nn.Linear(decoder_dim, patch_size * patch_size)

    def train(self, mode: bool = True) -> 'HiCFoundationResEnhancement':
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()  # keep frozen encoder's BatchNorm/Dropout (none here, but future-proof) in eval mode
        return self

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters
        ----------
        matrix: torch.Tensor
            `(B, matrix_size, matrix_size)` raw Hi-C contact counts.

        Returns
        -------
        torch.Tensor
            `(B, matrix_size, matrix_size)` predicted (symmetrized) contact matrix.
        """
        with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
            patch_tokens = self.encoder(matrix)  # (B, num_patches, embed_dim)

        x = self.bridge(patch_tokens)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        patches = self.pixel_head(x)  # (B, num_patches, patch_size * patch_size)

        B = matrix.shape[0]
        G, P = self.grid_size, self.patch_size
        patches = patches.reshape(B, G, G, P, P)  # raster order: dim1=row, dim2=col
        out = patches.permute(0, 1, 3, 2, 4).reshape(B, G * P, G * P)
        return (out + out.transpose(-1, -2)) / 2  # cheap symmetrization -- Hi-C matrices are symmetric


if __name__ == '__main__':
    import sys

    weights_path = sys.argv[1]

    encoder = HiCFoundationModel(weights_path).eval()
    with torch.no_grad():
        patch_tokens = encoder(torch.rand(2, 256, 256))
    print(f'patch_tokens shape: {tuple(patch_tokens.shape)}')  # (2, 256, 1024) at patch_size=16

    resenh = HiCFoundationResEnhancement(weights_path, decoder_dim=512, decoder_layers=4)
    matrix = torch.rand(2, 256, 256)
    prediction = resenh(matrix)
    print(f'resenh prediction shape: {tuple(prediction.shape)}')  # (2, 256, 256)
    assert torch.allclose(prediction, prediction.transpose(-1, -2), atol=1e-5), 'output should be symmetric'
    print('symmetric: OK')

    trainable = sum(p.numel() for p in resenh.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in resenh.parameters() if not p.requires_grad)
    print(f'trainable params: {trainable:,}  frozen (encoder) params: {frozen:,}')

    loss = prediction.pow(2).mean()
    loss.backward()
    assert resenh.bridge.weight.grad is not None, 'bridge should receive gradients'
    assert resenh.encoder.blocks[0].attn.qkv.weight.grad is None, 'encoder should stay frozen'
    print('gradient flow: OK (decoder trains, frozen encoder does not)')
