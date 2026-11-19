"""AMBER dit-flow: a joint-attention (MMDiT-style) diffusion transformer on 4x4 pixel patches with a word-level text
encoder trained jointly from scratch. Rectified flow: x_t = (1 - t) x0 + t noise, the model predicts v = noise - x0.

Standard published architectures, written for AMBER; no imported model parameters."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def zero(module):
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.SiLU(), nn.Linear(width * 4, width * 4))

    def forward(self, t):
        frequencies = torch.exp(-math.log(10000) * torch.arange(self.width // 2, device=t.device).float() / (self.width // 2))
        angles = t.float()[:, None] * frequencies[None]
        return self.mlp(torch.cat([angles.cos(), angles.sin()], dim=-1))



class TextEncoder(nn.Module):
    def __init__(self, vocabulary_size, width=256, depth=4, length=48):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, width, padding_idx=0)
        self.position = nn.Parameter(torch.randn(1, length, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        # TransformerEncoder clones layers; explicitly initialize each clone independently.
        for layer in self.encoder.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens):
        valid = tokens != 0
        if not bool(valid.any(dim=1).all()):
            raise ValueError('Each text sequence needs at least BOS and EOS tokens')
        h = self.embedding(tokens) + self.position[:, :tokens.shape[1]]
        h = self.norm(self.encoder(h, src_key_padding_mask=~valid))
        pooled = (h * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True)
        return h, valid, pooled



def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Stream(nn.Module):
    """One modality's half of a joint block."""

    def __init__(self, width, heads, condition, dropout):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, width * 3)
        # Query/key RMS norm keeps attention logits bounded under bf16 (SD3, ViT-22B).
        self.q_norm = nn.RMSNorm(width // heads, eps=1e-6)
        self.k_norm = nn.RMSNorm(width // heads, eps=1e-6)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(approximate='tanh'), nn.Dropout(dropout), nn.Linear(width * 4, width))
        self.modulation = zero(nn.Linear(condition, width * 6))

    def attention_inputs(self, x, condition):
        self.mod = self.modulation(F.silu(condition)).chunk(6, dim=-1)
        batch, length, width = x.shape
        q, k, v = self.qkv(modulate(self.norm1(x), self.mod[0], self.mod[1])).reshape(batch, length, 3, self.heads, width // self.heads).unbind(2)
        return self.q_norm(q).transpose(1, 2), self.k_norm(k).transpose(1, 2), v.transpose(1, 2)

    def finish(self, x, attended):
        x = x + self.mod[2][:, None] * self.projection(attended)
        return x + self.mod[5][:, None] * self.mlp(modulate(self.norm2(x), self.mod[3], self.mod[4]))


class JointBlock(nn.Module):
    def __init__(self, width, heads, condition, dropout=0.0):
        super().__init__()
        self.image = Stream(width, heads, condition, dropout)
        self.text = Stream(width, heads, condition, dropout)

    def forward(self, x, c, condition, mask):
        qx, kx, vx = self.image.attention_inputs(x, condition)
        qc, kc, vc = self.text.attention_inputs(c, condition)
        q, k, v = (torch.cat(pair, dim=2) for pair in ((qx, qc), (kx, kc), (vx, vc)))
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        h = h.transpose(1, 2).flatten(2)
        return self.image.finish(x, h[:, :x.shape[1]]), self.text.finish(c, h[:, x.shape[1]:])


class JointTransformer(nn.Module):
    """Shared trunk: text encoder, conditioning vector (a scalar level such as flow time or mask ratio, plus pooled
    text and augmentation labels) and joint blocks. Subclasses supply image tokens in and predictions out."""

    def __init__(self, tokens, width, depth, heads, text, augment_dim=0, dropout=0.0):
        super().__init__()
        self.text = TextEncoder(**text)
        text_width = text.get('width', 256)
        self.level = TimeEmbedding(width // 4)  # outputs width
        self.pooled = nn.Linear(text_width, width)
        self.augment = nn.Linear(augment_dim, width, bias=False) if augment_dim else None
        self.context = nn.Linear(text_width, width)
        self.position = nn.Parameter(torch.randn(1, tokens, width) * 0.02)
        self.blocks = nn.ModuleList([JointBlock(width, heads, width, dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = zero(nn.Linear(width, width * 2))

    def encode_text(self, tokens):
        return self.text(tokens)

    def trunk(self, x, level, tokens=None, context=None, augment=None):
        if context is None:
            context = self.encode_text(tokens)
        states, valid, pooled = context
        condition = self.level(level) + self.pooled(pooled)
        if augment is not None:
            condition = condition + self.augment(augment)
        x = x + self.position
        c = self.context(states)
        # Image tokens always attend; padded text tokens never do.
        mask = torch.cat([torch.ones(x.shape[:2], dtype=torch.bool, device=x.device), valid], dim=1)[:, None, None, :]
        for block in self.blocks:
            x, c = block(x, c, condition, mask)
        shift, scale = self.final_modulation(F.silu(condition)).chunk(2, dim=-1)
        return modulate(self.final_norm(x), shift, scale)


class FlowTransformer(JointTransformer):
    def __init__(self, resolution=64, patch=4, width=512, depth=8, heads=8, text=None, augment_dim=0, dropout=0.0):
        self.patch, self.side = patch, resolution // patch
        super().__init__(self.side ** 2, width, depth, heads, text, augment_dim, dropout)
        self.input = nn.Linear(3 * patch * patch, width)
        self.output = zero(nn.Linear(width, 3 * patch * patch))

    def forward(self, x, t, tokens=None, context=None, augment=None):
        batch, p, s = len(x), self.patch, self.side
        patches = x.reshape(batch, 3, s, p, s, p).permute(0, 2, 4, 1, 3, 5).reshape(batch, s * s, 3 * p * p)
        h = self.trunk(self.input(patches), t * 1000, tokens, context, augment)
        return self.output(h).reshape(batch, s, s, 3, p, p).permute(0, 3, 1, 4, 2, 5).reshape(batch, 3, s * p, s * p)
