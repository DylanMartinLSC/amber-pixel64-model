"""Bit-exact pixel-art post-processing: key colour to alpha, palette snap, native-grid collapse.

Every output pixel is either fully transparent or exactly one palette colour.
"""
import numpy as np
from PIL import Image

KEY = np.array([255, 0, 255], dtype=np.float32)
PALETTES = {
    # PICO-8's fixed 16-colour palette.
    'pico8': ['000000', '1d2b53', '7e2553', '008751', 'ab5236', '5f574f', 'c2c3c7', 'fff1e8',
              'ff004d', 'ffa300', 'ffec27', '00e436', '29adff', '83769c', 'ff77a8', 'ffccaa'],
    # 1-bit: black and white only.
    '1bit': ['000000', 'ffffff'],
}


def to_lab(rgb):
    """sRGB 0-255 -> CIELAB, so palette snapping matches what eyes see rather than raw RGB distance."""
    c = np.asarray(rgb, dtype=np.float32) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]], dtype=np.float32)
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def remove_key_fringe(rgb, solid, passes=2):
    """Edge pixels blended with the magenta key (strong red+blue over green) become transparent."""
    for _ in range(passes):
        padded = np.pad(solid, 1)
        edge = solid & ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        fringe = edge & (r - g > 60) & (b - g > 60) & (np.abs(r - b) < 90)
        if not fringe.any():
            break
        solid = solid & ~fringe
    return solid


def parse_palette(value):
    """Name from PALETTES, or comma-separated hex colours such as 'ff0000,00ff00'."""
    colours = PALETTES.get(value, value.split(',') if isinstance(value, str) else value)
    return np.array([[int(c.strip().lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)] for c in colours], dtype=np.float32)


def kmeans(pixels, k, iterations=20, seed=0):
    """Small deterministic k-means for an adaptive palette (k-means++ initialisation)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(pixels, axis=0)
    if len(unique) <= k:
        return unique
    centres = [unique[rng.integers(len(unique))]]
    for _ in range(1, k):
        distance = np.min([((unique - c) ** 2).sum(1) for c in centres], axis=0)
        centres.append(unique[rng.choice(len(unique), p=distance / distance.sum())])
    centres = np.array(centres)
    for _ in range(iterations):
        labels = ((pixels[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        updated = np.array([pixels[labels == i].mean(0) if (labels == i).any() else centres[i] for i in range(k)])
        if np.allclose(updated, centres):
            break
        centres = updated
    return np.round(centres)


def snap_preserving_contrast(pixels, palette, groups=8, lightness_gap=12.0, max_detour=1.4):
    """Map pixels to a fixed palette per colour group, keeping visibly different shades distinct.

    Nearest-colour snapping alone can merge e.g. a slime's body and face into one palette green.
    Groups are assigned darkest-first; a group whose lightness differs from an already-assigned group
    by more than `lightness_gap` (CIELAB L) may take another palette colour, but only one at most
    `max_detour` times as far (in CIELAB distance) as its nearest, so hues are never swapped wholesale.
    """
    centres = kmeans(pixels, min(groups, len(np.unique(pixels, axis=0))))
    labels = ((pixels[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
    centre_lab, palette_lab = to_lab(centres), to_lab(palette)
    distance = ((centre_lab[:, None] - palette_lab[None]) ** 2).sum(-1)
    chosen = {}
    for g in np.argsort(centre_lab[:, 0]):
        best = int(np.argmin(distance[g]))
        chosen[g] = best
        for option in np.argsort(distance[g]):
            if distance[g, option] > distance[g, best] * max_detour ** 2:
                break  # no acceptable alternative: let the shades merge rather than change hue
            if not any(o == option and abs(centre_lab[g, 0] - centre_lab[h, 0]) > lightness_gap for h, o in chosen.items() if h != g):
                chosen[g] = int(option)
                break
    return palette[np.array([chosen[g] for g in labels])]


def collapse(rgb, mask, factor):
    """Downsample by block majority so a 2x-upscaled sprite returns to its native grid."""
    height, width = mask.shape
    blocks_rgb = rgb[:height - height % factor, :width - width % factor].reshape(height // factor, factor, width // factor, factor, 3)
    blocks_mask = mask[:height - height % factor, :width - width % factor].reshape(height // factor, factor, width // factor, factor)
    solid = blocks_mask.mean((1, 3)) >= 0.5
    # Average only the sprite's own pixels: including key pixels tints every edge block purple.
    weight = blocks_mask[..., None].astype(np.float32)
    colour = (blocks_rgb * weight).sum((1, 3)) / np.maximum(weight.sum((1, 3)), 1)
    return colour, solid


def to_sprite(image, palette=None, colours=16, key_tolerance=90.0, factor=1):
    """CHW tensor/array in [-1, 1] -> RGBA PIL image whose pixels are transparent or exact palette colours."""
    array = np.asarray(image.detach().float().cpu() if hasattr(image, 'detach') else image, dtype=np.float32)
    rgb = ((array.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255)
    solid = remove_key_fringe(rgb, np.sqrt(((rgb - KEY) ** 2).sum(-1)) > key_tolerance)
    if factor > 1:
        rgb, solid = collapse(rgb, solid, factor)
    pixels = rgb[solid]
    out = np.zeros(rgb.shape[:2] + (4,), dtype=np.uint8)
    if len(pixels):
        if palette:
            snapped = snap_preserving_contrast(pixels, parse_palette(palette))
        else:
            centres = kmeans(pixels, colours)
            snapped = centres[((to_lab(pixels)[:, None] - to_lab(centres)[None]) ** 2).sum(-1).argmin(1)]
        out[solid, :3] = np.rint(snapped).clip(0, 255).astype(np.uint8)
        out[solid, 3] = 255
    return Image.fromarray(out, 'RGBA')


def preview(sprites, scale=8, columns=8, background=(40, 40, 46)):
    """Nearest-neighbour contact sheet of RGBA sprites over a dark backdrop."""
    width, height = sprites[0].size
    rows = (len(sprites) + columns - 1) // columns
    sheet = Image.new('RGBA', (columns * width, rows * height), background + (255,))
    for i, sprite in enumerate(sprites):
        sheet.alpha_composite(sprite, ((i % columns) * width, (i // columns) * height))
    return sheet.resize((sheet.width * scale, sheet.height * scale), Image.Resampling.NEAREST)
