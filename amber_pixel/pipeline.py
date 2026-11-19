"""Load AMBER Pixel64 weights and turn prompts into transparent 64x64 sprites."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .model import FlowTransformer
from .sampler import DEFAULTS, sample
from .sprite import to_sprite
from .tokenizer import Tokenizer

HERE = Path(__file__).resolve().parent.parent


class AmberPixel:
    def __init__(self, model, tokenizer, config, device):
        self.model, self.tokenizer, self.config, self.device = model, tokenizer, config, device

    @classmethod
    def load(cls, weights=None, config=None, tokenizer=None, device=None):
        device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        config = json.loads(Path(config or HERE / 'config.json').read_text(encoding='utf-8'))
        a = config['architecture']
        model = FlowTransformer(a['resolution'], a['patch'], a['width'], a['depth'], a['heads'], a['text'], a['augment_dim'])
        model.load_state_dict(load_file(str(weights or HERE / 'weights' / config['weights_file'])))
        model.to(device).eval()
        return cls(model, Tokenizer.load(tokenizer or HERE / 'tokenizer.json'), config, device)

    def prompt(self, text):
        """Training captions all start 'pixel art, <kind>, ...'; the prefix is added if missing."""
        return text if text.lower().startswith('pixel art') else f'pixel art, {text}'

    def generate(self, prompt, count=8, seed=0, raw=False, **settings):
        """Returns `count` RGBA PIL sprites (transparent background, adaptive palette), or the raw B,3,64,64
        tensor in [-1, 1] with raw=True. settings: steps, guidance, interval, key_start, frame,
        and for sprites palette ('pico8', '1bit' or 'hex,hex,...') / colours (adaptive palette size, default 16)."""
        palette, colours = settings.pop('palette', None), settings.pop('colours', 16)
        settings = {**DEFAULTS, **settings}
        text = self.prompt(prompt)
        tokens = torch.tensor([self.tokenizer.encode(text)] * count, device=self.device)
        null = torch.tensor([self.tokenizer.null()] * count, device=self.device)
        side = self.config['architecture']['resolution']
        generator = torch.Generator(device='cpu').manual_seed(seed)
        initial = torch.randn(count, 3, side, side, generator=generator).to(self.device)
        images = sample(self.model, tokens, null, initial, **settings)
        if raw:
            return images
        return [to_sprite(image, palette=palette, colours=colours) for image in images]
