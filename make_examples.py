"""Rebuild examples/grid.png: the 8 evaluation prompts used during training, 8 seeds each (rows = prompts)."""
import json
from pathlib import Path

from PIL import Image

from amber_pixel import AmberPixel

PROMPTS = [
    'pixel art, character, knight in grey plate armour facing right, holding a long steel sword and a round shield, plumed helmet',
    'pixel art, creature, white skull with two dark round eye sockets, a small nose hole and a row of teeth, light grey shading',
    'pixel art, object, steel sword pointing diagonally up, silver blade with a pale highlight, gold crossguard and brown grip',
    'pixel art, object, wooden barrel with brown vertical planks, two dark metal hoops and a flat round top',
    'pixel art, plant, red mushroom with a round cap covered in white spots and a pale cream stem',
    'pixel art, creature, green slime monster, rounded jelly body with two dark eyes, lime highlights and a wide mouth',
    'pixel art, object, red potion bottle, round glass flask with a narrow neck, brown cork stopper and a white glass highlight',
    'pixel art, object, wooden chair with a tall slatted back, four straight legs and a brown seat, seen from the front right',
]


def main(seed=0, count=8, scale=3, backdrop=(40, 40, 46, 255)):
    amber = AmberPixel.load()
    sheet = Image.new('RGBA', (count * 64, len(PROMPTS) * 64), backdrop)
    for row, prompt in enumerate(PROMPTS):
        for column, sprite in enumerate(amber.generate(prompt, count=count, seed=seed + row)):
            sheet.alpha_composite(sprite, (column * 64, row * 64))
    out = Path('examples')
    out.mkdir(exist_ok=True)
    sheet.convert('RGB').resize((sheet.width * scale, sheet.height * scale), Image.Resampling.NEAREST).save(out / 'grid.png')
    (out / 'grid_prompts.json').write_text(json.dumps({'seeds': [seed + r for r in range(len(PROMPTS))], 'prompts': PROMPTS}, indent=1))


if __name__ == '__main__':
    main()
