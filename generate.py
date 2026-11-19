"""Generate pixel-art sprites with AMBER Pixel64 v0.3.

    python generate.py "pixel art, object, red potion bottle, round glass flask with a narrow neck and a cork stopper"
    python generate.py "plant, red mushroom with white spots" --count 16 --seed 7 --palette pico8

Writes transparent PNGs at 64x64 plus an enlarged preview sheet to --out. Runs on a CUDA GPU or (slowly) on CPU.
"""
import argparse
import re
from pathlib import Path

from amber_pixel import AmberPixel
from amber_pixel.sprite import preview


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('prompt')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--guidance', type=float, default=5.0)
    parser.add_argument('--palette', help="'pico8', '1bit' or comma-separated hex colours; default is an adaptive palette")
    parser.add_argument('--colours', type=int, default=16, help='adaptive palette size')
    parser.add_argument('--out', default='sprites')
    parser.add_argument('--device', help='cuda or cpu (default: cuda if available)')
    args = parser.parse_args()
    amber = AmberPixel.load(device=args.device)
    unknown = amber.tokenizer.unknown_words(amber.prompt(args.prompt))
    if unknown:
        print(f"Words the model never saw in training (ignored as <unk>): {', '.join(unknown)}")
    sprites = amber.generate(args.prompt, count=args.count, seed=args.seed, steps=args.steps, guidance=args.guidance,
                             palette=args.palette, colours=args.colours)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r'\W+', '-', amber.prompt(args.prompt).removeprefix('pixel art,').strip()).strip('-')[:48] or 'sprite'
    for i, sprite in enumerate(sprites):
        sprite.save(out / f'{stem}-{args.seed}-{i:02d}.png')
    preview(sprites, scale=4).save(out / f'{stem}-{args.seed}-preview.png')
    print(f'Saved {len(sprites)} sprites and a preview to {out}/')


if __name__ == '__main__':
    main()
