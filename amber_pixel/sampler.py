"""The sampler AMBER v0.3 was judged with (tuned by eye on v0.3, 2026-11-14): 100 Euler steps of the rectified flow,
classifier-free guidance 5 between t = 0.2 and 0.95, and the outer 6 px started 1% of the way towards the
transparency key colour so backgrounds stay transparent. x0 is clipped to the pixel range at every step."""
import torch

KEY = (1.0, -1.0, 1.0)  # magenta #ff00ff = transparent, in the model's [-1, 1] range
DEFAULTS = {'steps': 100, 'guidance': 5.0, 'interval': (0.2, 0.95), 'key_start': 0.99, 'frame': 6}


@torch.no_grad()
def sample(model, tokens, null_tokens, initial, steps=100, guidance=5.0, interval=(0.2, 0.95), key_start=0.99, frame=6):
    """tokens/null_tokens: B,L long; initial: B,3,H,W standard normal noise. Returns B,3,H,W in [-1, 1]."""
    device = initial.device
    resolution = initial.shape[-1]
    amp = device.type == 'cuda'
    key = torch.tensor(KEY, device=device)[None, :, None, None]
    times = (key_start * torch.linspace(1, 0, steps + 1, device=device)).tolist()
    band = torch.ones(resolution, resolution, device=device)
    if frame:
        band[frame:-frame, frame:-frame] = 0
    x = (1 - times[0]) * key * band + times[0] * initial
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
        context = model.encode_text(tokens)
        null = model.encode_text(null_tokens) if guidance != 1 else None
    for t, following in zip(times[:-1], times[1:]):
        level = torch.full((len(x),), t, device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            v = model(x, level, context=context).float()
            if null is not None and interval[0] <= t <= interval[1]:
                unconditional = model(x, level, context=null).float()
                v = unconditional + guidance * (v - unconditional)
        clean = (x - t * v).clamp(-1, 1)
        noise = (x - (1 - t) * clean) / t
        x = (1 - following) * clean + following * noise
    return x.clamp(-1, 1)
