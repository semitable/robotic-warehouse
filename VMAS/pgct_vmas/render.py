"""Top-down RGB frames of VMAS Team Discovery for videos (evaluation side; Pillow only, no pyglet).

Targets are dots in their type colour with a faint covering-range ring; agents are circles coloured by index.
The small square next to an agent is its hidden team's target colour: evaluation-only information shown to the
human viewer, never to the agents.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZE = 480
HEADER = 36
TYPE = [(60, 170, 80), (210, 70, 60), (60, 120, 220), (220, 180, 40)]
AGENT = [(52, 101, 204), (150, 70, 180), (30, 160, 170), (220, 120, 30), (90, 90, 90), (200, 60, 140)]


def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_adapter(adapter, header: str = "", show_teams: bool = True) -> np.ndarray:
    cfg = adapter.discovery
    image = Image.new("RGB", (SIZE, SIZE + HEADER), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    small = _font(12)
    draw.text((6, 4), header, fill=(30, 30, 30), font=small)
    scale = SIZE / (2 * max(cfg.x_semidim, cfg.y_semidim) * 1.05)

    def px(p):
        return SIZE / 2 + p[0] * scale, HEADER + SIZE / 2 - p[1] * scale

    x0, y0 = px((-cfg.x_semidim, cfg.y_semidim))
    x1, y1 = px((cfg.x_semidim, -cfg.y_semidim))
    draw.rectangle([x0, y0, x1, y1], fill=(244, 242, 236), outline=(120, 120, 120), width=2)
    targets, types = adapter.target_state()
    for p, t in zip(targets, types):
        cx, cy = px(p)
        ring, dot = cfg.covering_range * scale, .05 * scale
        color = TYPE[int(t) % len(TYPE)]
        draw.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline=tuple(int(.5 * c + 127) for c in color))
        draw.ellipse([cx - dot, cy - dot, cx + dot, cy + dot], fill=color, outline=(30, 30, 30))
    teams = adapter.oracle_team_ids() if show_teams else None
    for i, p in enumerate(adapter.positions()):
        cx, cy = px(p)
        rad = .05 * scale
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=AGENT[i % len(AGENT)], outline=(20, 20, 20), width=2)
        draw.text((cx - 4, cy - 7), str(i), fill=(255, 255, 255), font=small)
        if teams is not None:
            draw.rectangle([cx + rad, cy - rad - 6, cx + rad + 8, cy - rad + 2], fill=TYPE[int(teams[i]) % len(TYPE)])
    return np.asarray(image)
