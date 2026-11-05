"""RGB frames of the multi-team LBF field for videos (evaluation side; never used by training).

Food is a square in its type colour with its level; agents are circles coloured by index with "index:level".
The small square in an agent's cell corner is its hidden team's food colour: evaluation-only information
shown to the human viewer, never to the agents.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CELL = 40
HEADER = 36
FOOD = [(222, 160, 40), (60, 140, 220), (200, 70, 160), (90, 170, 90)]
AGENT = [(52, 101, 204), (150, 70, 180), (30, 160, 170), (220, 120, 30), (90, 90, 90), (200, 60, 140),
         (120, 140, 40), (160, 90, 60)]


def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_rgb(env, header: str = "", show_teams: bool = True, cell: int = CELL) -> np.ndarray:
    """Frame [H, W, 3] uint8; width and height are multiples of 16 (video codecs)."""
    width, height = env.cols * cell, env.rows * cell + HEADER
    image = Image.new("RGB", (-(-width // 16) * 16, -(-height // 16) * 16), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    small, big = _font(max(10, cell // 4)), _font(max(12, cell // 2 - 2))
    draw.text((6, 4), header, fill=(30, 30, 30), font=small)
    for r in range(env.rows):
        for c in range(env.cols):
            x0, y0 = c * cell, HEADER + r * cell
            draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=(240, 236, 226), outline=(210, 204, 190))
            if env.field[r, c] > 0:
                color = FOOD[int(env.food_type[r, c]) % len(FOOD)]
                m = cell // 6
                draw.rectangle([x0 + m, y0 + m, x0 + cell - m, y0 + cell - m], fill=color, outline=(40, 40, 40))
                draw.text((x0 + cell // 2 - 5, y0 + cell // 2 - 9), str(int(env.field[r, c])), fill=(255, 255, 255),
                          font=big)
    teams = env.oracle_team_ids() if show_teams else None
    for i, (r, c) in enumerate(env.positions):
        x0, y0 = c * cell, HEADER + r * cell
        cx, cy, rad = x0 + cell // 2, y0 + cell // 2, cell // 2 - 4
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=AGENT[i % len(AGENT)], outline=(20, 20, 20), width=2)
        draw.text((cx - 9, cy - 7), f"{i}:{int(env.levels[i])}", fill=(255, 255, 255), font=small)
        if teams is not None:
            draw.rectangle([x0 + 1, y0 + 1, x0 + 10, y0 + 10], fill=FOOD[int(teams[i]) % len(FOOD)], outline=(40, 40, 40))
    return np.asarray(image)


def render_adapter(adapter, header=""):
    return render_rgb(adapter.env, header)
