"""RGB rendering of the kitchen for videos (evaluation side; never used by training).

Agents are coloured by index. The small letter under each agent is its hidden team's recipe (O = onion,
T = tomato): evaluation-only information shown to the human viewer, never to the agents.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .env.kitchen import DIRS

CELL = 48
HEADER = 40
COLORS = dict(floor=(236, 228, 210), counter=(150, 128, 104), onion_dispenser=(214, 181, 72),
              tomato_dispenser=(196, 72, 60), dish_dispenser=(206, 206, 214), pot=(70, 70, 76),
              serve=(92, 168, 96), grid=(205, 196, 176), text=(30, 30, 30))
ITEM = dict(onion=(240, 205, 70), tomato=(220, 60, 50), dish=(250, 250, 250), soup_onion=(240, 190, 40),
            soup_tomato=(210, 70, 40), soup_mixed=(130, 95, 60))
AGENT = [(52, 101, 204), (150, 70, 180), (30, 160, 170), (220, 120, 30), (90, 90, 90), (200, 60, 140)]
LETTER = dict(onion_dispenser="O", tomato_dispenser="T", dish_dispenser="D", serve="S")


def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _item(draw, cx, cy, kind, r):
    color = ITEM[kind]
    if kind == "dish":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(120, 120, 120), width=2)
    elif kind.startswith("soup"):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(250, 250, 250), outline=(120, 120, 120), width=2)
        draw.ellipse([cx - r + 4, cy - r + 4, cx + r - 4, cy + r - 4], fill=color)
    else:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(60, 60, 60))


def render_rgb(kitchen, header: str = "", recipes=None, cell: int = CELL) -> np.ndarray:
    """Frame [H, W, 3] uint8; width and height are multiples of 16 (video codecs)."""
    width, height = kitchen.width * cell, kitchen.height * cell + HEADER
    width16, height16 = -(-width // 16) * 16, -(-height // 16) * 16
    image = Image.new("RGB", (width16, height16), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    small, big = _font(max(10, cell // 4)), _font(max(12, cell // 3))
    draw.text((6, 4), header, fill=COLORS["text"], font=small)
    for r in range(kitchen.height):
        for c in range(kitchen.width):
            x0, y0 = c * cell, HEADER + r * cell
            kind = kitchen.static[r, c] or "floor"
            draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=COLORS[kind], outline=COLORS["grid"])
            if kind in LETTER:
                draw.text((x0 + 4, y0 + 2), LETTER[kind], fill=(255, 255, 255) if kind != "dish_dispenser" else (60, 60, 60),
                          font=big)
            if kind == "pot":
                pot = kitchen.pots[(r, c)]
                m = cell // 6
                draw.ellipse([x0 + m, y0 + m, x0 + cell - m, y0 + cell - m], fill=(40, 40, 44))
                for k, ingredient in enumerate(pot.ingredients):
                    _item(draw, x0 + cell // 4 + k * cell // 4, y0 + cell // 2, ingredient, cell // 9)
                if pot.cooking():
                    done = min(1.0, pot.timer / max(1, kitchen.config.cook_time))
                    color = (60, 200, 80) if done >= 1 else (240, 150, 40)
                    draw.rectangle([x0 + 4, y0 + cell - 8, x0 + 4 + int((cell - 8) * done), y0 + cell - 4], fill=color)
            if (r, c) in kitchen.counter_objects:
                _item(draw, x0 + cell // 2, y0 + cell // 2, kitchen.counter_objects[(r, c)], cell // 6)
    for i, agent in enumerate(kitchen.agents):
        r, c = agent.pos
        cx, cy = c * cell + cell // 2, HEADER + r * cell + cell // 2
        rad = cell // 3
        color = AGENT[i % len(AGENT)]
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=color, outline=(20, 20, 20), width=2)
        dr, dc = DIRS[agent.facing]
        tip = (cx + dc * (rad + 6), cy + dr * (rad + 6))
        side = (-dr, dc)
        base = (cx + dc * (rad - 2), cy + dr * (rad - 2))
        draw.polygon([tip, (base[0] + side[1] * 6, base[1] + side[0] * 6), (base[0] - side[1] * 6, base[1] - side[0] * 6)],
                     fill=(20, 20, 20))
        draw.text((cx - 4, cy - 8), str(i), fill=(255, 255, 255), font=small)
        if recipes is not None:  # hidden-team recipe badge in the cell's top-left corner (viewer only)
            x0, y0 = c * cell, HEADER + r * cell
            draw.rectangle([x0 + 2, y0 + 2, x0 + 15, y0 + 15], fill=ITEM[recipes[i]], outline=(40, 40, 40))
            draw.text((x0 + 4, y0 + 1), recipes[i][0].upper(), fill=(20, 20, 20), font=small)
        if agent.held is not None:
            _item(draw, cx + dc * rad // 2 + rad // 2, cy + dr * rad // 2 - rad // 2, agent.held, cell // 8)
    return np.asarray(image)
