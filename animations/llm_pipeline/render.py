#!/usr/bin/env python3
"""Render LLM pipeline spotlight frames for AI Theory.

Pipeline: Input -> Tokenize -> Embed -> Attention+KV -> MoE -> LM head -> Serve | Train
Example sentence: The cat sat on the
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FRAMES = ROOT / "frames"
W, H = 960, 540
FPS = 10

BG = (18, 22, 28)
PANEL = (28, 34, 44)
CHIP = (40, 48, 62)
CHIP_ACTIVE = (46, 125, 140)
CHIP_DONE = (55, 70, 88)
TEXT = (230, 235, 240)
MUTED = (140, 155, 170)
ACCENT = (90, 200, 180)
WARN = (220, 170, 90)
CARD_BG = (24, 30, 40)
BORDER = (60, 72, 90)
GROUP = (70, 90, 110)

STAGES = [
    "Input",
    "Tokenize",
    "Embed",
    "Attn+KV",
    "MoE",
    "LM head",
    "Serve",
    "Train",
]

# Visual category brackets over chips: (label, start_index, end_index inclusive)
CATEGORIES = [
    ("Tokenize", 0, 2),
    ("Transformer", 3, 4),
    ("Predict", 5, 7),
]

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered(
    draw: ImageDraw.ImageDraw,
    y: float,
    text: str,
    fill,
    fnt: ImageFont.ImageFont,
    x0: float = 0,
    x1: float = W,
) -> None:
    tw, _ = text_size(draw, text, fnt)
    draw.text((x0 + (x1 - x0 - tw) / 2, y), text, fill=fill, font=fnt)


@dataclass
class Beat:
    stage_index: int
    hold_s: float
    title: str
    lines: list[str]
    caption: str
    note: str = ""


BEATS: list[Beat] = [
    Beat(
        0,
        8.0,
        "0. Raw input",
        ['"The cat sat on the"', "", "A short prompt enters the model."],
        "Tokenize path: raw text before IDs and vectors",
        "Start of the pipeline.",
    ),
    Beat(
        1,
        8.0,
        "1. Tokenization",
        [
            "The | cat | sat | on | the",
            "",
            "IDs (illustrative):",
            "[15496, 3797, 3290, 319, 262]",
        ],
        "Tokenize: split text into tokens and IDs",
        "Text becomes token IDs the model knows.",
    ),
    Beat(
        2,
        8.0,
        "2. Embedding",
        [
            "Each ID -> vector (hidden state)",
            "",
            "15496 -> [ 0.12, -0.41, 0.08, ... ]",
            " 3797 -> [ 0.05,  0.22, -0.17, ... ]",
            "   ...",
        ],
        "Tokenize: IDs become vectors for the stack",
        "Numbers, not words, flow through the net.",
    ),
    Beat(
        3,
        8.0,
        "3a. Attention + KV cache",
        [
            "Last position looks left across the sequence.",
            "",
            "KV cache stores Keys/Values so we",
            "do not recompute the whole prompt",
            "on every new token.",
        ],
        "Inside transformer layer (stack repeats x N layers)",
        "Longer context -> more KV memory (often VRAM).",
    ),
    Beat(
        4,
        8.0,
        "3b. MoE (inside a transformer layer)",
        [
            "Router picks top experts for this token:",
            "  -> expert 3, expert 7  (sparse compute)",
            "",
            "Full expert pool still exists in RAM",
            "even while only a few run.",
        ],
        "Inside transformer layer: feed-forward is MoE",
        "Sparse compute; full expert pool still stored.",
    ),
    Beat(
        5,
        8.0,
        "4. LM head -> next-token probs",
        [
            "Logits -> softmax over vocabulary",
            "",
            "  mat   0.31",
            "  floor 0.18",
            "  rug   0.11",
            "  ...",
        ],
        "Predict: turn last hidden state into vocab probs",
        "Distribution over what comes next.",
    ),
    Beat(
        6,
        8.0,
        "Branch A - Inference (serve)",
        [
            "Pick token: mat",
            "Detokenize / append",
            "",
            '"The cat sat on the mat"',
            "",
            "Loop: feed back into the stack",
            "until stop. No loss step.",
        ],
        "Predict / serve: choose a token and continue",
        "What FreeToken / llama.cpp / chat do.",
    ),
    Beat(
        7,
        8.0,
        "Branch B - Training",
        [
            "Predicted: distribution over vocab",
            "Actual next token: mat",
            "",
            "Compare -> loss / error",
            "Backprop -> update weights",
        ],
        "Predict / train: compare to true next token",
        "Train only. Serve path skips this.",
    ),
]


def draw_rounded_rect(draw: ImageDraw.ImageDraw, xy, fill, outline=None, radius=12, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def chip_geometry(margin: int, gap: int) -> tuple[int, list[tuple[float, float]]]:
    n = len(STAGES)
    total_gap = gap * (n - 1)
    chip_w = (W - 2 * margin - total_gap) // n
    xs: list[tuple[float, float]] = []
    x = float(margin)
    for _ in range(n):
        xs.append((x, x + chip_w))
        x += chip_w + gap
    return chip_w, xs


def render_frame(beat: Beat) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    title_f = font(22, bold=True)
    body_f = font(16)
    small_f = font(13)
    chip_f = font(11, bold=True)
    group_f = font(12, bold=True)
    note_f = font(14)
    caption_f = font(14, bold=True)

    # Centered headers
    draw_centered(draw, 14, "AI Theory - LLM pipeline", TEXT, title_f)
    draw_centered(
        draw,
        42,
        "Example follows one prompt through tokenize -> transformer -> predict",
        MUTED,
        small_f,
    )

    margin = 28
    gap = 8
    chip_h = 34
    chip_y = 88
    chip_w, chip_xs = chip_geometry(margin, gap)

    # Category brackets above chips
    bracket_y = 68
    for label, i0, i1 in CATEGORIES:
        x0 = chip_xs[i0][0]
        x1 = chip_xs[i1][1]
        # bracket line
        mid_y = bracket_y + 6
        draw.line((x0 + 4, mid_y, x1 - 4, mid_y), fill=GROUP, width=2)
        draw.line((x0 + 4, mid_y, x0 + 4, mid_y + 6), fill=GROUP, width=2)
        draw.line((x1 - 4, mid_y, x1 - 4, mid_y + 6), fill=GROUP, width=2)
        # label centered in group
        active_group = i0 <= beat.stage_index <= i1
        draw_centered(
            draw,
            bracket_y - 12,
            label,
            ACCENT if active_group else MUTED,
            group_f,
            x0,
            x1,
        )

    # Stage chips
    for i, label in enumerate(STAGES):
        x0, x1 = chip_xs[i]
        if i == beat.stage_index:
            fill, outline = CHIP_ACTIVE, ACCENT
        elif i < beat.stage_index:
            fill, outline = CHIP_DONE, BORDER
        else:
            fill, outline = CHIP, BORDER
        draw_rounded_rect(draw, (x0, chip_y, x1, chip_y + chip_h), fill=fill, outline=outline, radius=8, width=2)
        tw, th = text_size(draw, label, chip_f)
        draw.text(
            (x0 + (x1 - x0 - tw) / 2, chip_y + (chip_h - th) / 2 - 1),
            label,
            fill=TEXT,
            font=chip_f,
        )
        if i < len(STAGES) - 1:
            ax0 = x1 + 1
            ax1 = chip_xs[i + 1][0] - 1
            mid = chip_y + chip_h // 2
            draw.line((ax0, mid, ax1, mid), fill=BORDER, width=2)

    # Centered per-step caption (same place as former transformer-only line)
    caption_y = 130
    draw_centered(draw, caption_y, beat.caption, ACCENT, caption_f)

    # Data card with centered body
    card_top = 158
    card_bottom = H - 70
    card = (margin, card_top, W - margin, card_bottom)
    draw_rounded_rect(draw, card, fill=CARD_BG, outline=BORDER, radius=14, width=2)

    # Centered title + lines as a block in the upper/mid card
    content_lines = [beat.title, ""] + beat.lines
    line_h = 26
    block_h = line_h * len(content_lines)
    # leave room for note strip
    note_reserved = 48 if beat.note else 16
    avail_top = card_top + 20
    avail_bottom = card_bottom - note_reserved - 8
    start_y = avail_top + max(0, (avail_bottom - avail_top - block_h) / 2)

    y = start_y
    for i, line in enumerate(content_lines):
        if i == 0:
            draw_centered(draw, y, line, ACCENT, title_f, margin, W - margin)
        elif line:
            draw_centered(draw, y, line, TEXT, body_f, margin, W - margin)
        y += line_h

    if beat.note:
        note_box = (margin + 20, card_bottom - 44, W - margin - 20, card_bottom - 10)
        draw_rounded_rect(draw, note_box, fill=PANEL, outline=BORDER, radius=8, width=1)
        tw, th = text_size(draw, beat.note, note_f)
        nx0, ny0, nx1, ny1 = note_box
        draw.text(
            (nx0 + (nx1 - nx0 - tw) / 2, ny0 + (ny1 - ny0 - th) / 2 - 1),
            beat.note,
            fill=WARN,
            font=note_f,
        )

    step = beat.stage_index + 1
    footer = f"Step {step}/{len(STAGES)}  ·  toy numbers for teaching, not a real model run"
    draw_centered(draw, H - 42, footer, MUTED, small_f)

    bar_x0, bar_x1 = margin, W - margin
    bar_y = H - 22
    draw.rectangle((bar_x0, bar_y, bar_x1, bar_y + 6), fill=CHIP)
    frac = step / len(STAGES)
    draw.rectangle(
        (bar_x0, bar_y, bar_x0 + int((bar_x1 - bar_x0) * frac), bar_y + 6),
        fill=CHIP_ACTIVE,
    )

    return img


def main() -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    for old in FRAMES.glob("frame_*.png"):
        old.unlink()

    idx = 0
    for beat in BEATS:
        frame = render_frame(beat)
        n_hold = max(1, int(round(beat.hold_s * FPS)))
        for _ in range(n_hold):
            path = FRAMES / f"frame_{idx:04d}.png"
            frame.save(path)
            idx += 1

    print(f"Wrote {idx} frames to {FRAMES} @ {FPS} fps (~{idx / FPS:.1f}s)")


if __name__ == "__main__":
    main()
