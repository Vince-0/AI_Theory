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

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


@dataclass
class Beat:
    stage_index: int
    hold_s: float
    title: str
    lines: list[str]
    note: str = ""


BEATS: list[Beat] = [
    Beat(
        0,
        8.0,
        "0. Raw input",
        ['"The cat sat on the"', "", "A short prompt enters the model."],
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
        "Step 3 is the transformer stack (repeat x N).",
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
        "Train only. Serve path skips this.",
    ),
]


def draw_rounded_rect(draw: ImageDraw.ImageDraw, xy, fill, outline=None, radius=12, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def render_frame(beat: Beat) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    title_f = font(22, bold=True)
    body_f = font(16)
    small_f = font(13)
    chip_f = font(12, bold=True)
    note_f = font(14)

    draw.text((28, 18), "AI Theory - LLM pipeline (spotlight)", fill=TEXT, font=title_f)
    draw.text(
        (28, 48),
        "Example follows one prompt through tokenize -> transformer -> predict",
        fill=MUTED,
        font=small_f,
    )

    chip_y = 78
    chip_h = 36
    gap = 8
    margin = 28
    n = len(STAGES)
    total_gap = gap * (n - 1)
    chip_w = (W - 2 * margin - total_gap) // n
    x = margin
    for i, label in enumerate(STAGES):
        if i == beat.stage_index:
            fill = CHIP_ACTIVE
            outline = ACCENT
        elif i < beat.stage_index:
            fill = CHIP_DONE
            outline = BORDER
        else:
            fill = CHIP
            outline = BORDER
        draw_rounded_rect(
            draw,
            (x, chip_y, x + chip_w, chip_y + chip_h),
            fill=fill,
            outline=outline,
            radius=8,
            width=2,
        )
        bbox = draw.textbbox((0, 0), label, font=chip_f)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x + (chip_w - tw) / 2, chip_y + (chip_h - th) / 2 - 1),
            label,
            fill=TEXT,
            font=chip_f,
        )
        if i < n - 1:
            ax0 = x + chip_w + 1
            ax1 = x + chip_w + gap - 1
            mid = chip_y + chip_h // 2
            draw.line((ax0, mid, ax1, mid), fill=BORDER, width=2)
        x += chip_w + gap

    if beat.stage_index in (3, 4):
        draw.text(
            (margin, 122),
            "Inside transformer layer (stack repeats x N layers)",
            fill=ACCENT,
            font=small_f,
        )

    card = (margin, 148, W - margin, H - 70)
    draw_rounded_rect(draw, card, fill=CARD_BG, outline=BORDER, radius=14, width=2)
    draw.text((margin + 24, 164), beat.title, fill=ACCENT, font=title_f)

    y = 204
    for line in beat.lines:
        draw.text((margin + 24, y), line, fill=TEXT if line else MUTED, font=body_f)
        y += 26

    if beat.note:
        draw_rounded_rect(
            draw,
            (margin + 20, H - 120, W - margin - 20, H - 82),
            fill=PANEL,
            outline=BORDER,
            radius=8,
            width=1,
        )
        draw.text((margin + 32, H - 112), beat.note, fill=WARN, font=note_f)

    step = beat.stage_index + 1
    draw.text(
        (margin, H - 42),
        f"Beat {step}/{len(STAGES)}  ·  toy numbers for teaching, not a real model run",
        fill=MUTED,
        font=small_f,
    )
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
