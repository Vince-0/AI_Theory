#!/usr/bin/env bash
# Render LLM pipeline spotlight GIF for AI Theory.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p frames out
python3 render.py

ffmpeg -y -framerate 10 -i frames/frame_%04d.png \
  -vf "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
  out/llm_pipeline.gif

ffmpeg -y -framerate 10 -i frames/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  out/llm_pipeline.mp4

ls -lh out/llm_pipeline.gif out/llm_pipeline.mp4
echo "Done. Embed: animations/llm_pipeline/out/llm_pipeline.gif"
