#!/usr/bin/env python3
"""Live terminal dashboard for video-generation jobs.

Reads video_jobs/<job_id>/progress.json (written by generate_video.py) and
renders a refreshing progress bar per job -- so you can watch a render without
tailing raw logs or hammering the API. Read-only; safe to run anytime.

Usage:
    python scripts/watch_jobs.py            # watch all jobs, refresh every 2s
    python scripts/watch_jobs.py --once     # print once and exit
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

JOBS_DIR = Path(__file__).resolve().parent.parent / "video_jobs"

BAR_WIDTH = 32
STAGE_LABEL = {
    "loading": "loading model",
    "rendering": "rendering",
    "stitching": "stitching",
    "done": "done",
}


def _bar(fraction: float) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * BAR_WIDTH))
    return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + f"] {fraction * 100:5.1f}%"


def _read_progress(job_dir: Path) -> dict | None:
    pf = job_dir / "progress.json"
    if not pf.exists():
        return None
    try:
        return json.loads(pf.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _job_line(job_dir: Path) -> str:
    short = job_dir.name[:8]
    final = job_dir / "final_story.mp4"
    prog = _read_progress(job_dir)

    if prog is None:
        # No progress yet: either just started, or a pre-progress-tracking job.
        state = "done" if final.exists() else "starting / no progress file"
        return f"  {short}  {state}"

    stage = prog.get("stage", "?")
    frac = prog.get("fraction", 0.0)
    if stage == "done" or final.exists():
        return f"  {short}  {_bar(1.0)}  done -> final_story.mp4"

    detail = STAGE_LABEL.get(stage, stage)
    if stage == "rendering":
        shot = prog.get("shot_index", 0) + 1
        shot_total = prog.get("shot_total", 0)
        step = prog.get("step", 0)
        step_total = prog.get("step_total", 0)
        detail = f"rendering shot {shot}/{shot_total} (step {step}/{step_total})"

    return f"  {short}  {_bar(frac)}  {detail}"


def render(once: bool) -> None:
    while True:
        job_dirs = sorted(
            (d for d in JOBS_DIR.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        ) if JOBS_DIR.exists() else []

        lines = ["video generation jobs (newest first)", ""]
        if not job_dirs:
            lines.append("  no jobs yet")
        else:
            lines.extend(_job_line(d) for d in job_dirs)

        if once:
            print("\n".join(lines))
            return

        # Clear screen + home cursor, then redraw.
        print("\033[2J\033[H" + "\n".join(lines), flush=True)
        time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="print once and exit")
    args = parser.parse_args()
    try:
        render(args.once)
    except KeyboardInterrupt:
        pass
