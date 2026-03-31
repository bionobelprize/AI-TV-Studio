"""Standalone FFmpeg invocation smoke test for AI-TV-Studio.

This script validates both FFmpeg call paths used by the project:
1) src.utils.ffmpeg_helper.FFmpegHelper wrapper methods.
2) src.pipeline.video_assembler.VideoAssembler direct subprocess calls.

Usage:
    python test_ffmpeg_invocation.py
    python test_ffmpeg_invocation.py --work-dir data/output/ffmpeg_smoke
"""

from __future__ import annotations

import argparse
import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

from src.models.episode import Episode
from src.pipeline.video_assembler import VideoAssembler, VideoSegment
from src.utils.ffmpeg_helper import FFmpegHelper


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    """Run a command and raise with useful context if it fails."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Command failed: "
            + " ".join(cmd)
            + "\nstdout:\n"
            + (exc.stdout or "")
            + "\nstderr:\n"
            + (exc.stderr or "")
        ) from exc


def require_binary(name: str, override: str = "") -> str:
    """Resolve an executable path from override or PATH, or raise error."""
    if override:
        candidate = Path(override)
        if candidate.exists():
            return str(candidate)
        path = shutil.which(override)
        if path:
            return path
        raise RuntimeError(
            f"Provided binary '{override}' for {name} does not exist or is not executable."
        )

    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            f"Required binary '{name}' was not found in PATH. "
            "Please install FFmpeg and ensure ffmpeg/ffprobe are available."
        )
    return path


def create_color_clip(
    output_path: Path,
    color: str,
    ffmpeg_bin: str,
    duration: float = 1.6,
) -> None:
    """Create a small synthetic MP4 clip using FFmpeg's lavfi color source."""
    cmd = [
        ffmpeg_bin,
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=640x360:d={duration}:r=24",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-an",
        "-y",
        str(output_path),
    ]
    run_cmd(cmd)


def run_helper_path(work_dir: Path, ffmpeg_bin: str, ffprobe_bin: str) -> Dict[str, object]:
    """Exercise FFmpegHelper wrapper methods."""
    helper = FFmpegHelper(ffmpeg_bin=ffmpeg_bin, ffprobe_bin=ffprobe_bin)

    clip_a = work_dir / "clip_a.mp4"
    clip_b = work_dir / "clip_b.mp4"
    create_color_clip(clip_a, "red", ffmpeg_bin)
    create_color_clip(clip_b, "blue", ffmpeg_bin)

    head_frame = work_dir / "clip_a_head.png"
    tail_frame = work_dir / "clip_b_tail.png"
    helper.extract_frame(str(clip_a), str(head_frame), timestamp=0.0)
    helper.extract_frame(str(clip_b), str(tail_frame))

    duration_a = helper.get_duration(str(clip_a))
    resolution_a = helper.get_resolution(str(clip_a))

    concat_copy = work_dir / "concat_copy.mp4"
    concat_xfade = work_dir / "concat_xfade.mp4"
    helper.concatenate_videos([str(clip_a), str(clip_b)], str(concat_copy), use_crossfade=False)
    helper.concatenate_videos(
        [str(clip_a), str(clip_b)],
        str(concat_xfade),
        use_crossfade=True,
        crossfade_duration=0.4,
    )

    return {
        "clip_a": str(clip_a),
        "clip_b": str(clip_b),
        "head_frame": str(head_frame),
        "tail_frame": str(tail_frame),
        "duration_clip_a": duration_a,
        "resolution_clip_a": list(resolution_a),
        "concat_copy": str(concat_copy),
        "concat_xfade": str(concat_xfade),
        "outputs_exist": {
            "head_frame": head_frame.exists(),
            "tail_frame": tail_frame.exists(),
            "concat_copy": concat_copy.exists(),
            "concat_xfade": concat_xfade.exists(),
        },
    }


def run_assembler_direct_path(
    work_dir: Path,
    segment_paths: List[Path],
    ffmpeg_bin: str,
    ffprobe_bin: str,
) -> Dict[str, object]:
    """Exercise VideoAssembler direct ffprobe/ffmpeg subprocess paths."""
    assembler = VideoAssembler(output_dir=str(work_dir / "assembled"))
    # Override binaries so this test can run even when PATH is not configured.
    os.environ["PATH"] = (
        str(Path(ffmpeg_bin).parent)
        + os.pathsep
        + str(Path(ffprobe_bin).parent)
        + os.pathsep
        + os.environ.get("PATH", "")
    )

    durations = [assembler._get_video_duration(str(p)) for p in segment_paths]
    segments = [
        VideoSegment(path=str(path), duration=durations[i], has_transition=i > 0)
        for i, path in enumerate(segment_paths)
    ]

    episode = Episode(
        id="ffmpeg_test_episode",
        series_title="FFmpeg Test",
        episode_number=1,
        episode_title="Invocation Smoke",
        logline="Smoke test for ffmpeg integration.",
        scenes=[],
    )

    output = assembler._ffmpeg_assemble(segments, episode)
    return {
        "segment_durations": durations,
        "assembled_output": output,
        "assembled_exists": Path(output).exists(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ffmpeg invocation smoke tests.")
    parser.add_argument(
        "--work-dir",
        default="",
        help=(
            "Optional working directory. If omitted, a temporary directory is created "
            "and kept for inspection."
        ),
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default=os.environ.get("FFMPEG_BIN", ""),
        help="Optional ffmpeg binary path (or command name).",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default=os.environ.get("FFPROBE_BIN", ""),
        help="Optional ffprobe binary path (or command name).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="ffmpeg_smoke_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] work_dir={work_dir}")

    try:
        ffmpeg_path = require_binary("ffmpeg", args.ffmpeg_bin)
        ffprobe_path = require_binary("ffprobe", args.ffprobe_bin)
        print(f"[INFO] ffmpeg={ffmpeg_path}")
        print(f"[INFO] ffprobe={ffprobe_path}")

        version = run_cmd([ffmpeg_path, "-version"]).stdout.splitlines()[0]
        print(f"[INFO] {version}")

        helper_result = run_helper_path(work_dir, ffmpeg_path, ffprobe_path)

        segment_paths = [
            Path(helper_result["clip_a"]),
            Path(helper_result["clip_b"]),
        ]
        assembler_result = run_assembler_direct_path(
            work_dir,
            segment_paths,
            ffmpeg_path,
            ffprobe_path,
        )

        report = {
            "status": "ok",
            "work_dir": str(work_dir),
            "helper_path": helper_result,
            "assembler_direct_path": assembler_result,
        }
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0
    except Exception as exc:
        print("[ERROR] ffmpeg smoke test failed")
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
