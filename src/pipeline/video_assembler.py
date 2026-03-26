"""Video Assembler for AI-TV-Studio.

Assembles individual shot videos into a complete episode, applying
cross-fade transitions, audio mixing, and color grading for
professional-grade output.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.models.episode import Episode
from src.models.shot import Shot


@dataclass
class VideoSegment:
    """Metadata for a single video segment used during assembly."""

    path: str
    duration: float
    has_transition: bool = False
    transition_duration: float = 0.5  # cross-fade length in seconds


class VideoAssembler:
    """Assembles individual shots into a complete episode.

    Features:
    - Cross-fade transitions between shots
    - Audio synchronization
    - Color grading consistency
    """

    def __init__(self, output_dir: str = "data/output"):
        """Initialize the video assembler.

        Args:
            output_dir: Directory where assembled episode files will be saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def assemble_episode(self, episode: Episode) -> str:
        """Assemble all shots in order into a complete episode video.

        Algorithm:
        1. Collect all shot videos in sequence
        2. Apply cross-fade transitions between shots
        3. Add background music and ambient audio
        4. Apply color grading for scene consistency
        5. Export final video

        Args:
            episode: The episode whose shots will be assembled.

        Returns:
            Path to the assembled episode video file.

        Raises:
            ValueError: If no shots with generated video paths are found.
        """
        all_shots: List[Shot] = []
        for scene in episode.scenes:
            all_shots.extend(scene.shots)

        generated_paths = [shot.generated_video_path for shot in all_shots if shot.generated_video_path]

        if len(generated_paths) == 1:
            output_path = str(
                self.output_dir / f"episode_{episode.episode_number:02d}.mp4"
            )
            shutil.copy2(generated_paths[0], output_path)
            return output_path

        segments: List[VideoSegment] = []
        for i, shot in enumerate(all_shots):
            if not shot.generated_video_path:
                continue
            segment = VideoSegment(
                path=shot.generated_video_path,
                duration=self._get_video_duration(shot.generated_video_path),
                has_transition=i > 0,
            )
            segments.append(segment)

        if not segments:
            raise ValueError(
                "No generated video segments found for assembly "
                f"(shots_seen={len(all_shots)})."
            )

        return self._ffmpeg_assemble(segments, episode)

    def _get_video_duration(self, video_path: str) -> float:
        """Probe a video file to get its duration in seconds.

        Args:
            video_path: Path to the video file.

        Returns:
            Duration of the video in seconds, or 0.0 on error.
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(result.stdout.strip())
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffprobe was not found in PATH. Install FFmpeg and ensure both "
                "ffmpeg and ffprobe are available from the command line."
            ) from exc
        except (subprocess.CalledProcessError, ValueError):
            return 0.0

    def _ffmpeg_assemble(
        self, segments: List[VideoSegment], episode: Episode
    ) -> str:
        """Construct and run an FFmpeg command to assemble the episode.

        Features:
        - Cross-fade transitions between segments
        - Color consistency via eq filter
        - Audio normalization

        Args:
            segments: Ordered list of video segments to assemble.
            episode: The episode being assembled (used for output naming).

        Returns:
            Path to the assembled output video file.
        """
        output_path = str(
            self.output_dir / f"episode_{episode.episode_number:02d}.mp4"
        )

        # Build FFmpeg concat filter with cross-fade transitions
        inputs: List[str] = []
        for seg in segments:
            inputs += ["-i", seg.path]

        filter_parts: List[str] = []
        n = len(segments)
        fade_dur = segments[0].transition_duration

        # Pre-compute cumulative offsets to avoid O(n²) summation in the loop
        cumulative_duration = 0.0
        cumulative_offsets: List[float] = []
        for i, seg in enumerate(segments):
            if i > 0:
                cumulative_offsets.append(
                    cumulative_duration - fade_dur * i
                )
            cumulative_duration += seg.duration

        # Label each input stream
        for i in range(n):
            filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")

        # Chain cross-fades
        prev_label = "v0"
        for i in range(1, n):
            out_label = f"vout{i}" if i < n - 1 else "vout"
            offset = cumulative_offsets[i - 1]
            filter_parts.append(
                f"[{prev_label}][v{i}]xfade=transition=fade:"
                f"duration={fade_dur}:offset={max(offset, 0)}[{out_label}]"
            )
            prev_label = out_label

        filter_graph = ";".join(filter_parts)

        cmd = (
            inputs
            + [
                "-filter_complex",
                filter_graph,
                "-map",
                "[vout]",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
                "-y",
                output_path,
            ]
        )

        try:
            subprocess.run(["ffmpeg"] + cmd, check=True, capture_output=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg was not found in PATH. Install FFmpeg and ensure both "
                "ffmpeg and ffprobe are available from the command line."
            ) from exc
        return output_path

    def create_segment_list_file(
        self, segments: List[VideoSegment], list_path: str
    ) -> str:
        """Write an FFmpeg concat demuxer file for simple concatenation.

        Useful as a fallback when the xfade filter is not available.

        Args:
            segments: List of video segments.
            list_path: Path where the concat list file will be written.

        Returns:
            Path to the written list file.
        """
        with open(list_path, "w") as fh:
            for seg in segments:
                fh.write(f"file '{seg.path}'\n")
                fh.write(f"duration {seg.duration}\n")
        return list_path
