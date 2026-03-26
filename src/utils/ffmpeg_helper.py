"""FFmpeg helper utilities for AI-TV-Studio.

Provides a high-level Python interface to common FFmpeg operations used
throughout the production pipeline: frame extraction, video concatenation,
cross-fade transitions, and audio mixing.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional


class FFmpegHelper:
    """High-level wrapper around FFmpeg for video processing tasks."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe"):
        """Initialize the FFmpeg helper.

        Args:
            ffmpeg_bin: Path or name of the ``ffmpeg`` executable.
            ffprobe_bin: Path or name of the ``ffprobe`` executable.
        """
        self.ffmpeg = ffmpeg_bin
        self.ffprobe = ffprobe_bin

    def extract_frame(
        self, video_path: str, output_path: str, timestamp: Optional[float] = None
    ) -> str:
        """Extract a single frame from a video file.

        Args:
            video_path: Path to the source video.
            output_path: Path where the frame image will be saved.
            timestamp: Time in seconds at which to extract the frame.
                If None, extracts the last frame.

        Returns:
            Path to the extracted frame image.
        """
        if timestamp is not None:
            cmd = [
                self.ffmpeg,
                "-ss",
                str(timestamp),
                "-i",
                video_path,
                "-vframes",
                "1",
                "-q:v",
                "1",
                output_path,
                "-y",
            ]
        else:
            # Extract the very last frame
            cmd = [
                self.ffmpeg,
                "-sseof",
                "-1",
                "-i",
                video_path,
                "-update",
                "1",
                "-q:v",
                "1",
                output_path,
                "-y",
            ]

        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    def get_duration(self, video_path: str) -> float:
        """Get the duration of a video file in seconds.

        Args:
            video_path: Path to the video file.

        Returns:
            Duration in seconds, or 0.0 if the probe fails.
        """
        try:
            result = subprocess.run(
                [
                    self.ffprobe,
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
        except (subprocess.CalledProcessError, ValueError):
            return 0.0

    def get_resolution(self, video_path: str) -> tuple:
        """Get the width and height of a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            Tuple of (width, height) in pixels, or (0, 0) on failure.
        """
        try:
            result = subprocess.run(
                [
                    self.ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=p=0",
                    video_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            parts = result.stdout.strip().split(",")
            return int(parts[0]), int(parts[1])
        except (subprocess.CalledProcessError, ValueError, IndexError):
            return (0, 0)

    def concatenate_videos(
        self,
        input_paths: List[str],
        output_path: str,
        use_crossfade: bool = False,
        crossfade_duration: float = 0.5,
    ) -> str:
        """Concatenate multiple video files into one.

        Args:
            input_paths: Ordered list of video file paths to concatenate.
            output_path: Path for the output video.
            use_crossfade: If True, apply cross-fade transitions between clips.
            crossfade_duration: Duration of each cross-fade in seconds.

        Returns:
            Path to the concatenated output video.

        Raises:
            ValueError: If fewer than two input files are provided.
        """
        if len(input_paths) < 2:
            raise ValueError("Need at least two videos to concatenate.")

        if not use_crossfade:
            return self._concat_simple(input_paths, output_path)

        return self._concat_with_crossfade(
            input_paths, output_path, crossfade_duration
        )

    def _concat_simple(
        self, input_paths: List[str], output_path: str
    ) -> str:
        """Concatenate videos using the concat demuxer (no re-encoding).

        Args:
            input_paths: Ordered list of video file paths.
            output_path: Path for the output video.

        Returns:
            Path to the output video.
        """
        list_file = output_path + ".concat_list.txt"
        with open(list_file, "w") as fh:
            for path in input_paths:
                abs_path = os.path.abspath(path)
                fh.write(f"file '{abs_path}'\n")
        try:
            subprocess.run(
                [
                    self.ffmpeg,
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    list_file,
                    "-c",
                    "copy",
                    "-y",
                    output_path,
                ],
                check=True,
                capture_output=True,
            )
        finally:
            if os.path.exists(list_file):
                os.remove(list_file)
        return output_path

    def _concat_with_crossfade(
        self,
        input_paths: List[str],
        output_path: str,
        crossfade_duration: float,
    ) -> str:
        """Concatenate videos with cross-fade transitions.

        Args:
            input_paths: Ordered list of video file paths.
            output_path: Path for the output video.
            crossfade_duration: Duration of each cross-fade in seconds.

        Returns:
            Path to the output video.
        """
        inputs: List[str] = []
        for p in input_paths:
            inputs += ["-i", p]

        n = len(input_paths)
        filter_parts: List[str] = []
        durations = [self.get_duration(p) for p in input_paths]

        for i in range(n):
            filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")

        prev = "v0"
        running_offset = 0.0
        for i in range(1, n):
            running_offset += durations[i - 1] - crossfade_duration
            out_label = "vout" if i == n - 1 else f"vx{i}"
            filter_parts.append(
                f"[{prev}][v{i}]xfade=transition=fade:"
                f"duration={crossfade_duration}:"
                f"offset={max(running_offset, 0):.3f}[{out_label}]"
            )
            prev = out_label

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
                "-y",
                output_path,
            ]
        )
        subprocess.run([self.ffmpeg] + cmd, check=True, capture_output=True)
        return output_path

    def add_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        loop_audio: bool = True,
    ) -> str:
        """Mix an audio track into a video file.

        Args:
            video_path: Path to the source video.
            audio_path: Path to the audio file to mix in.
            output_path: Path for the output video with audio.
            loop_audio: If True, loop the audio to match video duration.

        Returns:
            Path to the output video with audio.
        """
        audio_flags = ["-stream_loop", "-1"] if loop_audio else []
        cmd = (
            [self.ffmpeg, "-i", video_path]
            + audio_flags
            + [
                "-i",
                audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                "-y",
                output_path,
            ]
        )
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
