from pathlib import Path
import subprocess
import unittest

from src.models.episode import Episode
from src.models.scene import Scene
from src.models.shot import Shot
from src.pipeline.video_assembler import VideoAssembler, VideoSegment


def test_single_segment_episode_copies_without_ffprobe(tmp_path, monkeypatch):
    source_video = tmp_path / "shot.mp4"
    source_video.write_bytes(b"fake video bytes")

    shot = Shot(
        id="shot_1",
        scene_id="scene_1",
        sequence_number=1,
        action_description="A single completed shot.",
        generated_video_path=str(source_video),
    )
    scene = Scene(
        id="scene_1",
        episode_id="ep_1",
        scene_number=1,
        location="Test set",
        time_of_day="day",
        weather="clear",
        mood="neutral",
        shots=[shot],
    )
    episode = Episode(
        id="ep_1",
        series_title="Test Series",
        episode_number=1,
        episode_title="Test Episode",
        logline="A single-shot test episode.",
        scenes=[scene],
    )

    assembler = VideoAssembler(output_dir=str(tmp_path / "final_output"))

    def fail_if_called(_video_path: str) -> float:
        raise AssertionError("ffprobe should not be required for a single segment")

    monkeypatch.setattr(assembler, "_get_video_duration", fail_if_called)

    output_path = Path(assembler.assemble_episode(episode))

    assert output_path.exists()
    assert output_path.read_bytes() == source_video.read_bytes()


def test_ffmpeg_assemble_normalizes_resolution_before_xfade(tmp_path, monkeypatch):
    episode = Episode(
        id="ep_1",
        series_title="Test Series",
        episode_number=1,
        episode_title="Test Episode",
        logline="A mixed-resolution episode.",
        scenes=[],
    )
    segments = [
        VideoSegment(path="a.mp4", duration=6.0),
        VideoSegment(path="b.mp4", duration=6.0, has_transition=True),
    ]
    assembler = VideoAssembler(output_dir=str(tmp_path / "final_output"))

    def fake_resolution(video_path: str):
        if video_path == "a.mp4":
            return (1280, 720)
        return (1248, 704)

    recorded = {}

    def fake_run(cmd, check, capture_output, text):
        recorded["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(assembler, "_get_video_resolution", fake_resolution)
    monkeypatch.setattr(subprocess, "run", fake_run)

    assembler._ffmpeg_assemble(segments, episode)

    cmd = recorded["cmd"]
    filter_graph = cmd[cmd.index("-filter_complex") + 1]
    assert "scale=1280:720:force_original_aspect_ratio=decrease" in filter_graph
    assert "pad=1280:720:(ow-iw)/2:(oh-ih)/2" in filter_graph
    assert "format=yuv420p" in filter_graph
    assert "xfade=transition=fade:duration=0.5:offset=5.5" in filter_graph


def test_ffmpeg_assemble_surfaces_stderr(tmp_path, monkeypatch):
    episode = Episode(
        id="ep_1",
        series_title="Test Series",
        episode_number=1,
        episode_title="Test Episode",
        logline="An episode with a failed assembly.",
        scenes=[],
    )
    segments = [
        VideoSegment(path="a.mp4", duration=6.0),
        VideoSegment(path="b.mp4", duration=6.0, has_transition=True),
    ]
    assembler = VideoAssembler(output_dir=str(tmp_path / "final_output"))

    monkeypatch.setattr(assembler, "_get_video_resolution", lambda _path: (1280, 720))

    def fake_run(cmd, check, capture_output, text):
        raise subprocess.CalledProcessError(
            1,
            cmd,
            output="",
            stderr="xfade filter failed",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with unittest.TestCase().assertRaisesRegex(RuntimeError, "xfade filter failed"):
        assembler._ffmpeg_assemble(segments, episode)
