from pathlib import Path

from src.models.episode import Episode
from src.models.scene import Scene
from src.models.shot import Shot
from src.pipeline.video_assembler import VideoAssembler


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
