"""Tests for the two-phase video generation logic in AITVStudio."""

from unittest.mock import MagicMock, call, patch

import pytest

from src.models.episode import Episode
from src.models.scene import Scene
from src.models.shot import GenerationMode, Shot
from src.studio import AITVStudio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shot(
    shot_id: str,
    scene_id: str,
    seq: int,
    is_transition: bool = False,
    mode: GenerationMode = GenerationMode.REFERENCE_TO_VIDEO,
) -> Shot:
    return Shot(
        id=shot_id,
        scene_id=scene_id,
        sequence_number=seq,
        action_description=f"action {shot_id}",
        generation_mode=mode,
        duration=8,
        text_prompt=f"prompt {shot_id}",
        is_transition_shot=is_transition,
        transition_type="character_entry" if is_transition else None,
    )


def _make_episode(*shot_lists) -> Episode:
    """Build an Episode from one or more lists of shots (each list = one scene)."""
    scenes = []
    for i, shots in enumerate(shot_lists):
        scenes.append(
            Scene(
                id=f"sc{i + 1}",
                episode_id="ep1",
                scene_number=i + 1,
                location="Test",
                time_of_day="day",
                weather="clear",
                mood="neutral",
                shots=list(shots),
            )
        )
    return Episode(
        id="ep1",
        series_title="Test Series",
        episode_number=1,
        episode_title="Pilot",
        logline="A test.",
        scenes=scenes,
    )


def _make_studio(mcp_side_effect=None):
    """Return an AITVStudio with a mocked MCP server."""
    studio = AITVStudio()
    mcp = MagicMock()
    if mcp_side_effect:
        mcp.call_tool.side_effect = mcp_side_effect
    else:
        # Default: return a unique path per call using a counter.
        counter = {"n": 0}

        def _default_call_tool(tool_name, **kwargs):
            counter["n"] += 1
            return {"path": f"/videos/shot_{counter['n']}.mp4"}

        mcp.call_tool.side_effect = _default_call_tool

    studio._mcp_server = mcp
    return studio


# ---------------------------------------------------------------------------
# Phase ordering: non-transition shots before transition shots
# ---------------------------------------------------------------------------

class TestTwoPhaseOrdering:
    """Verify that non-transition shots are generated before transition shots."""

    def test_non_transition_shots_generated_first(self):
        """Non-transition shots must be sent to MCP before any transition shot."""
        shot_a = _make_shot("a", "sc1", 0)
        trans = _make_shot(
            "trans", "sc1", 1, is_transition=True, mode=GenerationMode.FIRSTLAST_FRAME
        )
        shot_b = _make_shot("b", "sc1", 2)

        episode = _make_episode([shot_a, trans, shot_b])

        generation_order = []

        def tracking_call_tool(tool_name, **kwargs):
            # record which shot is being generated based on prompt
            generation_order.append(tool_name)
            return {"path": f"/videos/{tool_name}_{len(generation_order)}.mp4"}

        studio = AITVStudio()
        mcp = MagicMock()
        mcp.call_tool.side_effect = tracking_call_tool
        studio._mcp_server = mcp

        with patch.object(
            studio.__class__, "_generate_shots", wraps=studio._generate_shots
        ):
            with patch("src.studio.FFmpegHelper") as mock_ffmpeg_cls:
                mock_ffmpeg = MagicMock()
                mock_ffmpeg_cls.return_value = mock_ffmpeg
                mock_ffmpeg.extract_frame.return_value = "/frame.png"

                studio._generate_shots(episode)

        # First two calls are for non-transition shots (shot_a, shot_b via ref2video)
        # third call is for the transition (firstlast_frame)
        assert generation_order[0] == "generate_reference_video"
        assert generation_order[1] == "generate_reference_video"
        assert generation_order[2] == "generate_firstlast_frame_video"

    def test_transition_shot_receives_extracted_frames(self):
        """Transition shot start/end frames must be set from adjacent videos."""
        shot_a = _make_shot("a", "sc1", 0)
        trans = _make_shot(
            "trans", "sc1", 1, is_transition=True, mode=GenerationMode.FIRSTLAST_FRAME
        )
        shot_b = _make_shot("b", "sc1", 2)

        episode = _make_episode([shot_a, trans, shot_b])

        studio = AITVStudio()
        mcp = MagicMock()

        # Phase 1 generates shot_a and shot_b; phase 2 generates trans.
        # Use side_effect to return distinct paths so we can track which
        # frames are extracted from which generated videos.
        paths = iter([
            {"path": "/videos/shot_a.mp4"},   # shot_a (phase 1)
            {"path": "/videos/shot_b.mp4"},   # shot_b (phase 1)
            {"path": "/videos/trans.mp4"},    # trans  (phase 2)
        ])
        mcp.call_tool.side_effect = lambda tool_name, **kwargs: next(paths)
        studio._mcp_server = mcp

        with patch("src.studio.FFmpegHelper") as mock_ffmpeg_cls:
            mock_ffmpeg = MagicMock()
            mock_ffmpeg_cls.return_value = mock_ffmpeg
            mock_ffmpeg.extract_frame.return_value = "/frame.png"

            studio._generate_shots(episode)

        # Two extract_frame calls: tail of shot_a, head of shot_b.
        assert mock_ffmpeg.extract_frame.call_count == 2
        calls = mock_ffmpeg.extract_frame.call_args_list

        # First call: last frame of shot_a (no timestamp argument → None)
        tail_call = calls[0]
        assert tail_call[0][0] == "/videos/shot_a.mp4"
        assert tail_call[1].get("timestamp") is None

        # Second call: first frame of shot_b (timestamp=0.0)
        head_call = calls[1]
        assert head_call[0][0] == "/videos/shot_b.mp4"
        assert head_call[1].get("timestamp") == 0.0

        # Transition shot gets start/end frame paths derived from the video paths.
        assert trans.start_frame_path == "/videos/shot_a_tail.png"
        assert trans.end_frame_path == "/videos/shot_b_head.png"


# ---------------------------------------------------------------------------
# Frame extraction uses correct adjacent shots
# ---------------------------------------------------------------------------

class TestAdjacentShotLookup:
    """Verify the correct prev/next shots are used for frame extraction."""

    def test_uses_nearest_non_transition_neighbours(self):
        """Multiple consecutive transition shots should each find the right neighbours."""
        s1 = _make_shot("s1", "sc1", 0)
        t1 = _make_shot(
            "t1", "sc1", 1, is_transition=True, mode=GenerationMode.FIRSTLAST_FRAME
        )
        s2 = _make_shot("s2", "sc1", 2)
        t2 = _make_shot(
            "t2", "sc1", 3, is_transition=True, mode=GenerationMode.FIRSTLAST_FRAME
        )
        s3 = _make_shot("s3", "sc1", 4)

        episode = _make_episode([s1, t1, s2, t2, s3])

        call_count = {"n": 0}

        def fake_call_tool(tool_name, **kwargs):
            call_count["n"] += 1
            return {"path": f"/v/{call_count['n']}.mp4"}

        studio = AITVStudio()
        mcp = MagicMock()
        mcp.call_tool.side_effect = fake_call_tool
        studio._mcp_server = mcp

        extracted = []

        with patch("src.studio.FFmpegHelper") as mock_ffmpeg_cls:
            mock_ffmpeg = MagicMock()
            mock_ffmpeg_cls.return_value = mock_ffmpeg

            def record_extract(video_path, output_path, timestamp=None):
                extracted.append((video_path, timestamp))
                return output_path

            mock_ffmpeg.extract_frame.side_effect = record_extract

            studio._generate_shots(episode)

        # t1: prev=s1, next=s2 → extract tail of s1, head of s2
        # t2: prev=s2, next=s3 → extract tail of s2, head of s3
        assert len(extracted) == 4

        s1_tail, s2_head, s2_tail, s3_head = extracted

        # tail of s1 (no timestamp)
        assert s1_tail[0] == s1.generated_video_path
        assert s1_tail[1] is None
        # head of s2 (timestamp=0.0)
        assert s2_head[0] == s2.generated_video_path
        assert s2_head[1] == 0.0
        # tail of s2
        assert s2_tail[0] == s2.generated_video_path
        assert s2_tail[1] is None
        # head of s3
        assert s3_head[0] == s3.generated_video_path
        assert s3_head[1] == 0.0

    def test_transition_at_start_has_no_prev_shot(self):
        """Transition shot with no preceding non-transition shot skips tail extraction."""
        trans = _make_shot(
            "t1", "sc1", 0, is_transition=True, mode=GenerationMode.FIRSTLAST_FRAME
        )
        s1 = _make_shot("s1", "sc1", 1)

        episode = _make_episode([trans, s1])

        studio = AITVStudio()
        mcp = MagicMock()
        mcp.call_tool.return_value = {"path": "/v/1.mp4"}
        studio._mcp_server = mcp

        extracted_args = []

        with patch("src.studio.FFmpegHelper") as mock_ffmpeg_cls:
            mock_ffmpeg = MagicMock()
            mock_ffmpeg_cls.return_value = mock_ffmpeg

            def record(video_path, output_path, timestamp=None):
                extracted_args.append((video_path, timestamp))
                return output_path

            mock_ffmpeg.extract_frame.side_effect = record

            studio._generate_shots(episode)

        # Only head-frame extraction (for next shot), no tail extraction.
        assert len(extracted_args) > 0
        assert all(ts == 0.0 for (_, ts) in extracted_args)
        assert trans.start_frame_path is None  # no prev shot


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

class TestFailureHandling:
    def test_ffmpeg_failure_is_logged_gracefully(self):
        """If frame extraction fails, generation still proceeds without crashing."""
        shot_a = _make_shot("a", "sc1", 0)
        trans = _make_shot(
            "t1", "sc1", 1, is_transition=True, mode=GenerationMode.FIRSTLAST_FRAME
        )
        shot_b = _make_shot("b", "sc1", 2)

        episode = _make_episode([shot_a, trans, shot_b])

        studio = AITVStudio()
        mcp = MagicMock()
        call_count = {"n": 0}

        def fake_call(tool_name, **kwargs):
            call_count["n"] += 1
            return {"path": f"/v/{call_count['n']}.mp4"}

        mcp.call_tool.side_effect = fake_call
        studio._mcp_server = mcp

        with patch("src.studio.FFmpegHelper") as mock_ffmpeg_cls:
            mock_ffmpeg = MagicMock()
            mock_ffmpeg_cls.return_value = mock_ffmpeg
            mock_ffmpeg.extract_frame.side_effect = RuntimeError("ffmpeg not found")

            # Should not raise
            studio._generate_shots(episode)

        # All three shots attempted; transition shot still tried even without frames.
        assert mcp.call_tool.call_count == 3

    def test_all_shots_fail_raises_runtime_error(self):
        """RuntimeError raised when every shot fails to generate."""
        shot_a = _make_shot("a", "sc1", 0)
        episode = _make_episode([shot_a])

        studio = AITVStudio()
        mcp = MagicMock()
        mcp.call_tool.side_effect = Exception("API error")
        studio._mcp_server = mcp

        with pytest.raises(RuntimeError, match="Shot generation failed for all shots"):
            studio._generate_shots(episode)

    def test_partial_failure_does_not_raise(self):
        """If at least one shot succeeds, no RuntimeError is raised."""
        shot_a = _make_shot("a", "sc1", 0)
        shot_b = _make_shot("b", "sc1", 1)
        episode = _make_episode([shot_a, shot_b])

        call_count = {"n": 0}

        def mixed_results(tool_name, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("first shot fails")
            return {"path": "/v/ok.mp4"}

        studio = AITVStudio()
        mcp = MagicMock()
        mcp.call_tool.side_effect = mixed_results
        studio._mcp_server = mcp

        # Should not raise
        studio._generate_shots(episode)

        assert shot_a.generation_error is not None
        assert shot_b.generated_video_path == "/v/ok.mp4"


# ---------------------------------------------------------------------------
# _generate_single_shot helper
# ---------------------------------------------------------------------------

class TestGenerateSingleShot:
    def _studio_with_mcp(self, return_value=None, side_effect=None):
        studio = AITVStudio()
        mcp = MagicMock()
        if side_effect:
            mcp.call_tool.side_effect = side_effect
        else:
            mcp.call_tool.return_value = return_value or {"path": "/v/1.mp4"}
        studio._mcp_server = mcp
        return studio, mcp

    def test_dispatches_firstlast_frame(self):
        studio, mcp = self._studio_with_mcp()
        shot = _make_shot("s1", "sc1", 0, mode=GenerationMode.FIRSTLAST_FRAME)
        shot.start_frame_path = "/start.png"
        shot.end_frame_path = "/end.png"
        result = studio._generate_single_shot(shot)
        assert result is True
        mcp.call_tool.assert_called_once_with(
            "generate_firstlast_frame_video",
            start_frame_path="/start.png",
            end_frame_path="/end.png",
            prompt=shot.text_prompt,
            duration=shot.duration,
        )

    def test_dispatches_first_frame(self):
        studio, mcp = self._studio_with_mcp()
        shot = _make_shot("s1", "sc1", 0, mode=GenerationMode.FIRST_FRAME)
        shot.start_frame_path = "/start.png"
        result = studio._generate_single_shot(shot)
        assert result is True
        mcp.call_tool.assert_called_once_with(
            "generate_first_frame_video",
            first_frame_path="/start.png",
            prompt=shot.text_prompt,
            duration=shot.duration,
        )

    def test_dispatches_reference_to_video(self):
        studio, mcp = self._studio_with_mcp()
        shot = _make_shot("s1", "sc1", 0, mode=GenerationMode.REFERENCE_TO_VIDEO)
        shot.reference_images = ["/ref1.png"]
        result = studio._generate_single_shot(shot)
        assert result is True
        mcp.call_tool.assert_called_once_with(
            "generate_reference_video",
            prompt=shot.text_prompt,
            reference_images=["/ref1.png"],
            duration=shot.duration,
        )

    def test_dispatches_text_to_video(self):
        studio, mcp = self._studio_with_mcp()
        shot = _make_shot("s1", "sc1", 0, mode=GenerationMode.TEXT_TO_VIDEO)
        result = studio._generate_single_shot(shot)
        assert result is True
        mcp.call_tool.assert_called_once_with(
            "generate_text_to_video",
            prompt=shot.text_prompt,
            duration=shot.duration,
        )

    def test_returns_false_on_mcp_exception(self):
        studio, mcp = self._studio_with_mcp(side_effect=Exception("boom"))
        shot = _make_shot("s1", "sc1", 0)
        result = studio._generate_single_shot(shot)
        assert result is False
        assert "boom" in shot.generation_error

    def test_returns_false_when_path_is_none(self):
        studio, mcp = self._studio_with_mcp(return_value={"path": None})
        shot = _make_shot("s1", "sc1", 0)
        result = studio._generate_single_shot(shot)
        assert result is False
