"""Tests for the MCP Video Director Server."""

import pytest
from unittest.mock import MagicMock

from src.mcp.video_director_server import VideoDirectorServer


def _make_server() -> VideoDirectorServer:
    api_client = MagicMock()
    api_client.generate_video.return_value = {"path": "/output/video.mp4"}
    api_client.generate_image.return_value = {"path": "/output/image.png"}
    return VideoDirectorServer(api_client=api_client)


class TestVideoDirectorServer:
    def setup_method(self):
        self.server = _make_server()

    def test_list_tools_returns_all(self):
        tools = self.server.list_tools()
        assert "generate_text_to_video" in tools
        assert "generate_firstlast_frame_video" in tools
        assert "generate_first_frame_video" in tools
        assert "generate_reference_video" in tools
        assert "generate_image" in tools
        assert "generate_reference_image" in tools

    def test_call_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            self.server.call_tool("nonexistent_tool")

    def test_generate_text_to_video(self):
        result = self.server.generate_text_to_video(
            prompt="A detective walking down a rain-soaked street",
            duration=8,
        )
        assert result["path"] == "/output/video.mp4"
        assert result["duration"] == 8
        assert result["mode"] == "txt2video"
        self.server.api.generate_video.assert_called_once()

    def test_generate_firstlast_frame_video(self):
        result = self.server.generate_firstlast_frame_video(
            start_frame_path="/frames/start.png",
            end_frame_path="/frames/end.png",
            prompt="Character enters",
            duration=8,
        )
        assert result["mode"] == "firstlast_frame"
        assert result["duration"] == 8
        call_kwargs = self.server.api.generate_video.call_args.kwargs
        assert call_kwargs["mode"] == "firstlast_frame"
        assert call_kwargs["start_frame"] == "/frames/start.png"
        assert call_kwargs["end_frame"] == "/frames/end.png"

    def test_generate_first_frame_video(self):
        result = self.server.generate_first_frame_video(
            first_frame_path="/frames/first.png",
            prompt="Camera slowly pulls back",
            duration=7,
        )
        assert result["mode"] == "first_frame"
        assert result["duration"] == 7
        call_kwargs = self.server.api.generate_video.call_args.kwargs
        assert call_kwargs["mode"] == "first_frame"
        assert call_kwargs["first_frame"] == "/frames/first.png"

    def test_generate_reference_video(self):
        result = self.server.generate_reference_video(
            prompt="Detective examines clue",
            reference_images=["/img/char.png"],
            duration=10,
        )
        assert result["mode"] == "ref2video"
        assert result["duration"] == 10
        assert result["mcp_trace"]["tool_name"] == "generate_reference_video"
        assert result["mcp_trace"]["request"]["reference_images"] == ["/img/char.png"]

    def test_generate_image(self):
        result = self.server.generate_image(
            prompt="Office interior, cinematic"
        )
        assert result["path"] == "/output/image.png"
        assert result["mode"] == "txt2image"

    def test_generate_reference_image(self):
        result = self.server.generate_reference_image(
            prompt="Detective Lin in office",
            reference_images=["/img/lin.png"],
            background_image="/bg/office.png",
            lighting_match=True,
        )
        assert result["mode"] == "ref2image"
        call_kwargs = self.server.api.generate_image.call_args.kwargs
        assert call_kwargs["lighting_match"] is True

    def test_call_tool_dispatch(self):
        result = self.server.call_tool(
            "generate_text_to_video",
            prompt="A rainy evening",
            duration=6,
        )
        assert result["mode"] == "txt2video"
