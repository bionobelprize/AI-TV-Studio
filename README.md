# AI-TV-Studio

AI-TV-Studio is an automated TV episode production framework that orchestrates:

- LLM script generation
- shot planning and transition insertion
- multi-mode video generation (text, first-frame, first-last-frame, references)
- episode assembly via FFmpeg

## Install

```bash
pip install -r requirements.txt
```

Make sure `ffmpeg` and `ffprobe` are available in your PATH.

## API Key Configuration

Video provider keys are managed in a dedicated file:

- `config/video_api_keys.yaml`

Fill `ark.api_key` before running Ark-based generation.

## Ark Video Generation Modes

The Ark adapter (`src/mcp/ark_video_client.py`) supports:

- `txt2video`: prompt only
- `first_frame`: prompt + first frame image
- `firstlast_frame`: prompt + first and last frame images
- `ref2video`: prompt + 1..N reference images

It also includes task polling and automatic media download to `outputs/`.

## Minimal Example

```python
from src.studio import AITVStudio
from src.mcp import ArkVideoAPIClient, VideoDirectorServer


class MyLLMClient:
	def chat(self, model, messages):
		# Return JSON text matching script schema expected by ScriptGenerator.
		raise NotImplementedError


studio = AITVStudio(config={"output_dir": "./output", "data_dir": "./data"})
studio.configure_llm(MyLLMClient(), model="gpt-4")

ark_client = ArkVideoAPIClient(config_path="config/video_api_keys.yaml")
studio.configure_mcp(VideoDirectorServer(api_client=ark_client))

# Then register characters and call studio.produce_episode(...)
```
