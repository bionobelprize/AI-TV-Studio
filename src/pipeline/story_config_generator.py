"""故事配置生成器 for AI-TV-Studio.

接受自由形式的剧集大纲，并使用LLM生成完整的故事配置YAML文件，
包括剧集元数据、带有视觉提示的角色档案以及剧集详细信息，
可供ScriptGenerator使用。

用法::

    from src.pipeline.story_config_generator import StoryConfigGenerator

    gen = StoryConfigGenerator()
    yaml_text = gen.generate(outline="一位侦探和她的搭档...")
    print(yaml_text)
"""

import logging
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import src.model_load as model_load
from src.utils.api_key_config import load_video_api_keys

try:
    from volcenginesdkarkruntime import Ark
except Exception:  # pragma: no cover - optional dependency at runtime
    Ark = None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结构化LLM输出的Pydantic模式
# ---------------------------------------------------------------------------


class VisualCoreOutput(BaseModel):
    reference_prompt: str = Field(
        default="",
        description=(
            "描述角色外貌的文生图提示词，适用于AI图像生成器。"
        ),
    )
    key_features: str = Field(
        default="",
        description="角色最显著视觉特征的逗号分隔列表。",
    )


class CharacterOutput(BaseModel):
    id: str = Field(description="唯一命名标识符，例如 char_detective。")
    name: str = Field(description="脚本中使用的显示名称。")
    age: int = Field(default=30)
    gender: str = Field(default="unspecified", description="male / female / unspecified")
    occupation: str = Field(default="unknown")
    aliases: List[str] = Field(default_factory=list, description="对话中使用的简短昵称列表。")
    personality_traits: List[str] = Field(default_factory=list)
    character_arc: str = Field(default="", description="角色成长的一句话描述。")
    visual_core: VisualCoreOutput = Field(default_factory=VisualCoreOutput)


class SeriesOutput(BaseModel):
    title: str = Field(description="简洁、富有感染力的剧集标题。")
    genre: str = Field(description="主要类型，如 thriller, drama, sci-fi。")
    description: str = Field(description="两句话的剧集前提。")


class StoryConfigLLMOutput(BaseModel):
    """LLM负责生成的所有内容。"""

    series: SeriesOutput
    characters: List[CharacterOutput] = Field(
        description="从大纲中推断出的所有主要角色（通常为2-5个）。"
    )


# ---------------------------------------------------------------------------
# 生成器
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "你是一位专业的电视剧创作者和角色设计师。"
    "根据给定的剧集大纲，提取或创造出一个连贯的剧集概念"
    "和一组角色。输出必须是与模式匹配的有效JSON。"
)

_TEMPLATE = """{system_prompt}

剧集大纲：
{outline}

根据以上大纲：
1. 设计一个引人入胜的剧集标题、主要类型和两句话的剧集前提。
2. 识别每个被命名或暗示的角色，并为每个角色创建完整的档案：
   - 选择一个清晰、唯一命名标识符。
   - 推断年龄、性别、职业、性格特征和一句话的角色弧线。
   - 为文生图模型编写一个的参考提示词，描述他们的外貌。
   - 列出3-5个核心特征（逗号分隔），使其在外观上具有独特性。

{format_instructions}"""


class StoryConfigGenerator:
    """从纯文本大纲生成完整的story_example.yaml。

    LLM负责创造性/推断性工作：
      - 剧集标题、类型和描述
      - 角色档案（姓名、年龄、性格、视觉提示）

    固定的脚手架（工作室路径、运行时标志、API配置路径）由本类组装，从不发送给模型。
    """

    DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    DEFAULT_IMAGE_MODEL = "doubao-seedream-5-0-260128"

    def __init__(
        self,
        episode_number: int = 1,
        image_generation_enabled: bool = True,
        image_model: Optional[str] = None,
        image_size: str = "2K",
        video_api_config_path: str = "config/video_api_keys.yaml",
    ):
        """初始化生成器。

        Args:
            episode_number: 嵌入输出YAML中的剧集编号。
            image_generation_enabled: 是否在生成配置后自动生成角色底图。
            image_model: Ark 图片模型ID，默认使用 doubao-seedream-5-0-260128。
            image_size: 图片尺寸参数，透传给 Ark images.generate。
            video_api_config_path: API配置文件路径（读取 ark.api_key / base_url）。
        """
        self.episode_number = episode_number
        self.image_generation_enabled = image_generation_enabled
        self.image_model = image_model or self.DEFAULT_IMAGE_MODEL
        self.image_size = image_size
        self.video_api_config_path = video_api_config_path

        self._ark_client = None
        self.llm = model_load.load()
        self.parser = JsonOutputParser(pydantic_object=StoryConfigLLMOutput)
        self.prompt = PromptTemplate(
            template=_TEMPLATE,
            input_variables=["outline"],
            partial_variables={
                "system_prompt": _SYSTEM_PROMPT,
                "format_instructions": self.parser.get_format_instructions(),
            },
        )
        self.chain = self.prompt | self.llm | self.parser

    # ------------------------------------------------------------------
    # 公共API
    # ------------------------------------------------------------------

    def generate(self, outline: str) -> str:
        """从自由形式的大纲生成完整的故事配置YAML。

        Args:
            outline: 完整的剧集大纲（一个或多个段落）。

        Returns:
            YAML文本，准备写入``config/story_example.yaml``
            或直接作为``series_config``字典使用。
        """
        llm_data: Dict[str, Any] = self.chain.invoke({"outline": outline})
        config = self._assemble_config(llm_data, outline)
        if self.image_generation_enabled:
            self._generate_character_images(config)
        return self._to_yaml(config)

    def generate_dict(self, outline: str) -> Dict[str, Any]:
        """类似于:meth:`generate`，但将配置作为普通字典返回。

        当您想直接将结果传递给``ScriptGenerator.generate_episode``而不写入文件时很有用。
        """
        llm_data: Dict[str, Any] = self.chain.invoke({"outline": outline})
        config = self._assemble_config(llm_data, outline)
        if self.image_generation_enabled:
            self._generate_character_images(config)
        return config

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _assemble_config(
        self, llm_data: Dict[str, Any], outline: str
    ) -> Dict[str, Any]:
        """将LLM输出与固定脚手架组合成完整的配置字典。"""
        series = llm_data.get("series", {})
        characters = llm_data.get("characters", [])

        return {
            "studio": {
                "output_dir": "./outputs",
                "data_dir": "./data",
                "llm_model": "deepseek-chat",
            },
            "series": {
                "title": series.get("title", "Untitled Series"),
                "genre": series.get("genre", "drama"),
                "description": series.get("description", ""),
            },
            "episode": {
                "number": self.episode_number,
                "outline": outline,
            },
            "characters": [
                self._build_character_dict(c) for c in characters
            ],
            "runtime": {
                "dry_run": True,
                "persist_dir": "./data/generated",
            },
            "api": {
                "video_api_config_path": "config/video_api_keys.yaml",
            },
        }

    def _build_character_dict(self, char: Dict[str, Any]) -> Dict[str, Any]:
        """将LLM角色字典转换为规范的YAML结构。"""
        char_id = char.get("id") or f"char_{uuid.uuid4().hex[:6]}"
        visual = char.get("visual_core") or {}

        return {
            "id": char_id,
            "name": char.get("name", "Unknown"),
            "age": char.get("age", 30),
            "gender": char.get("gender", "unspecified"),
            "occupation": char.get("occupation", "unknown"),
            "aliases": char.get("aliases", []),
            "personality_traits": char.get("personality_traits", []),
            "character_arc": char.get("character_arc", ""),
            "visual_core": {
                "base_image_path": f"./data/characters/{char_id}_base.jpg",
                "reference_prompt": visual.get("reference_prompt", ""),
                "key_features": visual.get("key_features", ""),
                "lora_trigger": None,
                "front_view": None,
                "side_view": None,
                "three_quarter_view": None,
            },
            "voice_id": None,
            "voice_emotion_profile": {},
        }

    def _generate_character_images(self, config: Dict[str, Any]) -> None:
        """为角色生成基础图并保存到 visual_core.base_image_path。"""
        characters = config.get("characters", [])
        if not isinstance(characters, list):
            return

        for character in characters:
            visual_core = character.get("visual_core") or {}
            base_image_path = visual_core.get("base_image_path")
            if not base_image_path:
                logger.warning("角色 %s 缺少 base_image_path，跳过图片生成", character.get("id"))
                continue

            output_path = Path(base_image_path)
            if output_path.exists():
                logger.info("角色底图已存在，跳过生成: %s", output_path)
                continue

            prompt = self._build_character_image_prompt(character)
            try:
                image_url = self._request_character_image_url(prompt)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(image_url, str(output_path))
                logger.info("角色底图生成成功: %s", output_path)
            except Exception as exc:
                logger.warning(
                    "角色 %s 图片生成失败，保留路径占位: %s",
                    character.get("id"),
                    exc,
                )

    def _build_character_image_prompt(self, character: Dict[str, Any]) -> str:
        """构建角色图片生成提示词。"""
        visual_core = character.get("visual_core") or {}
        reference_prompt = visual_core.get("reference_prompt", "")
        key_features = visual_core.get("key_features", "")

        name = character.get("name", "Unknown")
        age = character.get("age", "unspecified")
        gender = character.get("gender", "unspecified")
        occupation = character.get("occupation", "unknown")
        personality = ", ".join(character.get("personality_traits", []))

        prompt_parts = [
            f"single character portrait of {name}",
            f"{age} years old",
            gender,
            occupation,
            "cinematic lighting",
            "highly detailed",
            "professional character concept art",
        ]
        if personality:
            prompt_parts.append(f"personality: {personality}")
        if reference_prompt:
            prompt_parts.append(reference_prompt)
        if key_features:
            prompt_parts.append(f"key features: {key_features}")

        return ", ".join(prompt_parts)

    def _request_character_image_url(self, prompt: str) -> str:
        """调用 Ark images.generate 并返回图片 URL。"""
        client = self._get_ark_client()
        response = client.images.generate(
            model=self.image_model,
            prompt=prompt,
            sequential_image_generation="disabled",
            response_format="url",
            size=self.image_size,
            stream=False,
            watermark=True,
        )
        data = getattr(response, "data", None)
        if not data:
            raise RuntimeError("Ark images.generate 返回空数据")

        url = getattr(data[0], "url", None)
        if not url:
            raise RuntimeError("Ark images.generate 未返回图片 URL")
        return url

    def _get_ark_client(self):
        """懒加载 Ark 客户端，API Key 优先读取 video_api_keys.yaml。"""
        if self._ark_client is not None:
            return self._ark_client

        if Ark is None:
            raise ImportError("volcengine-python-sdk[ark] 未安装")

        cfg = load_video_api_keys(config_path=self.video_api_config_path)
        ark_cfg = cfg.get("ark", {}) if isinstance(cfg, dict) else {}
        api_key = ark_cfg.get("api_key")
        if not api_key:
            raise ValueError("未在 config/video_api_keys.yaml 中找到 ark.api_key")

        base_url = ark_cfg.get("base_url") or self.DEFAULT_ARK_BASE_URL
        self._ark_client = Ark(base_url=base_url, api_key=api_key)
        return self._ark_client

    @staticmethod
    def _to_yaml(config: Dict[str, Any]) -> str:
        """将配置字典序列化为YAML字符串。

        使用``default_flow_style=False``以获得可读的块样式，
        并设置``allow_unicode=True``以避免转义非ASCII字符。
        """
        return yaml.dump(
            config,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )