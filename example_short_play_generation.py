"""
完整示例：通过配置文件生成短剧
================================

此示例演示 AI-TV-Studio 的通用工作流程，包括：
1. 读取剧情配置（角色、系列信息、episode_outline）
2. 脚本生成 - LLM 调用（消耗 token）
3. 视频生成 - 视频 API 调用（消耗 token，价格较高）
4. 视频组装

说明：
- 本脚本不包含硬编码剧情内容。
- 所有剧情与角色信息均从 YAML 配置文件读取。
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.mcp.ark_video_client import ArkVideoAPIClient
from src.mcp.video_director_server import VideoDirectorServer
from src.models.character import Character, CharacterVisualCore
from src.studio import AITVStudio
import src.model_load as model_load


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_STORY_CONFIG = "config/story_example.yaml"


def load_story_config(config_path: str) -> Dict[str, Any]:
    """读取并校验剧情配置。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"剧情配置不存在: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    required_keys = ["studio", "series", "episode", "characters"]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"剧情配置缺少必要字段: {missing}")

    if not config["episode"].get("outline"):
        raise ValueError("剧情配置中 episode.outline 不能为空")

    if not isinstance(config["characters"], list) or not config["characters"]:
        raise ValueError("剧情配置中 characters 必须是非空列表")

    return config


def build_character(character_cfg: Dict[str, Any]) -> Character:
    """从配置构建 Character 对象。"""
    visual_cfg = character_cfg.get("visual_core") or {}
    if not visual_cfg.get("base_image_path"):
        raise ValueError(
            f"角色 {character_cfg.get('id', '<unknown>')} 缺少 visual_core.base_image_path"
        )

    visual_core = CharacterVisualCore(
        base_image_path=visual_cfg["base_image_path"],
        reference_prompt=visual_cfg.get("reference_prompt", ""),
        key_features=visual_cfg.get("key_features", ""),
        lora_trigger=visual_cfg.get("lora_trigger"),
        front_view=visual_cfg.get("front_view"),
        side_view=visual_cfg.get("side_view"),
        three_quarter_view=visual_cfg.get("three_quarter_view"),
    )

    return Character(
        id=character_cfg["id"],
        name=character_cfg["name"],
        age=int(character_cfg.get("age", 0)),
        gender=character_cfg.get("gender", "unknown"),
        occupation=character_cfg.get("occupation", "unknown"),
        aliases=character_cfg.get("aliases", []),
        personality_traits=character_cfg.get("personality_traits", []),
        character_arc=character_cfg.get("character_arc", ""),
        visual_core=visual_core,
        voice_id=character_cfg.get("voice_id"),
        voice_emotion_profile=character_cfg.get("voice_emotion_profile", {}),
    )


def create_characters(config: Dict[str, Any]) -> List[Character]:
    """根据配置创建角色对象列表。"""
    logger.info("[数据结构构建] 从配置创建角色...")
    characters = [build_character(item) for item in config["characters"]]
    logger.info("已创建 %d 个角色", len(characters))
    return characters


def create_series_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """根据配置创建 Series 配置字典。"""
    logger.info("[数据结构构建] 从配置创建 Series 信息...")

    series_cfg = config["series"]
    raw_characters = config["characters"]

    series_characters = series_cfg.get("characters")
    if not series_characters:
        series_characters = [
            {
                "id": c["id"],
                "name": c["name"],
                "occupation": c.get("occupation", "unknown"),
            }
            for c in raw_characters
        ]

    return {
        "title": series_cfg.get("title", "Untitled Series"),
        "genre": series_cfg.get("genre", "drama"),
        "description": series_cfg.get("description", ""),
        "characters": series_characters,
    }


def persist_episode_snapshot(
    episode,
    stage: str,
    output_dir: str = "./data/generated",
) -> Dict[str, str]:
    """将 Episode/Scene/Shot 数据落盘为 JSON 快照。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    episode_number = getattr(episode, "episode_number", 0)
    episode_file = output_path / f"episode_{episode_number:02d}_{stage}.json"

    episode_payload = asdict(episode)

    with episode_file.open("w", encoding="utf-8") as f:
        json.dump(episode_payload, f, ensure_ascii=False, indent=2)

    logger.info("[自动落盘] 已保存 Episode 快照: %s", episode_file)

    return {
        "episode": str(episode_file),
    }


def initialize_studio(config: Dict[str, Any]) -> AITVStudio:
    """根据配置初始化 AI-TV-Studio。"""
    logger.info("[数据结构构建] 初始化 AI-TV-Studio...")

    studio_cfg = config["studio"]
    studio = AITVStudio(
        config={
            "output_dir": studio_cfg.get("output_dir", "./outputs"),
            "data_dir": studio_cfg.get("data_dir", "./data"),
            "llm_model": studio_cfg.get("llm_model", "deepseek-chat"),
        }
    )
    return studio


def configure_studio_with_apis(studio: AITVStudio, config: Dict[str, Any]) -> None:
    """配置 Studio 的 LLM 和视频 API。"""
    model_name = config["studio"].get("llm_model", "deepseek-chat")
    video_api_cfg = config.get("api", {}).get("video_api_config_path", "config/video_api_keys.yaml")

    logger.info("[配置] 加载 LLM 客户端...")
    llm = model_load.load()
    studio.configure_llm(llm, model=model_name)
    logger.info("LLM 配置完成")

    logger.info("[配置] 配置 Ark 视频 API 客户端...")
    try:
        ark_client = ArkVideoAPIClient(config_path=video_api_cfg, output_dir="./outputs")
        video_director = VideoDirectorServer(api_client=ark_client)
        studio.configure_mcp(video_director)
        logger.info("视频 API 配置完成")
    except Exception as exc:
        logger.warning("视频 API 配置失败: %s，视频生成将使用 DRY RUN 模式", exc)
        studio._mcp_server = None


def generate_episode_script(
    studio: AITVStudio,
    series_config: Dict[str, Any],
    episode_outline: str,
    episode_number: int,
):
    """调用 LLM 生成剧集脚本。"""
    logger.info("=" * 70)
    logger.info("步骤2：脚本生成")
    logger.info("=" * 70)

    logger.info("[LLM 调用] 正在生成剧集脚本...")
    logger.info("剧情大纲长度: %d 字符", len(episode_outline))

    episode = studio.script_generator.generate_episode(
        series_config=series_config,
        episode_outline=episode_outline,
        episode_number=episode_number,
    )

    logger.info("脚本生成完成")
    logger.info("- 剧集标题: %s", episode.episode_title)
    logger.info("- Logline: %s", episode.logline)
    logger.info("- 场景数: %d", len(episode.scenes))

    print("\n" + "=" * 70)
    print("[数据结构构建] 生成的 Episode 结构")
    print("=" * 70)
    print(f"Episode ID: {episode.id}")
    print(f"Series Title: {episode.series_title}")
    print(f"Episode Number: {episode.episode_number}")
    print(f"Episode Title: {episode.episode_title}")
    print(f"Logline: {episode.logline}\n")

    for scene_idx, scene in enumerate(episode.scenes, 1):
        print(f"  Scene {scene_idx}: {scene.id}")
        print(f"    Location: {scene.location}")
        print(f"    Time: {scene.time_of_day}")
        print(f"    Shots: {len(scene.shots)}")
        for shot_idx, shot in enumerate(scene.shots, 1):
            print(f"      Shot {shot_idx}: {shot.id} (mode={shot.generation_mode.value})")
            print(f"        Duration: {shot.duration}s")
            print(f"        Prompt: {shot.text_prompt[:60]}...")
            print()

    return episode


def plan_shots(studio: AITVStudio, episode):
    """规划镜头与转场，不调用外部 API。"""
    logger.info("=" * 70)
    logger.info("步骤3：镜头规划")
    logger.info("=" * 70)
    logger.info("[数据结构构建] 正在规划镜头和转场...")

    from src.algorithms.shot_planner import ShotPlanner

    planner = ShotPlanner(character_registry=studio.asset_manager.get_all_characters())
    episode = planner.plan_episode(episode)

    total_shots = sum(len(scene.shots) for scene in episode.scenes)
    logger.info("镜头规划完成，总镜头数: %d", total_shots)

    print("\n" + "=" * 70)
    print("[数据结构构建] 规划完成后的镜头信息")
    print("=" * 70)
    for scene in episode.scenes:
        print(f"\nScene: {scene.location} ({scene.time_of_day})")
        for shot in scene.shots:
            print(f"  - Shot {shot.sequence_number}: {shot.action_description[:50]}...")
            print(f"    Mode: {shot.generation_mode.value} | Duration: {shot.duration}s")
            if shot.text_prompt:
                print(f"    Prompt: {shot.text_prompt[:60]}...")

    return episode


def generate_videos(studio: AITVStudio, episode, dry_run: bool = False):
    """为每个镜头生成视频。"""
    logger.info("=" * 70)
    logger.info("步骤4：视频生成")
    logger.info("=" * 70)

    all_shots = []
    for scene in episode.scenes:
        all_shots.extend(scene.shots)

    logger.info("总共需要生成 %d 个镜头视频", len(all_shots))

    if studio._mcp_server is None:
        dry_run = True
        logger.warning("MCP 服务器未配置，强制使用 DRY RUN 模式")

    if dry_run:
        logger.info("[DRY RUN] 以下为将执行的视频 API 调用（不实际调用）")
        for scene_idx, scene in enumerate(episode.scenes, 1):
            for shot_idx, shot in enumerate(scene.shots, 1):
                print(f"  [{scene_idx}.{shot_idx}] 生成视频")
                print(f"        Mode: {shot.generation_mode.value}")
                print(f"        Prompt: {shot.text_prompt[:60]}...")
                print(f"        Duration: {shot.duration}s")
                print()

        logger.info("DRY RUN 模式完成，未实际调用 API")
        return episode

    logger.warning("开始调用视频 API 生成视频，可能产生费用")
    episode = studio._generate_shots(episode)
    success_count = sum(
        1
        for scene in episode.scenes
        for shot in scene.shots
        if shot.generated_video_path
    )
    logger.info("视频生成完成: %d/%d", success_count, len(all_shots))
    return episode


def assemble_episode_video(studio: AITVStudio, episode):
    """将镜头视频组装成最终剧集视频。"""
    logger.info("=" * 70)
    logger.info("步骤5：视频组装")
    logger.info("=" * 70)

    logger.info("正在组装最终视频...")
    output_path = studio.video_assembler.assemble_episode(episode)
    logger.info("视频组装完成，输出: %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="AI-TV-Studio 配置驱动示例")
    parser.add_argument(
        "--config",
        default=DEFAULT_STORY_CONFIG,
        help=f"剧情配置文件路径（默认: {DEFAULT_STORY_CONFIG}）",
    )
    return parser.parse_args()


def main():
    """完整示例执行流程（配置驱动）。"""
    args = parse_args()
    config = load_story_config(args.config)

    print("=" * 70)
    print("AI-TV-Studio 配置驱动示例：生成短剧")
    print("=" * 70)
    print(f"配置文件: {args.config}")
    print()

    logger.info("=" * 70)
    logger.info("步骤0：初始化")
    logger.info("=" * 70)

    characters = create_characters(config)
    series_config = create_series_config(config)
    episode_outline = config["episode"]["outline"]
    episode_number = int(config["episode"].get("number", 1))

    studio = initialize_studio(config)
    for character in characters:
        studio.register_character(character)

    logger.info("初始化完成")

    logger.info("=" * 70)
    logger.info("步骤1：配置 API")
    logger.info("=" * 70)
    configure_studio_with_apis(studio, config)

    episode = generate_episode_script(
        studio=studio,
        series_config=series_config,
        episode_outline=episode_outline,
        episode_number=episode_number,
    )

    persist_dir = config.get("runtime", {}).get("persist_dir", "./data/generated")
    persist_episode_snapshot(episode, stage="script_generated", output_dir=persist_dir)

    episode = plan_shots(studio, episode)
    persist_episode_snapshot(episode, stage="shots_planned", output_dir=persist_dir)

    dry_run_mode = bool(config.get("runtime", {}).get("dry_run", True))
    episode = generate_videos(studio, episode, dry_run=dry_run_mode)
    persist_episode_snapshot(
        episode,
        stage="dry_run_completed" if dry_run_mode else "video_generated",
        output_dir=persist_dir,
    )

    logger.info("=" * 70)
    logger.info("示例执行完成")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
