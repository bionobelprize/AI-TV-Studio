"""
完整示例：一步一步生成小短剧
=====================================

此示例演示 AI-TV-Studio 的完整工作流程，包括：
1. 数据结构的构建（Episode、Scene、Shot）
2. 脚本生成 - LLM 调用（消耗token）
3. 视频生成 - 视频API调用（消耗token，价格较高 ⚠️）
4. 视频组装

注意标记：
- [数据结构构建] 标记：这里是内存中构建episode结构，不调用API
- [视频模型调用] 标记：这里消耗视频API token，需要支付费用
- [LLM调用] 标记：这里消耗LLM token
"""

import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# ============================================================================
# 第一部分：导入和基础配置
# ============================================================================

from src.studio import AITVStudio
from src.mcp.ark_video_client import ArkVideoAPIClient
from src.mcp.video_director_server import VideoDirectorServer
from src.models.character import Character, CharacterVisualCore, CharacterEmotion

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 第二部分：LLM客户端示例（这里使用OpenAI）
# ============================================================================

class LLMChatClient:
    """简单的LLM客户端包装，用于调用OpenAI或其他LLM服务。
    
    ⚠️ [LLM调用] 标记：每次调用 chat() 方法都会消耗LLM token
    """
    
    def __init__(self, model: str = "deepseek-chat"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")
        
        self.client = OpenAI()  # 需要设置 OPENAI_API_KEY 环境变量
        self.model = model
    
    def chat(self, model: str, messages: list) -> str:
        """调用LLM API生成响应。
        
        ⚠️ [LLM调用] 这里每次调用都消耗token
        
        Args:
            model: 模型名称
            messages: 对话消息列表
        
        Returns:
            LLM生成的响应文本
        """
        logger.info(f"📝 [LLM调用 ⚠️] 调用 {model} 生成脚本...")
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content


# ============================================================================
# 第三部分：数据结构构建 - 角色定义
# ============================================================================

def create_sample_characters() -> tuple[Character, Character]:
    """创建示例角色。
    
    [数据结构构建] 这里在内存中创建Character对象，不调用任何API
    """
    logger.info("🎭 [数据结构构建] 创建角色...")
    
    # 角色1：侦探小林
    character_lin = Character(
        id="char_lin_detective",
        name="林晓",
        age=32,
        gender="female",
        occupation="detective",
        personality_traits=["聪慧", "机敏", "坚决"],
        character_arc="一个在冷案上屡屡碰壁后，突然发现重要线索的侦探",
        visual_core=CharacterVisualCore(
            base_image_path="./data/characters/lin_base.jpg",
            reference_prompt="Asian female detective, 30s, sharp eyes, professional attire",
            key_features="short black hair, determined expression, detective coat"
        )
    )
    
    # 角色2：证人陈伟
    character_chen = Character(
        id="char_chen_witness",
        name="陈伟",
        age=28,
        gender="male",
        occupation="witness",
        personality_traits=["紧张", "犹豫", "隐瞒"],
        character_arc="一个知道关键线索但害怕透露真相的证人",
        visual_core=CharacterVisualCore(
            base_image_path="./data/characters/chen_base.jpg",
            reference_prompt="Asian male, 20s-30s, nervous, casual clothes",
            key_features="dark messy hair, anxious expression, blue shirt"
        )
    )
    
    logger.info(f"✅ 创建了 {character_lin.name}（{character_lin.occupation}）和 {character_chen.name}（{character_chen.occupation}）")
    
    return character_lin, character_chen


# ============================================================================
# 第四部分：Series配置
# ============================================================================

def create_series_config() -> Dict[str, Any]:
    """创建剧集系列的配置信息。
    
    [数据结构构建] 这是静态配置数据，不调用API
    """
    logger.info("📋 [数据结构构建] 创建Series配置...")
    
    config = {
        "title": "午夜档案",
        "genre": "悬疑犯罪",
        "description": "一个关于冷案复破的故事",
        "characters": [
            {
                "id": "char_lin_detective",
                "name": "林晓",
                "occupation": "detective"
            },
            {
                "id": "char_chen_witness",
                "name": "陈伟",
                "occupation": "witness"
            }
        ]
    }
    
    return config


# ============================================================================
# 第五部分：Studio初始化
# ============================================================================

def initialize_studio() -> AITVStudio:
    """初始化AI-TV-Studio。
    
    [数据结构构建] 创建Studio对象和相关管理器
    """
    logger.info("🎬 [数据结构构建] 初始化AI-TV-Studio...")
    
    studio = AITVStudio(config={
        "output_dir": "./outputs",
        "data_dir": "./data",
        "llm_model": "gpt-4"
    })
    
    return studio


# ============================================================================
# 第六部分：配置LLM和视频API
# ============================================================================

def configure_studio_with_apis(studio: AITVStudio, llm_client: LLMChatClient):
    """配置Studio使用LLM和视频API。
    
    ⚠️ [视频模型调用位置] 这里配置的是MCP服务器，后续调用时会消耗视频API token
    """
    logger.info("⚙️ [配置] 配置LLM客户端...")
    studio.configure_llm(llm_client, model="gpt-4")
    
    logger.info("⚙️ [配置] 配置Ark视频API客户端...")
    ark_client = ArkVideoAPIClient(
        config_path="config/video_api_keys.yaml",
        output_dir="./outputs"
    )
    
    video_director = VideoDirectorServer(api_client=ark_client)
    studio.configure_mcp(video_director)
    
    logger.info("✅ API配置完成")


# ============================================================================
# 第七部分：脚本生成 - LLM调用
# ============================================================================

def generate_episode_script(studio: AITVStudio, series_config: Dict[str, Any], episode_number: int = 1):
    """生成剧集脚本。
    
    ⚠️ [LLM调用] 这一步调用LLM生成脚本结构
    [数据结构构建] 返回的Episode对象在内存中构建
    
    Args:
        studio: 配置好的Studio实例
        series_config: Series配置
        episode_number: 剧集号
    
    Returns:
        生成的Episode对象
    """
    logger.info("=" * 70)
    logger.info("步骤2：脚本生成")
    logger.info("=" * 70)
    
    episode_outline = """
    侦探林晓正在调查一桩五年前的失踪案。她发现了新的线索，
    决定在一个下雨的夜晚前往旧工业区的废弃工厂。
    在那里，她意外遇到了失踪者的朋友陈伟，他似乎知道什么。
    两人在紧张对峙中，陈伟终于同意透露真相。
    """
    
    logger.info(f"📝 [LLM调用 ⚠️] 正在调用LLM生成剧集脚本...")
    logger.info(f"   剧集大纲：{episode_outline[:50]}...")
    
    # ⚠️ LLM调用发生在这里
    episode = studio.script_generator.generate_episode(
        series_config=series_config,
        episode_outline=episode_outline,
        episode_number=episode_number,
    )
    
    logger.info(f"✅ 脚本生成完成")
    logger.info(f"   - 剧集标题: {episode.episode_title}")
    logger.info(f"   - 话题: {episode.logline}")
    logger.info(f"   - 场景数: {len(episode.scenes)}")
    
    # 显示详细的数据结构
    print("\n" + "=" * 70)
    print("[数据结构构建] 生成的Episode结构：")
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


# ============================================================================
# 第八部分：镜头规划
# ============================================================================

def plan_shots(studio: AITVStudio, episode):
    """规划镜头和插入转场。
    
    [数据结构构建] 在内存中修改Episode结构，添加转场镜头等
    这一步不调用任何API
    """
    logger.info("=" * 70)
    logger.info("步骤3：镜头规划")
    logger.info("=" * 70)
    
    logger.info("🎥 [数据结构构建] 正在规划镜头和转场...")
    
    from src.algorithms.shot_planner import ShotPlanner
    
    planner = ShotPlanner(
        character_registry=studio.asset_manager.get_all_characters()
    )
    
    episode = planner.plan_episode(episode)
    
    total_shots = sum(len(scene.shots) for scene in episode.scenes)
    logger.info(f"✅ 镜头规划完成")
    logger.info(f"   - 总镜头数: {total_shots}")
    
    print("\n" + "=" * 70)
    print("[数据结构构建] 规划完成后的镜头信息：")
    print("=" * 70)
    
    for scene in episode.scenes:
        print(f"\nScene: {scene.location} ({scene.time_of_day})")
        for shot in scene.shots:
            print(f"  - Shot {shot.sequence_number}: {shot.action_description[:50]}...")
            print(f"    Mode: {shot.generation_mode.value} | Duration: {shot.duration}s")
            if shot.text_prompt:
                print(f"    Prompt: {shot.text_prompt[:60]}...")
    
    return episode


# ============================================================================
# 第九部分：视频生成 - 视频API调用
# ============================================================================

def generate_videos(studio: AITVStudio, episode, dry_run: bool = False):
    """为每个镜头生成视频。
    
    ⚠️ [视频模型调用 💰 高成本] 这一步为每个镜头调用视频API生成视频
    每次调用都会消耗API额度和费用
    
    Args:
        studio: 配置好的Studio实例
        episode: 规划好的Episode对象
        dry_run: 如果为True，只显示会发生什么，不实际调用API
    """
    logger.info("=" * 70)
    logger.info("步骤4：视频生成")
    logger.warning("⚠️ [视频模型调用 💰] 以下步骤将调用视频API，消耗token和费用")
    logger.info("=" * 70)
    
    all_shots = []
    for scene in episode.scenes:
        all_shots.extend(scene.shots)
    
    logger.info(f"总共需要生成 {len(all_shots)} 个镜头的视频")
    
    # 强制使用DRY RUN，如果MCP服务器未配置（模拟模式）
    if studio._mcp_server is None:
        dry_run = True
        logger.warning("⚠️ MCP服务器未配置，强制使用DRY RUN模式")
    
    if dry_run:
        logger.info("\n🔍 [DRY RUN] 显示将要发生的视频API调用（不实际调用）：\n")
        
        for scene_idx, scene in enumerate(episode.scenes, 1):
            for shot_idx, shot in enumerate(scene.shots, 1):
                print(f"  [{scene_idx}.{shot_idx}] 💰 [视频模型调用] 生成视频")
                print(f"        Mode: {shot.generation_mode.value}")
                print(f"        Prompt: {shot.text_prompt[:60]}...")
                print(f"        Duration: {shot.duration}s")
                print()
        
        logger.warning("⚠️ DRY RUN 模式: 未实际调用API")
        logger.info("   如需实际生成视频，请：")
        logger.info("   1. 配置 config/video_api_keys.yaml 中的 ark.api_key")
        logger.info("   2. 设置环境变量 OPENAI_API_KEY")
        logger.info("   3. 运行脚本时则会自动调用真实API")
        
        return episode
    
    else:
        logger.warning("💰 开始调用视频API生成视频...")
        logger.warning("每个镜头调用都会消耗token和费用")
        
        # ⚠️ 这是实际的视频API调用发生处
        from src.studio import AITVStudio as Studio
        episode = studio._generate_shots(episode)
        
        success_count = sum(
            1 for scene in episode.scenes 
            for shot in scene.shots 
            if shot.generated_video_path
        )
        
        logger.info(f"✅ 视频生成完成")
        logger.info(f"   - 成功生成: {success_count}/{len(all_shots)}")
        
        return episode


# ============================================================================
# 第十部分：视频组装
# ============================================================================

def assemble_episode_video(studio: AITVStudio, episode):
    """将所有镜头视频组装成最终剧集视频。
    
    [数据结构构建+处理] 这一步使用已生成的视频文件，进行组装
    不调用API，但I/O成本高
    """
    logger.info("=" * 70)
    logger.info("步骤5：视频组装")
    logger.info("=" * 70)
    
    logger.info("🎬 正在将所有镜头组装成最终视频...")
    
    output_path = studio.video_assembler.assemble_episode(episode)
    
    logger.info(f"✅ 视频组装完成")
    logger.info(f"   📹 最终视频: {output_path}")
    
    return output_path


# ============================================================================
# 第十一部分：主函数
# ============================================================================

def main():
    """完整的示例执行流程。"""
    
    print("=" * 70)
    print("AI-TV-Studio 完整示例：生成小短剧")
    print("=" * 70)
    print()
    
    # 步骤 0：数据结构构建 - 角色和配置
    logger.info("=" * 70)
    logger.info("步骤0：初始化")
    logger.info("=" * 70)
    
    character_lin, character_chen = create_sample_characters()
    series_config = create_series_config()
    
    # 步骤 1：初始化Studio
    studio = initialize_studio()
    
    studio.register_character(character_lin)
    studio.register_character(character_chen)
    
    logger.info("✅ 初始化完成\n")
    
    # 步骤 2：配置API - LLM和视频
    logger.info("=" * 70)
    logger.info("步骤1：配置API")
    logger.info("=" * 70)
    
    try:
        llm_client = LLMChatClient(model="gpt-4")
        configure_studio_with_apis(studio, llm_client)
    except (ImportError, Exception) as e:
        logger.error(f"❌ API配置失败: {type(e).__name__}: {e}")
        logger.info("\n提示：如需运行完整流程，请：")
        logger.info("  1. pip install openai")
        logger.info("  2. 设置环境变量 OPENAI_API_KEY")
        logger.info("  3. 配置 config/video_api_keys.yaml")
        logger.info("\n本示例将使用模拟模式继续演示流程\n")
        
        # 创建模拟客户端继续演示
        studio.configure_llm(DummyLLMClient(), model="gpt-4")
        
        # 模拟模式下跳过MCP配置，后续会在DRY RUN模式下演示
        studio._mcp_server = None
    
    # 步骤 3：生成脚本（LLM调用）
    episode = generate_episode_script(studio, series_config)
    
    # 步骤 4：规划镜头
    episode = plan_shots(studio, episode)
    
    # 步骤 5：生成视频（在DRY RUN模式）
    logger.info("\n💡 为了演示目的，这里使用DRY RUN模式（不实际调用视频API）\n")
    episode = generate_videos(studio, episode, dry_run=True)
    
    # 如果在生产环境，取消dry_run=True以实际生成视频：
    # episode = generate_videos(studio, episode, dry_run=False)
    
    logger.info("\n" + "=" * 70)
    logger.info("示例执行完成")
    logger.info("=" * 70)
    
    logger.info("\n💡 关键要点：")
    logger.info("  1. [数据结构构建] 标记处理内存中的Episode/Scene/Shot对象")
    logger.info("  2. [LLM调用] 标记处是脚本生成，消耗LLM token")
    logger.info("  3. [视频模型调用] 标记处是视频生成，消耗视频API token和费用")
    logger.info("\n建议验证策略：")
    logger.info("  - 先用小规模脚本测试数据结构 (步骤0-4)")
    logger.info("  - 用DRY RUN模式验证镜头规划和参数 (步骤5, dry_run=True)")
    logger.info("  - 单个镜头测试视频API (修改 generate_videos 函数)")
    logger.info("  - 最后运行完整流程 (dry_run=False)")


class DummyLLMClient:
    """用于演示的模拟LLM客户端。"""
    
    def chat(self, model: str, messages: list) -> str:
        """返回示例脚本JSON。"""
        return json.dumps({
            "episode_title": "午夜档案 - 第一集",
            "logline": "侦探林晓在废弃工厂发现了失踪案的关键线索",
            "scenes": [
                {
                    "scene_number": 1,
                    "location": "废弃工业区",
                    "time_of_day": "night",
                    "weather": "rain",
                    "mood": "tense",
                    "bgm_mood": "suspenseful",
                    "bgm_tempo": "moderate",
                    "ambient_sounds": ["rain", "distant wind"],
                    "shots": [
                        {
                            "sequence_number": 1,
                            "action_description": "警车停在工厂门口，林晓打开车门走出来",
                            "dialogue": None,
                            "characters_in_shot": ["char_lin_detective"],
                            "character_emotions": {"char_lin_detective": "neutral"},
                            "duration": 6,
                            "camera_motion": {"type": "static"},
                            "lighting_description": "雨夜，警车灯光照亮",
                            "text_prompt": "Night scene, abandoned industrial factory, rain, police car with lights, detective getting out of car, cinematic noir style"
                        }
                    ]
                }
            ]
        })


if __name__ == "__main__":
    main()
