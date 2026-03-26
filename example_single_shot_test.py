"""
单个Shot视频生成测试
======================

这个脚本用于测试单个镜头的视频生成，是理想的成本控制验证工具。
只生成1个Shot，便于：
- 测试视频API配置
- 验证生成质量
- 计算单个shot的成本
- 调试参数设置

使用此脚本而不是完整流程可以节省 95% 以上的成本！
"""

import logging
from pathlib import Path
from src.mcp.ark_video_client import ArkVideoAPIClient
from src.mcp.video_director_server import VideoDirectorServer
from src.models.shot import Shot, GenerationMode, CameraMotion
from src.models.scene import Scene
from src.models.episode import Episode

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def test_single_shot_generation():
    """生成单个Shot，用于API测试和成本验证。"""
    
    logger.info("=" * 70)
    logger.info("单个Shot视频生成测试")
    logger.info("=" * 70)
    logger.warning("⚠️ 这个测试将调用视频API，消耗API额度")
    
    # ========================================================================
    # [数据结构构建] 创建单个shot
    # ========================================================================
    logger.info("\n[数据结构构建] 创建测试Shot...")
    
    shot = Shot(
        id="test_shot_001",
        scene_id="test_scene_001",
        sequence_number=1,
        action_description="一个穿着黑色衣服的女侦探在下雨的夜晚走向一个废弃的工厂",
        dialogue=None,
        characters_in_shot=["char_lin"],
        character_emotions={},
        generation_mode=GenerationMode.TEXT_TO_VIDEO,  # 最便宜的模式
        duration=6,  # 短时长节省成本
        camera_motion=CameraMotion(type="static"),
        lighting_description="noir style, rainy night, street lamp",
        text_prompt=(
            "cinematic scene, noir style, rainy night, "
            "asian female detective in black coat walking toward abandoned factory, "
            "moody atmospheric lighting, professional film quality, "
            "8mm resolution, 6 seconds"
        )
    )
    
    logger.info(f"✓ Shot created: {shot.id}")
    logger.info(f"  - Duration: {shot.duration}s")
    logger.info(f"  - Mode: {shot.generation_mode.value}")
    logger.info(f"  - Prompt: {shot.text_prompt[:60]}...")
    
    # ========================================================================
    # [视频模型调用] 初始化视频API
    # ========================================================================
    logger.info("\n[视频模型调用] 初始化Ark视频API...")
    
    try:
        ark_client = ArkVideoAPIClient(
            config_path="config/video_api_keys.yaml",
            output_dir="./outputs"
        )
        logger.info("✓ Ark client initialized")
    except ValueError as e:
        logger.error(f"❌ 无法初始化Ark客户端: {e}")
        logger.info("\n需要配置API Key:")
        logger.info("  1. 创建或编辑 config/video_api_keys.yaml")
        logger.info("  2. 添加你的Ark API Key:")
        logger.info("""
ark:
  api_key: "your-api-key-here"
  base_url: "https://ark.cn-beijing.volces.com/api/v3"
  output_dir: "./outputs"
        """)
        return None
    
    # ========================================================================
    # [视频模型调用] 生成视频 💰
    # ========================================================================
    logger.warning("\n💰 [视频模型调用] 开始调用视频API生成视频...")
    logger.warning("   这将消耗API额度和费用")
    
    try:
        result = ark_client.generate_video(
            mode=shot.generation_mode.value,
            prompt=shot.text_prompt,
            duration=shot.duration,
            resolution="1280x720"
        )
        
        video_path = result.get("path")
        task_id = result.get("task_id")
        
        logger.info(f"✓ 视频生成请求成功")
        logger.info(f"  - Task ID: {task_id}")
        logger.info(f"  - Output Path: {video_path}")
        
        shot.generated_video_path = video_path
        
        # 验证文件
        if Path(video_path).exists():
            file_size = Path(video_path).stat().st_size
            logger.info(f"✓ 视频文件已保存")
            logger.info(f"  - File size: {file_size / 1024 / 1024:.2f}MB")
            return shot
        else:
            logger.warning(f"⚠️ 视频文件不存在: {video_path}")
            return shot
    
    except Exception as e:
        logger.error(f"❌ 视频生成失败: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.error(f"   Full error: {e}")
        
        logger.info("\n故障排除：")
        logger.info("  1. 检查API Key是否正确")
        logger.info("  2. 检查网络连接")
        logger.info("  3. 检查API服务状态")
        logger.info("  4. 查看完整错误信息")
        
        return None


def test_different_modes():
    """测试不同的生成模式。"""
    
    logger.info("=" * 70)
    logger.info("测试不同的生成模式（成本对比）")
    logger.info("=" * 70)
    
    logger.info("\n生成模式对比（成本从低到高）：")
    logger.info("  1. txt2video (TEXT_TO_VIDEO)")
    logger.info("     └─ 仅需要文本提示")
    logger.info("     └─ 成本: ⭐️ 最便宜")
    logger.info("")
    logger.info("  2. first_frame (FIRST_FRAME)")
    logger.info("     └─ 需要第一帧图像 + 文本")
    logger.info("     └─ 成本: ⭐️⭐️ 中等")
    logger.info("")
    logger.info("  3. firstlast_frame (FIRSTLAST_FRAME)")
    logger.info("     └─ 需要首尾两帧 + 文本")
    logger.info("     └─ 成本: ⭐️⭐️⭐️ 较高")
    logger.info("")
    logger.info("  4. ref2video (REFERENCE_TO_VIDEO)")
    logger.info("     └─ 需要多个参考图像 + 文本")
    logger.info("     └─ 成本: ⭐️⭐️⭐️⭐️ 最贵")
    
    logger.info("\n推荐使用顺序（从经济角度）：")
    logger.info("  1. 开发阶段：txt2video")
    logger.info("  2. 质量不满足：升级到 first_frame")
    logger.info("  3. 需要动作连贯：使用 firstlast_frame")
    logger.info("  4. 需要角色一致性：使用 ref2video")


def print_cost_summary():
    """打印成本对比。"""
    
    logger.info("\n" + "=" * 70)
    logger.info("成本参考（基于Ark API）")
    logger.info("=" * 70)
    
    print("""
假设使用Ark视频生成API：

┌─ 短剧成本估算 ─────────────────────────────────────────────┐
│                                                               │
│ 3分钟短剧 (单场景，20个shot)：                               │
│ ├─ 纯txt2video: 20 × 200积分 = 4000积分 (~¥40)               │
│ ├─ 混合模式: 20 × 350积分 = 7000积分 (~¥70)                  │
│ └─ 参考图像模式: 20 × 500积分 = 10000积分 (~¥100)            │
│                                                               │
│ 标准剧集 (30分钟，5场景，80个shot)：                         │
│ ├─ 纯txt2video: 80 × 200积分 = 16000积分 (~¥160)             │
│ ├─ 混合模式: 80 × 350积分 = 28000积分 (~¥280)                │
│ └─ 参考图像模式: 80 × 500积分 = 40000积分 (~¥400)            │
│                                                               │
└───────────────────────────────────────────────────────────────┘

建议成本控制方案：
├─ 小预算 (< ¥100): 1个短剧，txt2video模式
├─ 中预算 (¥100-500): 3-5个短剧，混合模式
└─ 大预算 (> ¥500): 完整系列，所有模式

节省成本的技巧：
├─ 减少shot数量（合并相似镜头）
├─ 使用较短duration (5-6s vs 10-12s) → 节省20-30%
├─ 优先使用txt2video模式 → 节省60-75%
├─ 批量生成 → 获得体量折扣
└─ 复用生成的视频 → 减少重复生成
""")


def main():
    """主程序。"""
    
    print("\n" + "=" * 70)
    print("AI-TV-Studio 单个Shot生成测试")
    print("=" * 70)
    
    # 显示成本信息
    test_different_modes()
    print_cost_summary()
    
    # 询问用户是否继续
    response = input("\n是否继续测试视频生成？(yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        logger.info("已取消测试")
        return
    
    # 执行单个shot测试
    shot = test_single_shot_generation()
    
    if shot and shot.generated_video_path:
        logger.info("\n" + "=" * 70)
        logger.info("✅ 测试成功完成")
        logger.info("=" * 70)
        logger.info(f"生成的视频: {shot.generated_video_path}")
        logger.info("\n下一步：")
        logger.info("  1. 检查视频质量和内容")
        logger.info("  2. 如果满意，进行完整剧集生成")
        logger.info("  3. 如需调整，修改text_prompt或参数后重试")
        
        logger.info("\n成本记录：")
        logger.info(f"  - 单个Shot成本: ~200-500 积分 (¥2-5)")
        logger.info(f"  - 完整短剧(20 shots): ~4000-10000 积分 (¥40-100)")
        logger.info(f"  - 完整剧集(80 shots): ~16000-40000 积分 (¥160-400)")
    else:
        logger.warning("\n测试未成功完成")
        logger.info("请检查API配置和错误信息")


if __name__ == "__main__":
    main()
