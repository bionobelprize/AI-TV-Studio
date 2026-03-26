"""
成本分析工具
============

用于分析和预估AI-TV-Studio的API消耗成本，包括LLM和视频API。
在实际生成前进行成本估算，避免惊人的账单。
"""

import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


# ============================================================================
# 成本数据
# ============================================================================

@dataclass
class APIpricing:
    """API定价配置。"""
    
    # LLM定价（OpenAI GPT-4）
    GPT4_INPUT_TOKENS_PER_1K = 0.03  # $
    GPT4_OUTPUT_TOKENS_PER_1K = 0.06  # $
    
    # 视频API定价（Ark - 基于积分系统）
    # 积分转人民币汇率（需根据实际查询调整）
    ARK_CREDIT_RATE = 0.01  # 1积分 = 0.01元
    
    # 生成模式的积分消耗（每秒）
    ARK_CREDITS_PER_SECOND = {
        "txt2video": 35,          # 最便宜
        "first_frame": 40,        # 需要第一帧
        "firstlast_frame": 45,    # 需要首尾帧
        "ref2video": 60,          # 最贵（参考图像）
    }
    
    # Runway视频API定价
    RUNWAY_COST_PER_SECOND = 0.015  # $


# ============================================================================
# 成本计算函数
# ============================================================================

def estimate_llm_cost(episodes: int, avg_tokens_per_script: int = 3500) -> Tuple[int, float]:
    """估算LLM脚本生成成本。
    
    Args:
        episodes: 剧集数量
        avg_tokens_per_script: 平均每个脚本的token数
    
    Returns:
        (总tokens, 估计成本$)
    """
    total_input_tokens = episodes * avg_tokens_per_script
    total_output_tokens = episodes * (avg_tokens_per_script // 2)  # 输出约为输入一半
    
    cost = (
        (total_input_tokens / 1000) * APIpricing.GPT4_INPUT_TOKENS_PER_1K +
        (total_output_tokens / 1000) * APIpricing.GPT4_OUTPUT_TOKENS_PER_1K
    )
    
    return total_input_tokens + total_output_tokens, cost


def estimate_video_api_cost(
    total_shots: int,
    avg_duration_per_shot: int = 8,
    generation_mode_distribution: Dict[str, float] = None,
    api_provider: str = "ark"
) -> Tuple[int, float]:
    """估算视频API生成成本。
    
    Args:
        total_shots: 总镜头数
        avg_duration_per_shot: 平均每个镜头的时长（秒）
        generation_mode_distribution: 各生成模式的比例分布
        api_provider: API提供商 ("ark" 或 "runway")
    
    Returns:
        (总单位数, 估计成本)
    """
    if generation_mode_distribution is None:
        # 默认分布：70% txt2video, 20% first_frame, 10% firstlast_frame
        generation_mode_distribution = {
            "txt2video": 0.7,
            "first_frame": 0.2,
            "firstlast_frame": 0.1,
            "ref2video": 0.0,
        }
    
    if api_provider == "ark":
        total_cost = 0
        total_units = 0
        
        for mode, ratio in generation_mode_distribution.items():
            shots_in_mode = int(total_shots * ratio)
            units_per_shot = APIpricing.ARK_CREDITS_PER_SECOND[mode] * avg_duration_per_shot
            mode_cost = shots_in_mode * units_per_shot * APIpricing.ARK_CREDIT_RATE
            
            total_cost += mode_cost
            total_units += shots_in_mode * units_per_shot
        
        return int(total_units), total_cost
    
    elif api_provider == "runway":
        total_seconds = total_shots * avg_duration_per_shot
        cost = total_seconds * APIPrice.RUNWAY_COST_PER_SECOND
        return total_seconds, cost
    
    else:
        raise ValueError(f"Unknown API provider: {api_provider}")


def calculate_total_cost(
    episodes: int,
    shots_per_episode: int = 20,
    avg_shot_duration: int = 8,
    generation_mode_distribution: Dict[str, float] = None,
    enable_llm: bool = True
) -> Dict[str, Any]:
    """计算完整生成流程的总成本。
    
    Args:
        episodes: 剧集数量
        shots_per_episode: 每个剧集的平均镜头数
        avg_shot_duration: 平均镜头时长（秒）
        generation_mode_distribution: 生成模式分布
        enable_llm: 是否包括LLM成本
    
    Returns:
        成本分析字典
    """
    total_shots = episodes * shots_per_episode
    
    result = {
        "episodes": episodes,
        "shots_per_episode": shots_per_episode,
        "total_shots": total_shots,
        "avg_shot_duration": avg_shot_duration,
        "total_duration_seconds": total_shots * avg_shot_duration,
        "total_duration_minutes": total_shots * avg_shot_duration / 60,
    }
    
    # LLM成本
    if enable_llm:
        tokens, llm_cost = estimate_llm_cost(episodes)
        result["llm_tokens"] = tokens
        result["llm_cost_usd"] = round(llm_cost, 2)
    else:
        result["llm_tokens"] = 0
        result["llm_cost_usd"] = 0
    
    # 视频API成本（Ark）
    units_ark, cost_ark = estimate_video_api_cost(
        total_shots,
        avg_shot_duration,
        generation_mode_distribution,
        api_provider="ark"
    )
    result["video_api_ark_credits"] = int(units_ark)
    result["video_api_ark_cost_rmb"] = round(cost_ark, 2)
    result["video_api_ark_cost_usd"] = round(cost_ark / 6.5, 2)  # 大约汇率
    
    # 总成本
    result["total_cost_usd"] = round(
        result["llm_cost_usd"] + result["video_api_ark_cost_usd"],
        2
    )
    result["total_cost_rmb"] = round(
        result["llm_cost_usd"] * 6.5 + result["video_api_ark_cost_rmb"],
        2
    )
    
    return result


# ============================================================================
# 打印和格式化
# ============================================================================

def print_cost_analysis(analysis: Dict[str, Any]):
    """格式化打印成本分析。"""
    
    print("\n" + "=" * 70)
    print("💰 AI-TV-Studio 成本分析")
    print("=" * 70)
    
    print(f"\n📊 生产规模：")
    print(f"  • 剧集数: {analysis['episodes']}")
    print(f"  • 每集镜头数: {analysis['shots_per_episode']}")
    print(f"  • 总镜头数: {analysis['total_shots']}")
    print(f"  • 采样镜头时长: {analysis['avg_shot_duration']}秒")
    print(f"  • 总时长: {analysis['total_duration_minutes']:.1f}分钟 ({analysis['total_duration_seconds']}秒)")
    
    print(f"\n💰 LLM成本 (GPT-4脚本生成)：")
    if analysis['llm_cost_usd'] > 0:
        print(f"  • Tokens总计: {analysis['llm_tokens']:,}")
        print(f"  • 成本: ${analysis['llm_cost_usd']:.2f} (≈¥{analysis['llm_cost_usd']*6.5:.2f})")
    else:
        print(f"  • 已禁用")
    
    print(f"\n🎥 视频API成本 (Ark)：")
    print(f"  • 积分总计: {analysis['video_api_ark_credits']:,}")
    print(f"  • 成本: ¥{analysis['video_api_ark_cost_rmb']:.2f} (≈${analysis['video_api_ark_cost_usd']:.2f})")
    print(f"  • 每个镜头: ¥{analysis['video_api_ark_cost_rmb']/analysis['total_shots']:.2f}")
    print(f"  • 每分钟: ¥{analysis['video_api_ark_cost_rmb']/analysis['total_duration_minutes']:.2f}")
    
    print(f"\n💳 总成本：")
    print(f"  • 人民币: ¥{analysis['total_cost_rmb']:.2f}")
    print(f"  • 美元: ${analysis['total_cost_usd']:.2f}")
    print(f"  • 每集: ¥{analysis['total_cost_rmb']/analysis['episodes']:.2f}")
    
    print("\n" + "=" * 70)


def print_cost_scenarios():
    """打印常见场景的成本对比。"""
    
    print("\n" + "=" * 70)
    print("📋 常见场景成本估算")
    print("=" * 70)
    
    scenarios = [
        {
            "name": "小测试 (1分钟短视频)",
            "episodes": 1,
            "shots_per_episode": 10,
            "avg_duration": 6,
        },
        {
            "name": "短剧试验 (3分钟)",
            "episodes": 1,
            "shots_per_episode": 20,
            "avg_duration": 8,
        },
        {
            "name": "完整短剧 (10分钟)",
            "episodes": 1,
            "shots_per_episode": 40,
            "avg_duration": 8,
        },
        {
            "name": "迷你系列 (3集，各5分钟)",
            "episodes": 3,
            "shots_per_episode": 25,
            "avg_duration": 8,
        },
        {
            "name": "标准剧集 (1集，20分钟)",
            "episodes": 1,
            "shots_per_episode": 60,
            "avg_duration": 8,
        },
        {
            "name": "完整季度 (10集，各20分钟)",
            "episodes": 10,
            "shots_per_episode": 60,
            "avg_duration": 8,
        },
    ]
    
    for scenario in scenarios:
        analysis = calculate_total_cost(
            episodes=scenario["episodes"],
            shots_per_episode=scenario["shots_per_episode"],
            avg_shot_duration=scenario["avg_duration"],
        )
        
        print(f"\n{scenario['name']}:")
        print(f"  └─ 视频: {analysis['total_shots']} shots × {scenario['avg_duration']}s = {analysis['total_duration_minutes']:.1f}min")
        print(f"  └─ 成本: ¥{analysis['total_cost_rmb']:.2f} (${analysis['total_cost_usd']:.2f})")
        print(f"     • LLM: ¥{analysis['llm_cost_usd']*6.5:.2f}")
        print(f"     • 视频: ¥{analysis['video_api_ark_cost_rmb']:.2f}")


def print_cost_optimization_tips():
    """打印成本优化建议。"""
    
    print("\n" + "=" * 70)
    print("💡 成本优化建议")
    print("=" * 70)
    
    tips = """
1. 减少镜头数量
   • 合并相似的场景 → -25% 成本
   • 使用长镜头代替多个短镜头 → -30% 成本

2. 缩短视频时长
   • 从8秒改为6秒 → -25% 成本
   • 从8秒改为5秒 → -38% 成本

3. 优化生成模式
   • 使用 txt2video 替代 ref2video → -60% 成本
   • 使用 first_frame 替代 ref2video → -40% 成本

4. 批量生成策略
   • 单集成本: ¥50-200
   • 批量10集: 均价可降20-30%

5. 分阶段生成
   • 第1阶段: 脚本验证 (仅LLM)
   • 第2阶段: 单镜头测试 (1-2个视频)
   • 第3阶段: 完整生成 (全部视频)

6. 复用内容
   • 相同背景复用镜头 → -50% 成本
   • 多个剧集共用过渡镜头 → -20% 成本
"""
    print(tips)


# ============================================================================
# 交互式工具
# ============================================================================

def interactive_cost_calculator():
    """交互式成本计算工具。"""
    
    print("\n" + "=" * 70)
    print("🧮 交互式成本计算器")
    print("=" * 70)
    
    try:
        episodes = int(input("\n请输入剧集数 (默认1): ") or "1")
        shots_per_episode = int(input("请输入每集镜头数 (默认20): ") or "20")
        avg_duration = int(input("请输入平均镜头时长秒数 (默认8): ") or "8")
        
        analysis = calculate_total_cost(
            episodes=episodes,
            shots_per_episode=shots_per_episode,
            avg_shot_duration=avg_duration,
        )
        
        print_cost_analysis(analysis)
        
    except ValueError:
        print("❌ 输入无效")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序。"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║           AI-TV-Studio 成本分析工具                              ║
    ║                                                                  ║
    ║  在实际生成前估算API成本，避免惊人的账单！                        ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # 打印场景对比
    print_cost_scenarios()
    
    # 打印优化建议
    print_cost_optimization_tips()
    
    # 交互式计算
    response = input("\n是否进行自定义成本计算? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        interactive_cost_calculator()
    
    print("\n" + "=" * 70)
    print("✅ 成本分析完成")
    print("=" * 70)
    print("""
下一步建议：
  1. 根据成本选择合适的生产规模
  2. 使用 example_script_verification.py 验证脚本生成
  3. 使用 example_single_shot_test.py 测试单个镜头
  4. 最后使用 example_short_play_generation.py 完整生成
    """)


if __name__ == "__main__":
    main()
