"""
验证指南：AI-TV-Studio 成本分析和分步验证
================================================

本文档提供了如何逐步验证AI-TV-Studio的各个阶段，
特别是如何在调用高成本的视频API之前进行充分的验证。
"""

# ============================================================================
# 一、架构层次分析
# ============================================================================

"""
AI-TV-Studio 的执行阶段分为三类：

1️⃣ [数据结构构建] - 0成本
   └─ 在内存中创建和修改数据对象
   └─ 包括：角色定义、Series配置、Episode/Scene/Shot对象
   └─ 消耗：CPU + 内存
   └─ 时间：毫秒到秒级

2️⃣ [LLM调用] - 中等成本
   └─ 脚本生成
   └─ 消耗：OpenAI/DeepSeek等LLM API token
   └─ 时间：5-30秒
   └─ 成本参考：
      - GPT-4: ~$0.03 per 1K input tokens, $0.06 per 1K output tokens
      - 平均脚本生成：2000-5000 tokens
      - 估计成本：$0.06-$0.30 per episode

3️⃣ [视频模型调用] - 高成本 💰
   └─ 视频生成
   └─ 消耗：Ark/Runway/其他视频API额度
   └─ 时间：30秒-2分钟 per shot
   └─ 成本参考：
      - Ark: ~100-500 积分 per video (取决于时长和模式)
      - Runway: ~$0.05-$0.15 per second
      - 一个5-10镜头短剧：500-5000 积分 或 $2.50-$15
      - 一个20-30镜头标准剧集：2000-15000 积分 或 $10-$50

4️⃣ [视频组装] - 低成本
   └─ FFmpeg 处理
   └─ 消耗：CPU + 磁盘I/O
   └─ 时间：几秒到十几秒


# ============================================================================
# 二、分步验证流程
# ============================================================================

STAGE 0: 初始化 [数据结构构建]
────────────────────────────────────────────────────────────────────────────
✅ 验证步骤：
   
   python -c "
from src.studio import AITVStudio
from src.models.character import Character, CharacterVisualCore

# 创建角色（内存操作）
char = Character(
    id='test_char',
    name='Test',
    role='Test Role',
    description='Test'
)
print('✓ Character created')

# 创建Studio
studio = AITVStudio(config={'output_dir': './test_output'})
print('✓ Studio initialized')

studio.register_character(char)
print('✓ Character registered')
   "

⏱️ 预期耗时：< 1 秒
💰 成本：0
✅ 检查点：Studio对象创建成功，无错误


STAGE 1: 配置API [配置]
────────────────────────────────────────────────────────────────────────────
✅ 验证步骤：

   # 检查 API Key 是否正确配置
   python -c "
from src.utils.api_key_config import load_video_api_keys

config = load_video_api_keys('config/video_api_keys.yaml')
print('Video API config:', config)

if 'ark' in config and config['ark'].get('api_key'):
    print('✓ Ark API key configured')
else:
    print('⚠️  No Ark API key found')
   "

⏱️ 预期耗时：< 1 秒
💰 成本：0（仅读取配置）
✅ 检查点：API key已正确配置


STAGE 2: 脚本生成 [LLM调用] ⚠️
────────────────────────────────────────────────────────────────────────────
✅ 验证步骤：

   python example_script_verification.py

   此脚本会：
   - 准备小规模的Series配置
   - 调用LLM生成脚本
   - 打印生成的Episode结构
   - 执行时间：5-30秒
   - 消耗：小量LLM token（$0.01-$0.05）

⏱️ 预期耗时：10-30 秒
💰 成本：约 $0.01-$0.05 per episode
✅ 检查点：
   - Episode 对象成功创建
   - Scene 数量合理（2-5个）
   - Shot 数量合理（5-20个）
   - 每个 Shot 的 text_prompt 非空
   - generation_mode 正确设置


STAGE 3: 镜头规划 [数据结构构建]
────────────────────────────────────────────────────────────────────────────
✅ 验证步骤：

   python -c "
from src.algorithms.shot_planner import ShotPlanner
from src.models.episode import Episode

# 以Stage 2生成的Episode为输入
episode = ...  # 从上一步获取

planner = ShotPlanner(character_registry={})
planned_episode = planner.plan_episode(episode)

total_shots = sum(len(s.shots) for s in planned_episode.scenes)
print(f'Total shots after planning: {total_shots}')

# 检查每个Shot是否有text_prompt
for scene in planned_episode.scenes:
    for shot in scene.shots:
        assert shot.text_prompt, f'Shot {shot.id} has no prompt'
        assert shot.generation_mode, f'Shot {shot.id} has no mode'
print('✓ All shots have prompts and generation modes')
   "

⏱️ 预期耗时：< 1 秒
💰 成本：0
✅ 检查点：
   - Shot 总数无意外增加/减少
   - 转场镜头正确插入
   - 每个 Shot 都有清晰的 text_prompt


STAGE 4A: 视频生成 - DRY RUN [数据分析]
────────────────────────────────────────────────────────────────────────────
✅ 验证步骤（❌ 不消耗API额度）：

   python -c "
from src.studio import AITVStudio
from src.mcp.ark_video_client import ArkVideoAPIClient
# 假设已从Stage 2和3获得已规划的Episode对象

episode = ...

# 分析将要生成的视频
all_shots = []
for scene in episode.scenes:
    all_shots.extend(scene.shots)

total_duration = sum(shot.duration for shot in all_shots)
print(f'Total video duration: {total_duration}s')

# 按生成模式分类
modes = {}
for shot in all_shots:
    mode = shot.generation_mode.value
    modes[mode] = modes.get(mode, 0) + 1

print('Shots by generation mode:')
for mode, count in modes.items():
    print(f'  {mode}: {count}')

# 估算成本（基于Ark的粗略估计）
estimated_credits = len(all_shots) * 200  # 平均每个shot 200积分
print(f'Estimated Ark credits: ~{estimated_credits} (rough estimate)')
   "

⏱️ 预期耗时：< 1 秒
💰 成本：0
✅ 检查点：
   - Shot总数符合预期
   - 生成模式分布合理
   - 总时长< 180s（一般建议）


STAGE 4B: 视频生成 - 单个Shot测试 [视频模型调用] 💰
────────────────────────────────────────────────────────────────────────────
✅ 验证步骤（⚠️ 消耗少量API额度）：

   python example_single_shot_generation.py

   此脚本会：
   - 生成一个单独的Shot
   - 调用视频API生成单个视频
   - 保存到 outputs/test_shot_*.mp4
   - 验证视频文件完整性

⏱️ 预期耗时：30秒-2分钟 per shot
💰 成本：约 200-500 积分（Ark）或 $0.05-$0.15（Runway）
✅ 检查点：
   - 视频文件生成成功
   - 视频长度符合 duration 参数
   - 视频质量可接受


STAGE 4C: 视频生成 - 完整流程 [视频模型调用] 💰
────────────────────────────────────────────────────────────────────────────
✅ 验证步骤（⚠️ 消耗大量API额度 - 准备好支付费用）：

   # 选项1：在已有Episode和规划的基础上执行
   python -c "
from src.studio import AITVStudio
studio = ...
episode = ...
episode = studio._generate_shots(episode)
   "

   # 选项2：使用完整的生产脚本
   python example_short_play_generation.py  # 修改注释取消 dry_run=False

⏱️ 预期耗时：2-10 分钟（取决于shot数量和持续时间）
💰 成本：约 $2.50-$50 or 500-15000 积分
✅ 检查点：
   - 每个Shot都生成了视频文件
   - 视频文件路径保存在 shot.generated_video_path 中
   - 所有视频文件在 outputs/ 目录中


STAGE 5: 视频组装 [视频处理]
────────────────────────────────────────────────────────────────────────────
✅ 验证步骤：

   python -c "
from src.studio import AITVStudio
studio = AITVStudio()
episode = ...  # 从Stage 4获取（已生成视频）
output_path = studio.video_assembler.assemble_episode(episode)
print(f'Final video: {output_path}')
   "

⏱️ 预期耗时：10-60 秒
💰 成本：0（本地处理）
✅ 检查点：
   - outputs/episode_*.mp4 文件存在
   - 文件大小> 10MB（通常）
   - 可以在视频播放器中正常播放


# ============================================================================
# 三、成本节省建议
# ============================================================================

1. 开发阶段
   ├─ 使用 DRY RUN 模式验证逻辑（第4步）
   ├─ 单个shot测试而不是完整流程
   └─ 使用较短的 duration（5-6s 而不是 12s）

2. 测试阶段
   ├─ 先生成 1-2 个 shots 验证质量
   ├─ 如满意，再运行完整流程
   └─ 记录平均成本/shot

3. 生产阶段
   ├─ 批量生成多个 episodes
   ├─ 考虑购买 API 套餐（volume discounts）
   └─ 监控成本，设置告警

4. 参数优化
   ├─ 减少 shot 数量（合并相似场景）
   ├─ 减少视频 duration（5-8s vs 10-12s）
   ├─ 优先使用 txt2video 模式（通常最便宜）
   └─ 避免过度使用 ref2video（图像处理较贵）


# ============================================================================
# 四、快速起步命令
# ============================================================================

# 1. 验证基础设置
python example_short_play_generation.py

# 2. 检查API配置
python -c "from src.utils.api_key_config import load_video_api_keys; print(load_video_api_keys())"

# 3. 运行单个shot测试（如果存在test脚本）
python example_single_shot_generation.py

# 4. 完整流程（修改dry_run参数）
# 编辑 example_short_play_generation.py，改 dry_run=True 为 dry_run=False


# ============================================================================
# 五、调试技巧
# ============================================================================

1. 启用详细日志
   import logging
   logging.basicConfig(level=logging.DEBUG)

2. 保存中间状态
   import json
   with open('episode.json', 'w') as f:
       json.dump(episode.__dict__, f, indent=2)

3. 检查单个Shot的生成参数
   shot = episode.scenes[0].shots[0]
   print(f"Text Prompt: {shot.text_prompt}")
   print(f"Generation Mode: {shot.generation_mode}")
   print(f"Duration: {shot.duration}")
   print(f"Characters: {shot.characters_in_shot}")

4. 模拟视频生成（用于UI测试）
   # 创建虚拟视频文件进行测试
   touch outputs/test_video.mp4
   shot.generated_video_path = "outputs/test_video.mp4"


# ============================================================================
# 六、常见问题
# ============================================================================

Q1: 如何知道具体会消耗多少成本？
A:  1. 统计 shots 数量和 duration
    2. 查看具体 generation_mode
    3. 查询 API 提供商的价目表
    4. 运行估算脚本

Q2: 能否先完成整个流程而延迟视频生成？
A:  可以！流程分离：
    - 步骤0-3：完全在内存中，准备好 Episode 结构
    - 步骤4：可选且独立，需要时执行
    - 步骤5：需要步骤4的结果

Q3: 如何限制成本？
A:  ├─ 减少 shots 数量
    ├─ 缩短 duration
    ├─ 选择便宜的生成模式
    ├─ 设置 API 配额限制
    └─ 定期审计成本

Q4: 可以暂停和恢复吗？
A:  是的，保存 Episode 对象：
    - 保存为 JSON
    - 之后加载继续生成
    - 已生成的 shots 跳过
"""

if __name__ == "__main__":
    print(__doc__)
