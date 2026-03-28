# 剧情配置文档格式说明

本文档说明 `example_short_play_generation.py` 所使用的剧情配置文件格式。

## 1. 文件类型

- 推荐使用 YAML，编码 UTF-8。
- 示例文件：`config/story_example.yaml`

## 2. 顶层结构

配置文件包含以下顶层字段：

- `studio`：运行与模型配置
- `series`：系列信息
- `episode`：单集信息（包含大纲）
- `characters`：角色详细定义列表
- `runtime`：运行时行为（可选）
- `api`：外部 API 相关配置（可选）

其中必填字段为：

- `studio`
- `series`
- `episode`
- `characters`

程序还会校验：

- `episode.outline` 必须非空
- `characters` 必须是非空列表

## 3. 字段说明

### 3.1 `studio`

- `output_dir` (string, 可选)：输出目录，默认 `./outputs`
- `data_dir` (string, 可选)：数据目录，默认 `./data`
- `llm_model` (string, 可选)：LLM 模型名，默认 `deepseek-chat`

### 3.2 `series`

- `title` (string, 可选)：系列标题
- `genre` (string, 可选)：类型，如 `drama`、`crime`
- `description` (string, 可选)：系列简介
- `characters` (list, 可选)：给脚本生成器的简版角色清单

`series.characters` 中每个元素建议包含：

- `id` (string)：角色 ID
- `name` (string)：角色名称
- `occupation` (string, 可选)：角色职业

如果不提供 `series.characters`，程序会根据顶层 `characters` 自动生成简版清单。

### 3.3 `episode`

- `number` (int, 可选)：集数，默认 `1`
- `outline` (string, 必填)：剧情大纲，会直接送入 LLM 进行脚本生成

建议 `outline` 使用 YAML 多行字符串（`|`）书写。

### 3.4 `characters`

`characters` 是角色详细定义列表，每个角色元素建议包含：

- `id` (string, 必填)：角色唯一 ID
- `name` (string, 必填)：角色名
- `age` (int, 可选)
- `gender` (string, 可选)
- `occupation` (string, 可选)
- `aliases` (list[string], 可选)
- `personality_traits` (list[string], 可选)
- `character_arc` (string, 可选)
- `voice_id` (string/null, 可选)
- `voice_emotion_profile` (dict, 可选)
- `visual_core` (object, 必填)

`visual_core` 字段：

- `base_image_path` (string, 必填)：基础参考图路径
- `reference_prompt` (string, 可选)
- `key_features` (string, 可选)
- `lora_trigger` (string/null, 可选)
- `front_view` (string/null, 可选)
- `side_view` (string/null, 可选)
- `three_quarter_view` (string/null, 可选)

## 4. 可选运行参数

### 4.1 `runtime`

- `dry_run` (bool, 可选)：是否仅演示视频生成调用，默认 `true`
- `persist_dir` (string, 可选)：中间快照输出目录，默认 `./data/generated`

### 4.2 `api`

- `video_api_config_path` (string, 可选)：视频 API Key 配置路径，默认 `config/video_api_keys.yaml`

## 5. 最小可用示例

```yaml
studio:
  output_dir: "./outputs"
  data_dir: "./data"
  llm_model: "deepseek-chat"

series:
  title: "通用系列"
  genre: "drama"

episode:
  number: 1
  outline: |
    这里填写通用剧情大纲。

characters:
  - id: "char_1"
    name: "角色1"
    visual_core:
      base_image_path: "./data/characters/char_1.jpg"
```

## 6. 运行方式

```bash
python example_short_play_generation.py --config config/story_example.yaml
```

如果省略 `--config`，默认读取：

- `config/story_example.yaml`
