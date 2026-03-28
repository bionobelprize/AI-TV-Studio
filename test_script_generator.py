"""Tests for the ScriptGenerator pipeline module."""

import json
import os

import pytest

from src.models.character import CharacterEmotion
from src.pipeline.script_generator import ScriptGenerator


class DummyLLM:
    """Simple LLM stub that records prompts and returns a fixed response.

    Compatible with LangChain chain composition (``prompt | llm | parser``)
    because it implements ``__call__``, which LangChain wraps in a
    ``RunnableLambda`` automatically.
    """

    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, prompt):
        to_str = getattr(prompt, "to_string", None)
        prompt_str = to_str() if callable(to_str) else str(prompt)
        self.calls.append(prompt_str)
        return self.response

    def invoke(self, prompt):
        return self(prompt)


class MultiResponseDummyLLM:
    """LLM stub that returns a different response for each successive call.

    Responses are consumed in order; the last response is repeated if more
    calls are made than responses provided.  All prompts are recorded in
    ``calls`` for assertion in tests.
    """

    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.calls = []
        self._call_index = 0

    def __call__(self, prompt):
        to_str = getattr(prompt, "to_string", None)
        prompt_str = to_str() if callable(to_str) else str(prompt)
        self.calls.append(prompt_str)
        idx = min(self._call_index, len(self.responses) - 1)
        self._call_index += 1
        return self.responses[idx]

    def invoke(self, prompt):
        return self(prompt)


class ResponseObject:
    """Mimics an object response with a content field."""

    def __init__(self, content):
        self.content = content


class TestScriptGenerator:
    def _series_config(self):
        return {
            "title": "Midnight Detective",
            "genre": "crime",
            "characters": [
                {"id": "char_lin", "name": "Lin Xiao"},
                {"id": "char_chen", "name": "Chen Wei"},
            ],
        }

    def _valid_blueprint_payload(self):
        """Phase 1 response: episode structure without individual shots."""
        return {
            "episode_title": "The Silent Alley",
            "logline": "Lin follows a clue into an alley and finds a witness.",
            "scenes": [
                {
                    "scene_number": 1,
                    "location": "Old Alley",
                    "time_of_day": "night",
                    "weather": "fog",
                    "mood": "tense",
                    "bgm_mood": "suspenseful",
                    "bgm_tempo": "moderate",
                    "ambient_sounds": ["distant siren"],
                    "scene_summary": "Lin enters the alley and spots a shadowy figure.",
                }
            ],
        }

    def _valid_shots_payload(self):
        """Phase 2 response: shots for a single scene."""
        return {
            "shots": [
                {
                    "sequence_number": 1,
                    "action_description": "Lin enters the alley.",
                    "dialogue": "Stay where you are!",
                    "characters_in_shot": ["char_lin"],
                    "character_emotions": {
                        "char_lin": "neutral",
                        "char_chen": "not_an_emotion",
                    },
                    "duration": 7,
                    "camera_motion": {"type": "push_in", "speed": 1.2},
                    "lighting_description": "cold streetlight",
                    "text_prompt": "cinematic noir alley at night",
                },
                {
                    "sequence_number": 2,
                    "action_description": "A shadow disappears.",
                    "duration": 9,
                    "text_prompt": "shadow running through fog",
                },
            ]
        }

    def _valid_script_payload(self):
        """Legacy combined payload (kept for _parse_response tests)."""
        payload = self._valid_blueprint_payload()
        payload["scenes"][0]["shots"] = self._valid_shots_payload()["shots"]
        return payload

    def test_generate_episode_success(self):
        """Two-phase generation: blueprint call then per-scene shots call."""
        llm = MultiResponseDummyLLM([
            json.dumps(self._valid_blueprint_payload()),
            json.dumps(self._valid_shots_payload()),
        ])
        generator = ScriptGenerator(llm_client=llm)

        episode = generator.generate_episode(
            series_config=self._series_config(),
            episode_outline="A clue leads Lin into a dangerous alley.",
            episode_number=3,
        )

        assert episode.series_title == "Midnight Detective"
        assert episode.episode_number == 3
        assert episode.episode_title == "The Silent Alley"
        assert episode.characters == ["char_lin", "char_chen"]
        assert len(episode.scenes) == 1
        assert len(episode.scenes[0].shots) == 2
        assert episode.runtime_estimate == 16

        shot = episode.scenes[0].shots[0]
        assert shot.camera_motion is not None
        assert shot.camera_motion.type == "push_in"
        assert shot.character_emotions == {"char_lin": CharacterEmotion.NEUTRAL}

        # Two-phase: 1 blueprint call + 1 shots call (one per scene)
        assert len(llm.calls) == 2
        # Phase 1 blueprint prompt must include the strict format marker
        assert "STRICT OUTPUT FORMAT" in llm.calls[0]
        # Phase 2 shots prompt must embed the full episode blueprint for coherence
        assert "EPISODE BLUEPRINT" in llm.calls[1]

    def test_generate_episode_multiple_scenes(self):
        """Each scene in the blueprint triggers its own shots LLM call."""
        blueprint = {
            "episode_title": "Two Scenes",
            "logline": "A two-scene episode.",
            "scenes": [
                {
                    "scene_number": 1,
                    "location": "Rooftop",
                    "time_of_day": "dusk",
                    "weather": "clear",
                    "mood": "tense",
                    "bgm_mood": "dramatic",
                    "bgm_tempo": "fast",
                    "ambient_sounds": [],
                    "scene_summary": "Chase begins.",
                },
                {
                    "scene_number": 2,
                    "location": "Alley",
                    "time_of_day": "night",
                    "weather": "rain",
                    "mood": "dark",
                    "bgm_mood": "ominous",
                    "bgm_tempo": "slow",
                    "ambient_sounds": ["rain"],
                    "scene_summary": "Chase ends.",
                },
            ],
        }
        shots = {"shots": [{"sequence_number": 1, "action_description": "Shot.", "duration": 5, "text_prompt": "p"}]}
        llm = MultiResponseDummyLLM([
            json.dumps(blueprint),
            json.dumps(shots),
            json.dumps(shots),
        ])
        generator = ScriptGenerator(llm_client=llm)

        episode = generator.generate_episode(
            series_config=self._series_config(),
            episode_outline="A chase across the city.",
            episode_number=2,
        )

        assert len(episode.scenes) == 2
        # 1 blueprint call + 2 shots calls (one per scene)
        assert len(llm.calls) == 3
        # Both shots prompts embed the full blueprint
        assert "EPISODE BLUEPRINT" in llm.calls[1]
        assert "EPISODE BLUEPRINT" in llm.calls[2]
        # Each shots prompt references all scene summaries for coherence
        assert "Chase begins" in llm.calls[1]
        assert "Chase begins" in llm.calls[2]

    def test_format_all_scenes_summary(self):
        """Utility method formats scene list into a readable narrative overview."""
        scenes = [
            {"scene_number": 1, "location": "Park", "time_of_day": "morning", "mood": "calm", "scene_summary": "Hero wakes up."},
            {"scene_number": 2, "location": "Office", "time_of_day": "afternoon", "mood": "tense", "scene_summary": "Conflict arises."},
        ]
        summary = ScriptGenerator._format_all_scenes_summary(scenes)

        # Check that both scenes appear and in the right order
        idx1 = summary.index("Scene 1")
        idx2 = summary.index("Scene 2")
        assert idx1 < idx2, "Scene 1 must appear before Scene 2"

        # Verify key fields are present for each scene
        assert "Park" in summary
        assert "morning" in summary
        assert "calm" in summary
        assert "Hero wakes up" in summary

        assert "Office" in summary
        assert "afternoon" in summary
        assert "tense" in summary
        assert "Conflict arises" in summary

    def test_parse_response_accepts_markdown_wrapped_json(self):
        payload = self._valid_script_payload()
        raw = """```json\n""" + json.dumps(payload) + """\n```"""
        generator = ScriptGenerator(llm_client=DummyLLM("{}"))

        parsed = generator._parse_response(raw)
        assert parsed["episode_title"] == "The Silent Alley"

    def test_parse_response_raises_on_invalid_json(self):
        generator = ScriptGenerator(llm_client=DummyLLM("{}"))

        with pytest.raises(ValueError, match="Failed to parse LLM response as JSON"):
            generator._parse_response("this is not json")

    def test_extract_text_from_response_object_content_list(self):
        response = ResponseObject(
            [
                {"text": '{"episode_title": "E1", "logline": "L", "scenes": []}'},
                {"ignored": "x"},
            ]
        )
        generator = ScriptGenerator(llm_client=DummyLLM("{}"))

        text = generator._extract_text_from_response(response)
        assert '"episode_title": "E1"' in text

    def test_compute_runtime_sums_all_shot_durations(self):
        llm = MultiResponseDummyLLM([
            json.dumps(self._valid_blueprint_payload()),
            json.dumps(self._valid_shots_payload()),
        ])
        generator = ScriptGenerator(llm_client=llm)

        episode = generator.generate_episode(
            series_config=self._series_config(),
            episode_outline="Outline",
            episode_number=1,
        )
        assert generator._compute_runtime(episode) == 16


@pytest.mark.integration
def test_script_generator_real_run_prints_runtime_status():
    """Run ScriptGenerator with the real configured LLM and print key outputs.

    Enable with:
    RUN_REAL_SCRIPT_GENERATOR=1 python -m pytest -s -q tests/test_script_generator.py -k real_run
    """

    series_config = {
        "title": "Midnight Detective",
        "genre": "crime",
        "characters": [
            {"id": "char_lin", "name": "Lin Xiao"},
            {"id": "char_chen", "name": "Chen Wei"},
        ],
    }
    episode_outline = (
        '''致我的童年-老麻子和老辫子
一说到老麻子和老辫子，我就想到青青，我童年时期最好的朋友。Wxz 我现在35岁，我者半生或者三分之一生也没有什么好朋友，请求算是我小时后的饿以一个好朋友。想到她就想到很多很多的日子，还有我们之间的绰号，丢丢和啾啾，长棍和扁担。她矮我高，她脸圆我脸长，她找人喜欢我比较孤独，她比较聪明我感觉我挺笨的。我们一起度过了童年最美好的时光，我们说了很多彪话做了一些彪事，但是我们挺快乐，我还记得一起说故事笑得尿裤子，那是我人生很少的开怀大笑，后面几乎都没有了。如今我们中年了，我们都各自的伴侣各自的人生，很少联系几乎不联系，北京去年见了一面，16年她给我说了鼓励的话我还记得，你在某一领域有非凡的天赋你应该相信自己，我这辈子没有人对我说过这句话，我一直存记在心最后国画入选全国中国画展览，我还是要感激我后来的我先生，他说我是艺术家，我每次听到他这么说我都眼泪在眼圈，我觉得我一辈子没有被人这样认可过，我很委屈，很感动，也有无力和怀疑，我一辈子可能半辈子吧都在追求别人的认可，我感谢鼓励我的他吧，让我能写下老麻子和老辫子的故事。
真的很烦电脑卡，真的恨烦，想砸了它。我是个偶尔暴躁直接的艺术家。虽然我时常觉得自己一无是处，但是我仍然在用各种方式比如手机命名上说我是艺术家，我想给自己洗脑，然后希望有一天我就是了。
老麻子和老辫子，主角是老麻子，虽然只有两个人，但是更多的是围绕老麻子。老麻子原型是小学语文老师，她尖酸刻薄的样子，我提到她的名字就出现画面，她是一个更年期的老女人，干巴巴的瘦瘦小小，戴个假发油又亮那种特别廉价一看就特别假的假发，还带颜色，应该是红棕色，她买不起好的假发，好的假发她也不配，那个假发是齐刘海娃娃头的，她虽然很老还是把自己弄得很小，但是又假又怪，小三角眼睛低硫转，细柳眉毛染的那种老式的俗不可耐，小鼻子如果没有鼻孔都看不见鼻子了，小嘴最有特色，像腚眼一样抽巴在一起一揪一揪的，涂的通红的口红像猴腚一样。她穿的很土，但是头总是上扬，很自信的模样，不苟言笑，眼睛很少正眼看人总是斜眼弄那个死样子，然后训人的时候腚眼嘴就一蕨一蕨的。还有她一脸的斑和麻子，感觉都要从脸上迸出来。对了她还戴一副眼镜，但是我总觉得她没有素质没有学问很脏很恶心。除了她平常对我们严厉尖酸刻薄，她还是一个非常没有爱心的人。一个女同学来大姨妈卫生巾掉地上了，同学们踢来踢去被她发现，让女生站出来承认，还让女生围着卫生巾转一圈看是谁得，让那个女生极尽难看让我们这些其他女生也很尴尬，最后也没有人承认，但是大概她从谁的脸最红已经发现了，通过这件事我真的对她厌恶透顶，觉得她才是那个满身骚气然后不自知的人。
老辫子有点傻，是衬托老麻子的，是一个数学老师，也挺二逼的，比老麻子还二。原型是个数学老师。因为一个女学生撒谎没写作业当众在班级门口删她脸，我到现在都记得那个女生的鼻涕和她尖利的牙好像要吃人一般。这些二逼老师没有一点智慧素质和人味，不值得怎么当上好事的。她的特点是辫子很粗很长，大嘴唇，大眼睛，大鼻孔，很高，和老麻子相反，憨憨傻傻的，很听老麻子的话，虽然老麻子常常利用她耍他，但是她还是傻傻的跟随。
故事很乱，都是即兴的大概，都是一些彪事，老麻子老辫子是拥有一些特意功能的非一般人，比如老麻子的头发可以变色因为老麻子虽然丑但是她爱美而且她不自知，她真的觉得自己美，老麻子的腋臭很大可以作为攻击的烟雾弹，老麻子有特殊癖好喜欢偷人内裤穿，老麻子很脏把身上的灰搓下来当作大力丸去攻击人或者给老辫子吃或者卖给别人。
一天，老麻子裤瘾犯了，她想偷内裤了，她看到一个美女然后觉得人家美，觉得自己如果穿上她的内裤就比她还要美，于是她尾随美女到她家找好门牌，夜里夜深人静准备闯入偷内裤，结果她半夜爬上楼（她有很多特异功能，比如像猴一样爬楼）从窗进入房间，她笨手笨脚还有身上味道太大被发现了，美女一生惊吼是小偷来人啊，她来不及就慌张从窗跳下来结果砸到头起了个大包（老麻子生命力顽强很抗造），她跑的途中被美女的男朋友追上了，然后她以为自己的美貌吸引来帅哥，结果那男的上来一顿拳脚相加，老麻子很伤心说你怎么打美女呢，美女男朋友以为是变态打的更狠了，老麻子感觉赵家不住呼叫老辫子，她在天空放了一个烟雾弹，老辫子转动自己的辫子直升机闻弹赶来救援，老麻子抓着老辫子的脚老辫子拼命旋转直升机脑袋他们飞上天逃跑了，老麻子被打的鼻青脸肿老辫子笑得不行，她的假发也乱了，老麻子和老辫子回到他们的老巢，他们在荒郊野岭深林里书上搭窝，因为这样没有人发现他们，老麻子炫耀她偷回来的内衣，说她是天下最美的女人。老辫子连声负荷，说还是麻子姐姐厉害，辫子妹妹佩服佩服。老麻子虽然满脸青还是很欢喜，答应老辫子，下回多偷一个送给她。

'''
    )

    generator = ScriptGenerator()
    episode = generator.generate_episode(
        series_config=series_config,
        episode_outline=episode_outline,
        episode_number=1,
    )

    print("\n=== ScriptGenerator Real Run ===")
    print(f"episode_id={episode.id}")
    print(f"title={episode.episode_title}")
    print(f"logline={episode.logline}")
    print(f"scenes={len(episode.scenes)}")
    print(f"runtime_estimate={episode.runtime_estimate}")
    for scene in episode.scenes:
        print(
            f"scene#{scene.scene_number} location={scene.location} shots={len(scene.shots)}"
        )
        for shot in scene.shots:
            motion = shot.camera_motion.type if shot.camera_motion else "none"
            print(
                f"  shot#{shot.sequence_number} duration={shot.duration} motion={motion}"
            )

    assert episode.episode_title
    assert len(episode.scenes) > 0
    assert episode.runtime_estimate > 0

if __name__ == "__main__":
    test_script_generator_real_run_prints_runtime_status()