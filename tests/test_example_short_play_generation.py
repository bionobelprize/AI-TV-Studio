from example_short_play_generation import build_character, plan_shots
from src.models.episode import Episode
from src.models.scene import Scene
from src.models.shot import GenerationMode, Shot
from src.studio import AITVStudio


def test_plan_shots_populates_reference_images_from_registered_characters():
    studio = AITVStudio(config={"data_dir": "./data", "output_dir": "./outputs"})
    character = build_character(
        {
            "id": "char_narrator",
            "name": "叙述者",
            "age": 35,
            "gender": "female",
            "occupation": "artist",
            "visual_core": {
                "base_image_path": "./data/characters/char_narrator_base.jpg",
                "reference_prompt": "artist in a studio",
                "key_features": "tired eyes",
            },
        }
    )
    studio.register_character(character)

    episode = Episode(
        id="ep1",
        series_title="Test Series",
        episode_number=1,
        episode_title="Pilot",
        logline="A test episode.",
        scenes=[
            Scene(
                id="scene_1",
                episode_id="ep1",
                scene_number=1,
                location="Studio",
                time_of_day="night",
                weather="clear",
                mood="reflective",
                shots=[
                    Shot(
                        id="shot_1",
                        scene_id="scene_1",
                        sequence_number=0,
                        action_description="Narrator stares at the screen.",
                        characters_in_shot=["char_narrator"],
                    )
                ],
            )
        ],
    )

    planned_episode = plan_shots(studio, episode)
    planned_shot = planned_episode.scenes[0].shots[0]

    assert planned_shot.generation_mode == GenerationMode.REFERENCE_TO_VIDEO
    assert planned_shot.reference_images == ["./data/characters/char_narrator_base.jpg"]