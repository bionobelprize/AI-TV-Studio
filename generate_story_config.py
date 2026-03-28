"""CLI entry-point: generate a story_example.yaml from a plain-text outline.

Usage (from the project root):

    # Read outline from stdin, print YAML to stdout
    python generate_story_config.py

    # Read outline from a text file, write YAML to a target path
    python generate_story_config.py --outline outline.txt --output config/my_story.yaml

    # Pass outline directly as a string argument
    python generate_story_config.py --text "A detective and her AI partner investigate..."

    # Specify episode number (default: 1)
    python generate_story_config.py --text "..." --episode 3
"""

import argparse
import sys
from pathlib import Path


def _read_outline(args: argparse.Namespace) -> str:
    """Resolve and return the outline text from CLI arguments or stdin."""
    if args.text:
        return args.text.strip()
    if args.outline:
        path = Path(args.outline)
        if not path.exists():
            sys.exit(f"Error: outline file not found: {path}")
        return path.read_text(encoding="utf-8").strip()
    # Fall back to stdin
    print("Enter the episode outline (press Ctrl+D / Ctrl+Z to finish):", file=sys.stderr)
    return sys.stdin.read().strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a story_example.yaml from a free-text episode outline."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--outline", "-i",
        metavar="FILE",
        help="Path to a plain-text file containing the episode outline.",
    )
    source.add_argument(
        "--text", "-t",
        metavar="TEXT",
        help="Episode outline as an inline string.",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        default=None,
        help="Output path for the generated YAML (default: print to stdout).",
    )
    parser.add_argument(
        "--episode", "-e",
        type=int,
        default=1,
        metavar="N",
        help="Episode number to embed in the config (default: 1).",
    )
    args = parser.parse_args()

    outline = _read_outline(args)
    if not outline:
        sys.exit("Error: outline must not be empty.")

    # Lazy import so startup is cheap when --help is requested.
    from src.pipeline.story_config_generator import StoryConfigGenerator

    gen = StoryConfigGenerator(episode_number=args.episode)
    print("Generating story config via LLM...", file=sys.stderr)
    yaml_text = gen.generate(outline)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml_text, encoding="utf-8")
        print(f"Saved: {out_path}", file=sys.stderr)
    else:
        print(yaml_text)


if __name__ == "__main__":
    main()
