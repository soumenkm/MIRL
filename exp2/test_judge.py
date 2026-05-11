"""Quick smoke-test for the running judge server.

Usage:
    python test_judge.py                          # uses default connection file
    python test_judge.py --url http://IP:PORT     # override URL directly
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("test_judge.log", mode="w"),
    ],
)
logger = logging.getLogger("JudgeTest")

TEST_CASES = [
    {
        "name": "simple_math",
        "system": "You are a math judge. Given a problem and a solution, reply CORRECT or INCORRECT, then briefly explain.",
        "user": (
            "Problem: What is 12 × 15?\n"
            "Solution: 12 × 15 = 180\n"
            "Is this solution correct?"
        ),
    },
    {
        "name": "wrong_answer",
        "system": "You are a math judge. Given a problem and a solution, reply CORRECT or INCORRECT, then briefly explain.",
        "user": (
            "Problem: What is the square root of 144?\n"
            "Solution: The square root of 144 is 14.\n"
            "Is this solution correct?"
        ),
    },
    {
        "name": "multilingual_hindi",
        "system": "You are a math judge. Given a problem and a solution, reply CORRECT or INCORRECT, then briefly explain.",
        "user": (
            "Problem: 5 + 7 = ?\n"
            "Solution: पाँच और सात का योग बारह होता है।\n"  # "Five plus seven equals twelve"
            "Is this solution correct?"
        ),
    },
]


def load_url_from_connection_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        logger.error(f"Connection file not found: {path}")
        sys.exit(1)
    with p.open() as f:
        info = json.load(f)
    if not info.get("ready"):
        logger.error("Connection file says server is not ready yet.")
        sys.exit(1)
    return info["url"]


def run_test(client: OpenAI, model: str, case: dict) -> None:
    logger.info("=" * 60)
    logger.info(f"Test case : {case['name']}")
    logger.info(f"System    : {case['system']}")
    logger.info(f"User      : {case['user']}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": case["system"]},
            {"role": "user",   "content": case["user"]},
        ],
        temperature=0.0,
        max_tokens=256,
    )

    output = response.choices[0].message.content
    finish = response.choices[0].finish_reason
    usage  = response.usage

    logger.info(f"Output    : {output}")
    logger.info(f"Finish    : {finish}")
    logger.info(f"Tokens    : prompt={usage.prompt_tokens}, "
                f"completion={usage.completion_tokens}, "
                f"total={usage.total_tokens}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="Judge server URL e.g. http://192.168.1.44:8765")
    parser.add_argument("--connection-file", default="exp2/outputs/judge_connection.json")
    args = parser.parse_args()

    url = args.url or load_url_from_connection_file(args.connection_file)
    logger.info(f"Connecting to judge server at: {url}")

    client = OpenAI(base_url=f"{url}/v1", api_key="none")

    # Discover served model name
    models = client.models.list()
    model = models.data[0].id
    logger.info(f"Served model: {model}")

    for case in TEST_CASES:
        try:
            run_test(client, model, case)
        except Exception as e:
            logger.error(f"Test '{case['name']}' failed: {e}")

    logger.info("=" * 60)
    logger.info("All tests done. Check test_judge.log for full output.")


if __name__ == "__main__":
    main()