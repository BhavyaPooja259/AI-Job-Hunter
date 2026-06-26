"""
AI Job Hunter — entry point.

Runs one full Scout scan across all active companies.

Usage:
    python main.py
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from agents import ScoutAgent


def main() -> None:
    agent = ScoutAgent()
    result = agent.run()
    sys.exit(0 if result.failure_count < result.companies_checked else 1)


if __name__ == "__main__":
    main()
