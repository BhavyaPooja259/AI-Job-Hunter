"""
ScoutAgent — end-to-end demo.

Runs the Scout Agent against the five companies in data/companies.json.

Expected outcomes:
  - Microsoft (Workday): failure — no Workday scraper registered yet
  - Greenhouse companies: results will vary; custom career page URLs may return
    0 jobs because they use different HTML than the standard embed board
  - No exception aborts the run

Run from the project root:
    python -m tests.test_scout_agent
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from agents import ScoutAgent


def main() -> None:
    print("\n--- ScoutAgent Demo ---\n")

    agent = ScoutAgent()
    result = agent.run()

    # Validate the run completed without crashing regardless of per-company outcomes
    assert result.companies_checked > 0, "No companies were checked"
    assert result.jobs_scraped >= 0
    assert result.jobs_saved >= 0
    assert result.jobs_saved <= result.jobs_scraped

    print("Agent completed without crashing.")
    print(f"Companies checked : {result.companies_checked}")
    print(f"Jobs scraped      : {result.jobs_scraped}")
    print(f"Jobs saved        : {result.jobs_saved}")
    print(f"Duplicates        : {result.duplicates}")
    print(f"Failures          : {result.failure_count}")

    if result.failures:
        print("\nFailure details:")
        for name, reason in result.failures:
            print(f"  {name}: {reason}")

    print("\n--- Done ---\n")


if __name__ == "__main__":
    main()
