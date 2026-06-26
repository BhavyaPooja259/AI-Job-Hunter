"""
Demo: add sample referral contacts and generate outreach messages.

Pipeline
--------
  1. Open the referral database (data/referrals.db).
  2. Track three sample contacts at target companies.
  3. Generate all three outreach messages for each contact using the
     deterministic template engine (no API key required).
  4. Advance one contact's status to show the lifecycle in action.
  5. Print all generated messages and the outreach statistics dashboard.

No-AI note
----------
Message generation works entirely without a Gemini API key.  Templates
produce professional, personalized messages from the contact fields.
To enable AI-enhanced messages, set GEMINI_API_KEY in your .env file.

Run
---
  python run_referral_demo.py
"""

import logging
import sys
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)s — %(message)s",
)
for ns in ("agents", "referral"):
    logging.getLogger(ns).setLevel(logging.INFO)

from config import settings
from agents.referral_agent import ReferralAgent
from referral.referral_repository import ReferralRepository
from referral.referral_status import ReferralStatus

BANNER = "=" * 60

# Sample contacts — realistic data for demo purposes
_SAMPLE_CONTACTS = [
    {
        "contact_name": "Arjun Nair",
        "company": "Stripe",
        "contact_title": "Senior Backend Engineer",
        "job_title": "Software Engineer II",
        "job_url": "https://stripe.com/jobs/listing/software-engineer",
        "contact_url": "https://linkedin.com/in/arjun-nair-stripe",
    },
    {
        "contact_name": "Meera Krishnan",
        "company": "Databricks",
        "contact_title": "Staff Engineer",
        "job_title": "Backend Engineer",
        "job_url": "https://databricks.com/company/careers/engineering",
        "contact_url": "https://linkedin.com/in/meera-krishnan",
    },
    {
        "contact_name": "Sahil Verma",
        "company": "Atlassian",
        "contact_title": "Engineering Manager",
        "job_title": "Platform Engineer",
        "job_url": "https://www.atlassian.com/company/careers",
        "contact_url": "https://linkedin.com/in/sahil-verma-atlassian",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(BANNER)
    print("  Referral Assistant Demo")
    print(BANNER)

    candidate_name = settings.user_name or "Bhavya L"
    candidate_email = settings.user_email or ""

    # Decide AI vs template mode
    client = None
    if settings.gemini_api_key:
        from google import genai
        client = genai.Client(api_key=settings.gemini_api_key)

    with ReferralRepository() as repo:
        agent = ReferralAgent(
            repo=repo,
            client=client,
            candidate_name=candidate_name,
            candidate_email=candidate_email,
            candidate_title="Software Engineer II",
        )

        # ── step 1: track contacts ────────────────────────────────────────
        print()
        print("[ 1/4 ]  Adding sample referral contacts")
        print("-" * 40)

        mode = "AI (Gemini)" if client else "template (no API key)"
        print(f"  Candidate : {candidate_name}")
        print(f"  Mode      : {mode}")
        print()

        referrals = []
        for contact in _SAMPLE_CONTACTS:
            ref, is_new = agent.track(**contact)
            status_tag = "NEW" if is_new else "EXISTS"
            print(f"  [{status_tag}]  {ref.contact_name} @ {ref.company}"
                  f"  ({ref.job_title})")
            referrals.append(ref)

        # ── step 2: generate messages for all contacts ────────────────────
        print()
        print("[ 2/4 ]  Generating outreach messages")
        print("-" * 40)
        if client:
            print("  Sending contacts to Gemini … (may take ~20 s)")
        else:
            print("  Assembling from templates …")
            print("  Tip: set GEMINI_API_KEY for AI-personalised messages.")
        print()

        all_messages = []
        for ref in referrals:
            msgs = agent.generate_messages(ref, save=True)
            all_messages.append((ref, msgs))
            print(f"  ✓  {ref.contact_name} @ {ref.company}")

        # ── step 3: show lifecycle advance ────────────────────────────────
        print()
        print("[ 3/4 ]  Simulating lifecycle — advancing first contact")
        print("-" * 40)

        first_ref = referrals[0]
        prev_status = first_ref.status
        if first_ref.status == ReferralStatus.NOT_CONTACTED:
            first_ref = agent.advance(
                first_ref.id,
                ReferralStatus.REQUEST_SENT,
                notes="Sent connection request on LinkedIn",
            )
            print(f"  {first_ref.contact_name} @ {first_ref.company}")
            print(f"  {prev_status.value}  →  {first_ref.status.value}")
        else:
            print(f"  {first_ref.contact_name} @ {first_ref.company}")
            print(f"  Already at: {first_ref.status.value}  (advance skipped on repeat run)")
        if first_ref.contacted_at:
            print(f"  contacted_at : {first_ref.contacted_at.strftime('%Y-%m-%d %H:%M')}")

        # ── step 4: print messages and stats ──────────────────────────────
        for ref, msgs in all_messages:
            print()
            print(BANNER)
            print(f"  MESSAGES — {ref.contact_name} @ {ref.company}")
            print(f"  Role: {ref.job_title or 'General outreach'}")
            print(BANNER)

            print()
            print("  [1] LinkedIn Connection Request")
            print(f"      ({len(msgs.linkedin_request)}/300 chars)")
            print("  " + "-" * 54)
            for line in msgs.linkedin_request.splitlines():
                print(f"  {line}")

            print()
            print("  [2] Referral Request Message")
            print("  " + "-" * 54)
            for line in msgs.referral_message.splitlines():
                print(f"  {line}")

            print()
            print("  [3] Follow-up Message")
            print("  " + "-" * 54)
            for line in msgs.followup_message.splitlines():
                print(f"  {line}")

        # ── stats dashboard ───────────────────────────────────────────────
        print()
        print(BANNER)
        print("  OUTREACH STATISTICS")
        print(BANNER)
        stats = agent.stats()
        print(f"  {stats.summary()}")
        print()
        if stats.by_status:
            for status_val, count in sorted(stats.by_status.items()):
                bar = "█" * count
                print(f"    {status_val:<20}  {bar}  ({count})")
        print()
        print(BANNER)
        print()


if __name__ == "__main__":
    main()
