# agents/

AI agents that perform intelligent tasks across the job hunting pipeline.

Each agent is responsible for a single, well-defined task and communicates with the LLM via the prompt templates in `../prompts/`.

## Planned Agents

- **RankingAgent** — Scores and ranks job postings against the user's profile
- **ResumeTailoringAgent** — Rewrites resume bullets to match a specific job description
- **CoverLetterAgent** — Generates a personalized cover letter for a job posting
- **ReferralAgent** — Drafts a referral request message for a given company and role
- **InterviewPrepAgent** — Produces a structured interview preparation guide

## Design Notes

- Agents are stateless — they receive input, call the LLM, and return output
- Agents do not access the database directly; they receive data from services
- Prompt templates live in `../prompts/`, not inside agent code
