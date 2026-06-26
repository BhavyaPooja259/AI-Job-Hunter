# AI Job Hunter

A personal AI-powered Job Hunting Assistant built to help land interviews and offers at top product companies.

## What This Does

- Monitors company career pages for new job postings
- Matches jobs against your profile using AI
- Tailors resumes and generates cover letters
- Drafts referral messages
- Tracks applications through the full lifecycle
- Sends daily summaries of new opportunities

## Project Structure

| Folder | Purpose |
|--------|---------|
| `agents/` | AI agents for ranking, tailoring, and generation tasks |
| `browser/` | Headless browser automation for dynamic career pages |
| `config/` | All configuration files and settings |
| `data/` | Seed data, company lists, and static reference files |
| `database/` | Database models, migrations, and repository layer |
| `docs/` | Architecture decisions, guides, and reference documentation |
| `notifications/` | Daily digest and alert delivery |
| `prompts/` | LLM prompt templates for every AI task |
| `scrapers/` | ATS-specific adapters (Greenhouse, Lever, Ashby, Workday, etc.) |
| `services/` | Core business logic and orchestration layer |
| `storage/` | File-based output: resumes, cover letters, reports |
| `tests/` | Unit and integration tests |
| `utils/` | Shared utilities used across modules |

## Guides

- [CLAUDE.md](CLAUDE.md) — Project rules and development guidelines
- [ROADMAP.md](ROADMAP.md) — Phased feature roadmap
