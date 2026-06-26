# AI Job Hunter — Project Roadmap

This roadmap organizes the project into logical phases, each building on the previous.
No phase should be started until the previous one is stable and confirmed.

---

## Phase 1 — Foundation

### Milestone 1.1 — Project Setup

**Goal:** Establish a clean, runnable project skeleton.

**Why it exists:** Every subsequent feature depends on a consistent project structure, dependency management, and a working build system.

**Features included:**
- Java + Spring Boot project initialization
- Build tool setup (Maven or Gradle)
- Package structure defined (modular from day one)
- Application entry point
- Logging configuration
- `.gitignore` and basic project hygiene

**Expected outcome:** The project builds, runs, and logs a startup message. Nothing else.

**Estimated complexity:** Low

---

### Milestone 1.2 — Configuration System

**Goal:** Centralize all runtime configuration so no values are hardcoded anywhere.

**Why it exists:** The system will talk to multiple ATS platforms, AI providers, and file paths. A clean config layer prevents scattered magic strings and makes the system easy to adapt.

**Features included:**
- `application.yml` with profiles (dev, prod)
- Typed configuration classes (`@ConfigurationProperties`)
- Company list configuration (loaded from file or config)
- AI provider keys and endpoints
- Output directory paths
- Feature flags for enabling/disabling modules

**Expected outcome:** All tunable parameters live in one place. Changing a company or API key requires no code change.

**Estimated complexity:** Low

---

### Milestone 1.3 — Company Database

**Goal:** Define and persist the list of target companies the system monitors.

**Why it exists:** The entire job discovery pipeline is driven by the company list. It must be structured, queryable, and easy to extend without code changes.

**Features included:**
- Company entity (name, careers page URL, ATS type, active flag)
- Local database setup (SQLite or H2 for local-first)
- Seed data for initial 18 target companies
- Repository layer for CRUD operations
- CLI or config-driven way to add new companies

**Expected outcome:** The system knows which companies to watch. Adding a new company takes one line in a config or seed file.

**Estimated complexity:** Low–Medium

---

## Phase 2 — Job Discovery

### Milestone 2.1 — Browser Automation Layer

**Goal:** Build a controlled, reusable browser automation foundation.

**Why it exists:** Many career pages require JavaScript rendering. A headless browser layer enables the system to interact with dynamic pages that raw HTTP clients cannot handle.

**Features included:**
- Playwright or Selenium integration
- Headless browser configuration
- Page load and wait utilities
- Screenshot capture for debugging
- Retry and timeout handling
- Rate limiting and polite crawling delays

**Expected outcome:** The system can open any URL in a headless browser, wait for content to load, and extract the page HTML reliably.

**Estimated complexity:** Medium

---

### Milestone 2.2 — ATS Integrations

**Goal:** Implement dedicated adapters for each major Applicant Tracking System.

**Why it exists:** Each ATS (Greenhouse, Lever, Ashby, Workday, Oracle, SuccessFactors) has a distinct URL structure and HTML layout. Dedicated adapters make extraction reliable and maintainable per platform.

**Features included:**
- Common `AtsAdapter` interface
- Greenhouse adapter
- Lever adapter
- Ashby adapter
- Workday adapter
- Oracle Careers adapter
- SuccessFactors adapter
- Fallback generic scraper for unknown ATS platforms

**Expected outcome:** Given a company and its ATS type, the system can extract a list of job postings from that company's career page.

**Estimated complexity:** High (one adapter at a time)

---

### Milestone 2.3 — Career Page Discovery

**Goal:** Automatically detect which ATS a company uses if not already known.

**Why it exists:** As the company list grows, manually identifying the ATS for each company becomes impractical. Auto-detection keeps the system scalable.

**Features included:**
- URL and HTML fingerprinting to identify ATS type
- Confidence scoring per detection
- Manual override support in company config
- Logging of unknown/undetected ATS for manual review

**Expected outcome:** The system can be pointed at any company careers URL and correctly identify or approximate the ATS in use.

**Estimated complexity:** Medium

---

### Milestone 2.4 — Job Discovery Pipeline

**Goal:** Orchestrate the full end-to-end job fetching process across all configured companies.

**Why it exists:** Individual ATS adapters work in isolation. This milestone wires them together into a scheduled, automated pipeline that fetches jobs from all companies.

**Features included:**
- Job discovery scheduler (configurable interval)
- Per-company fetch orchestration
- Raw job data normalization into a common `Job` model
- Error handling and per-company failure isolation
- Discovery run logging and audit trail

**Expected outcome:** Running the discovery pipeline fetches current job listings from all configured companies and produces a normalized list of jobs.

**Estimated complexity:** Medium

---

### Milestone 2.5 — Duplicate Detection

**Goal:** Ensure each unique job is stored exactly once, even across multiple discovery runs.

**Why it exists:** Career pages are crawled repeatedly. Without deduplication, the same job accumulates many copies, polluting the database and confusing downstream ranking and tracking.

**Features included:**
- Job fingerprinting (hash of title + company + location + posting date)
- Duplicate check before insert
- Detection of jobs that have been updated vs. re-posted
- Deduplication report per discovery run

**Expected outcome:** Re-running the discovery pipeline on the same companies never creates duplicate job records.

**Estimated complexity:** Low–Medium

---

### Milestone 2.6 — Job Storage

**Goal:** Persist all discovered jobs in a structured, queryable local database.

**Why it exists:** Downstream features (ranking, tracking, tailoring) all need reliable access to the full job history. A well-designed schema here makes everything downstream easier.

**Features included:**
- `Job` entity with all relevant fields (title, company, location, URL, description, date found, ATS type, status)
- Repository layer with filtering and search support
- Job status lifecycle (new, reviewed, applied, rejected, archived)
- Query support: by company, by date, by status, by keyword

**Expected outcome:** Every discovered job is persisted locally and can be queried by any attribute.

**Estimated complexity:** Low–Medium

---

## Phase 3 — Intelligence

### Milestone 3.1 — AI Job Ranking

**Goal:** Use an LLM to score and rank discovered jobs against my profile.

**Why it exists:** Dozens of jobs can be discovered per day. AI ranking surfaces the most relevant ones first so time is spent on high-fit opportunities.

**Features included:**
- Profile context loader (my background, skills, target roles)
- Prompt engineering for job relevance scoring
- LLM integration (Claude API)
- Score and rationale stored per job
- Ranked job list output (CLI or file)

**Expected outcome:** Every new job receives a relevance score and a short explanation of why it is or isn't a good fit.

**Estimated complexity:** Medium

---

### Milestone 3.2 — Resume Tailoring

**Goal:** Generate a tailored version of my resume for a specific job posting.

**Why it exists:** A generic resume underperforms a targeted one. AI-assisted tailoring saves hours of manual work per application.

**Features included:**
- Base resume loaded from a structured source (JSON or YAML)
- Job description parsed and key requirements extracted
- LLM prompt to align resume bullets with job requirements
- Tailored resume output in Markdown and plain text
- Diff view showing what changed from base resume

**Expected outcome:** Given a job, the system produces a tailored resume in under a minute.

**Estimated complexity:** Medium–High

---

### Milestone 3.3 — Cover Letter Generation

**Goal:** Generate a personalized cover letter for a specific job posting.

**Why it exists:** Cover letters that directly reference the company's values, tech stack, and the role requirements perform significantly better than templates.

**Features included:**
- Cover letter prompt with job description + company context + my background
- Multiple tone options (formal, conversational)
- Output in Markdown and plain text
- Regeneration support with different prompts

**Expected outcome:** Given a job, the system generates a ready-to-use cover letter in seconds.

**Estimated complexity:** Medium

---

### Milestone 3.4 — Referral Assistant

**Goal:** Draft a professional, personalized referral request message for a specific job and company.

**Why it exists:** Referrals dramatically increase interview odds. A well-crafted message to the right person needs to be concise and genuine — AI can help draft and personalize it.

**Features included:**
- Referral message prompt with job title, company, and my background
- Multiple message styles (LinkedIn DM, email)
- Placeholder support for recipient name
- Output as ready-to-send text

**Expected outcome:** Given a job, the system produces a referral message ready to send in one edit.

**Estimated complexity:** Low–Medium

---

## Phase 4 — Tracking & Workflow

### Milestone 4.1 — Application Tracker

**Goal:** Track the full lifecycle of every job application in one place.

**Why it exists:** Without tracking, it is easy to lose sight of where applications stand, miss follow-up windows, or apply to the same role twice.

**Features included:**
- Application status workflow: bookmarked → applied → OA → phone screen → interview → offer → rejected
- Date tracking per status transition
- Notes field per application
- Filter and search across all applications
- Export to CSV

**Expected outcome:** Every application has a clear status and history. Nothing falls through the cracks.

**Estimated complexity:** Medium

---

### Milestone 4.2 — Notifications

**Goal:** Deliver a daily summary of new jobs and application updates.

**Why it exists:** The system works in the background. Proactive notifications ensure nothing important goes unnoticed without requiring the user to check a dashboard manually.

**Features included:**
- Daily digest of new high-ranked jobs
- Application status change alerts
- Configurable delivery (terminal output, local file, or macOS notification)
- Quiet hours configuration

**Expected outcome:** Each morning, a summary arrives with new jobs worth reviewing and any application updates.

**Estimated complexity:** Low–Medium

---

## Phase 5 — Interface

### Milestone 5.1 — Dashboard

**Goal:** Provide a single view of everything happening across the job search.

**Why it exists:** As data accumulates across companies, jobs, and applications, a unified view is essential for making decisions quickly.

**Features included:**
- New jobs discovered today
- Top-ranked jobs not yet actioned
- Active applications by status
- Companies being monitored
- Recent activity log
- Delivery as terminal output or local HTML report

**Expected outcome:** One command shows the complete state of the job search.

**Estimated complexity:** Medium

---

## Phase 6 — Interview Preparation

### Milestone 6.1 — Interview Preparation Assistant

**Goal:** Generate targeted interview preparation material for a specific company and role.

**Why it exists:** Interview prep is the most time-intensive part of the job search. AI can bootstrap company-specific prep instantly.

**Features included:**
- Common interview questions per role type (Backend, Platform)
- Company-specific system design topics (based on known tech stack)
- Behavioral question bank aligned with company values
- Java and Spring Boot coding question suggestions
- Study plan generation based on available time

**Expected outcome:** Given a company and role, the system produces a structured prep guide within seconds.

**Estimated complexity:** Medium

---

## Phase 7 — Future AI Features

### Milestone 7.1 — Advanced AI Capabilities (Future)

**Goal:** Expand intelligence across the full pipeline as the system matures.

**Why it exists:** These features require the foundation built in earlier phases and represent the long-term ceiling of the assistant.

**Features included (exploratory):**
- Auto-apply to pre-screened roles (with human confirmation gate)
- Salary negotiation guidance per offer
- Competitor analysis (which companies are hiring aggressively)
- Skill gap detection based on repeated missing requirements
- Interview outcome tracking with pattern detection
- AI-powered follow-up email drafting
- Personalized learning plan based on job requirement gaps

**Expected outcome:** The system becomes a full career coaching assistant, not just a job discovery tool.

**Estimated complexity:** High

---

## Summary

| Phase | Focus | Milestones |
|-------|-------|------------|
| 1 | Foundation | Setup, Configuration, Company Database |
| 2 | Job Discovery | Browser Automation, ATS Integrations, Career Page Discovery, Discovery Pipeline, Deduplication, Storage |
| 3 | Intelligence | AI Ranking, Resume Tailoring, Cover Letters, Referral Messages |
| 4 | Tracking | Application Tracker, Notifications |
| 5 | Interface | Dashboard |
| 6 | Interview Prep | Interview Preparation Assistant |
| 7 | Future AI | Advanced Automation and Coaching |
