# config/

Central configuration package for AI Job Hunter.

Every module in the project imports its settings and constants from here.
No other module reads environment variables or defines domain enums directly.

---

## Files

### `settings.py`

Runtime configuration backed by Pydantic `BaseSettings`.

- Reads values from the `.env` file in the project root
- Validates types at startup — misconfigured values fail immediately, not at runtime
- Exposes a single `settings` singleton used everywhere
- Covers: AI provider, user profile, database, scraping, storage paths, notifications, feature flags

### `constants.py`

Fixed domain values that do not change between environments.

- `ATSType` — enum of supported Applicant Tracking Systems
- `JobStatus` — enum of job application lifecycle states
- `TARGET_COMPANIES` — list of companies with their known ATS platform
- `TARGET_ROLES` — list of target job titles used for AI matching

### `__init__.py`

Re-exports `settings`, `ATSType`, `JobStatus`, `TARGET_COMPANIES`, and `TARGET_ROLES`
so every module can use a single import:

```python
from config import settings, ATSType, JobStatus, TARGET_COMPANIES
```

---

## How to Use

**1. Copy and fill the example env file:**

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY and USER_NAME at minimum
```

**2. Import settings anywhere:**

```python
from config import settings

api_key = settings.anthropic_api_key
model   = settings.ai_model
```

**3. Import constants anywhere:**

```python
from config import ATSType, JobStatus, TARGET_COMPANIES

for company in TARGET_COMPANIES:
    print(company["name"], company["ats"])
```

**4. Check a feature flag before running optional logic:**

```python
from config import settings

if settings.ai_ranking_enabled:
    # call the ranking agent
```

---

## Adding a New Company

Open `constants.py` and add an entry to `TARGET_COMPANIES`:

```python
{"name": "New Company", "ats": ATSType.GREENHOUSE},
```

No code changes are needed anywhere else.

## Adding a New Setting

Add a field to the `Settings` class in `settings.py` with a `Field` descriptor,
then add the corresponding key to `.env.example`.
