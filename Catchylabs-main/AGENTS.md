# CatchyLab Project Rules

## Database Safety
- **NEVER** run destructive SQL (`DELETE`, `DROP`, `TRUNCATE`, `ALTER`) on the live production database (`DATABASE_URL`) without explicit user confirmation first.
- All tests MUST use `TEST_DATABASE_URL` only — never `DATABASE_URL`.
- Test fixtures must roll back all changes at the end of each test.

## External API Safety
- **No live Gemini or SerpAPI calls** unless the user explicitly asks for them.
- All tests must mock Gemini and SerpAPI calls.

## Secrets & Environment
- **NEVER** open, print, log, or display the contents of `.env` files.
- Do not commit or expose API keys, database credentials, or tokens.

## Content Approval
- **NEVER** mark content as `approved` automatically. Only a human reviewer may approve content.

## Test Database
- Tests use `TEST_DATABASE_URL` environment variable exclusively.
- If `TEST_DATABASE_URL` is missing or equals `DATABASE_URL`, the test session must abort immediately.

## Git Safety
- **NEVER** run `git checkout`, `git reset`, `git clean`, or `git stash` without asking first.

