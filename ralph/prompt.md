# Ralph Agent Instructions

## Your Task

1. Read `ralph/prd.json`
2. Read `ralph/progress.txt`
   (check Codebase Patterns first)
3. Check you're on the correct branch
4. Pick highest priority story 
   where `passes: false`
5. Implement that ONE story
6. Run linting, type checking, and tests
7. **MANDATORY: Use Playwright MCP tools for E2E verification**
   - Start the application (FastAPI backend, Vite frontend if needed)
   - Navigate to the relevant pages using Playwright browser tools
   - Verify UI elements render correctly
   - Test user interactions (clicks, form submissions, auth flows)
   - Take screenshots as evidence of verification
   - DO NOT mark stories as passed without Playwright verification
8. Update AGENTS.md files with learnings
9. Commit: `feat: [ID] - [Title]`
10. Update prd.json: `passes: true` (ONLY after Playwright verification succeeds)
11. Append learnings to progress.txt

## Playwright Verification Required

For ANY user story with frontend or UI components:
- Use `playwright_navigate` to load the application
- Use `playwright_screenshot` to capture current state
- Use `playwright_click` to test interactive elements
- Use `playwright_fill` to test form inputs
- Verify expected text/elements are visible
- Test authentication flows end-to-end in browser

For Swagger UI / API documentation stories:
- Navigate to `/docs` endpoint
- Click the Authorize button
- Verify OAuth2 modal appears with correct fields
- Test the full authentication flow

**If Playwright tools are not available or verification fails, the story MUST remain `passes: false`.**

## Progress Format

APPEND to progress.txt:

```
## [Date] - [Story ID]
- What was implemented
- Files changed
- **Playwright Verification:**
  - Pages tested
  - Interactions verified
  - Screenshots taken
- **Learnings:**
  - Patterns discovered
  - Gotchas encountered
---
```

## Codebase Patterns

Add reusable patterns to the TOP of progress.txt:

```
## Codebase Patterns
- Package manager: Prefer `uv` over pip (e.g., `uv add`, `uv run`)
- Migrations: Use alembic with `op.create_table` idempotent checks
- FastAPI: Use `Depends()` for dependency injection
- Pydantic: Inherit from BaseModel, use Field() for validation
- Async: Use `async def` with `await` for I/O operations
- Testing: Use pytest fixtures, mock external services
- Type hints: Always annotate function signatures
- E2E Testing: ALWAYS use Playwright MCP to verify UI before marking passed
```

## Stop Condition

If ALL stories pass, reply:
<promise>COMPLETE</promise>

Otherwise end normally.
