# Add gemini-as-claude wrapper script

## Overview
Add a `scripts/gemini-as-claude/` directory with a wrapper script that translates Gemini CLI plain-text output into Claude-compatible stream-json format, following the same conventions as `codex-as-claude` and `opencode-as-claude`. Includes test script and README, plus a docs update.

## Context
- Files involved:
  - Create: `scripts/gemini-as-claude/gemini-as-claude.sh`
  - Create: `scripts/gemini-as-claude/gemini-as-claude_test.sh`
  - Create: `scripts/gemini-as-claude/README.md`
  - Modify: `docs/custom-providers.md`
- Related patterns: `scripts/codex-as-claude/`, `scripts/opencode/`
- Key difference: Gemini CLI outputs plain text (not JSONL), so each output line is wrapped as a `content_block_delta` event (simpler translation than codex/opencode)

## Development Approach
- Testing approach: Regular (script first, then tests)
- Complete each task fully before moving to the next
- **CRITICAL: every task MUST include new/updated tests**
- **CRITICAL: all tests must pass before starting next task**

## Implementation Steps

### Task 1: Create gemini-as-claude.sh

**Files:**
- Create: `scripts/gemini-as-claude/gemini-as-claude.sh`

- [x] Add shebang, header comment with config example and env var docs (matching codex/opencode style)
- [x] Verify `jq` and `gemini` are available (exit with error messages if not)
- [x] Parse `-p` prompt from args, ignore unknown flags gracefully (same `while [[ $# -gt 0 ]]; do case "$1" in` pattern)
- [x] Error if no prompt provided
- [x] Support `GEMINI_MODEL` env var (passed as `--model` flag when set)
- [x] Support `GEMINI_VERBOSE` env var (0/1, with invalid-value warning)
- [x] Detect review prompts (`<<<RALPHEX:REVIEW_DONE>>>` in prompt) and prepend review adapter text (same pattern as opencode)
- [x] Create temp dir + stderr capture file; trap cleanup on EXIT; trap SIGTERM forward to gemini child PID
- [x] Run `gemini` in background, capturing stderr, piping stdout through named FIFO
- [x] Translate each output line to `content_block_delta` event via `jq -cn --arg text "$line" '...'`
- [x] After main loop: wait for gemini exit code, emit stderr lines as `content_block_delta` events, emit fallback result event, preserve gemini exit code
- [x] Make script executable (`chmod +x`)

### Task 2: Create gemini-as-claude_test.sh

**Files:**
- Create: `scripts/gemini-as-claude/gemini-as-claude_test.sh`

- [x] Create mock `gemini` script (reads `MOCK_STDOUT_FILE`, `MOCK_STDERR_FILE`, `MOCK_EXIT_CODE`, writes args to file)
- [x] Test: signal passthrough (`<<<RALPHEX:ALL_TASKS_DONE>>>` appears in `content_block_delta` output)
- [x] Test: REVIEW_DONE signal passthrough
- [x] Test: FAILED signal passthrough
- [x] Test: exit code 0 on success
- [x] Test: exit code 1 preserved on failure
- [x] Test: non-standard exit code (42) preserved
- [x] Test: stderr captured and emitted as `content_block_delta` events
- [x] Test: empty stderr produces no extra events
- [x] Test: no prompt exits with error and message
- [x] Test: unknown flags ignored, output produced normally
- [x] Test: all output lines are valid JSON
- [x] Test: large prompt (5000+ chars) handled
- [x] Test: GEMINI_MODEL passed as `--model` flag
- [x] Test: `--model` omitted when GEMINI_MODEL is empty
- [x] Test: GEMINI_VERBOSE=1 behavior (if applicable)
- [x] Test: GEMINI_VERBOSE invalid value produces warning
- [x] Test: review adapter prepended for review prompts, not for non-review prompts
- [x] Test: gemini not found exits with error
- [x] Test: fallback result event emitted when gemini exits without explicit end
- [x] Test: malformed/non-JSON input handled gracefully (plain text lines become valid JSON events)
- [x] Test: SIGTERM forwarded to gemini child process
- [x] Make test script executable; run it: `bash scripts/gemini-as-claude/gemini-as-claude_test.sh`

### Task 3: Create README.md

**Files:**
- Create: `scripts/gemini-as-claude/README.md`

- [x] Add title and description (drop-in replacement for claude in task and review phases)
- [x] Add configuration section (claude_command, claude_args =)
- [x] Add environment variables table (GEMINI_MODEL, GEMINI_VERBOSE)
- [x] Add requirements section (gemini CLI, jq)
- [x] Add testing section (`bash scripts/gemini-as-claude/gemini-as-claude_test.sh`)

### Task 4: Update docs/custom-providers.md

**Files:**
- Modify: `docs/custom-providers.md`

- [x] Add "Gemini CLI wrapper (included example)" section after the OpenCode wrapper section
- [x] Include setup config snippet, env var table, and brief how-it-works note
- [x] Note that Gemini outputs plain text (no JSONL translation needed, simpler than codex/opencode)

### Task 5: Verify acceptance criteria

- [x] Manual test: `bash scripts/gemini-as-claude/gemini-as-claude_test.sh` — all tests pass
- [x] Verify no Go tests broken: `make test`
- [x] Run linter: `make lint`
- [x] Verify both scripts are executable: `ls -la scripts/gemini-as-claude/`

### Task 6: Update documentation

- [x] Update `CLAUDE.md` if needed (add gemini-as-claude to the scripts/ structure description)
- [x] Move this plan to `docs/plans/completed/`
