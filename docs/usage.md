# Using Zorlo

## Serve a model

```
zorlo serve models\qwen3-30b-a3b.zorlo --port 8080
```

A pack comes from `zorlo pack <model.gguf> <out_dir>`, or ready-made from Hugging Face with
`zorlo pull <owner/name>[@revision] [out_dir] [--lowbit]`. Pull downloads every file in the repo root except
`experts.lo.bin` (only with `--lowbit`) and `hot.bin`. A cut download resumes from `<file>.part`; each file is
checked against the SHA-256 (LFS) or git SHA-1 that the Hub lists before it is renamed into place. `HF_TOKEN`
is sent for private repos, only to huggingface.co. A pack repo holds `manifest.json`, `core.gguf`,
`experts.bin`, optionally `experts.lo.bin`, plus the model's LICENSE. Don't upload `kv/`, `hot.bin`,
`imatrix.bin` or the experimental `experts.lo.*.bin` copies. A pack without `experts.lo.bin` still loads;
only `--lowbit` / `--mode fast` then need it.

Pulling again updates a pack. `<out_dir>/.zorlo-pull.json` records each file's checksum; a file whose checksum
changed on the Hub is downloaded again, and the old copy stays until the new one is verified. Files without a
record (an older pull, or `zorlo pack`) are checksummed once, then recorded.

### `zorlo start` and the model catalog

`zorlo start` (what "Start Zorlo.bat" runs) lists the models, downloads or updates the chosen one into
`<zorlo folder>\models\<id>`, serves it and opens the chat page once `/health` answers. It remembers the last
model (`models\last.txt`) and skips the menu when there is only one model and it is ready.

The list comes from `models.json` in the Hugging Face dataset `Haidershahi1/zorlo-catalog`, so a new model
reaches every Zorlo already installed without a new release. If that can't be read, the list built into
`src/cli/start.cpp` is used. The source of truth is [release/models.json](../release/models.json); format
`zorlo-catalog-1`, one entry per model:

| field | meaning |
|---|---|
| `id` | folder name under `models\` (letters, digits, `-_.`); never change it for a published model |
| `name`, `about` | shown in the menu |
| `repo` | Hugging Face repo with the pack |
| `download_gb`, `min_ram_gb` | shown in the menu |
| `recommended` | the default choice for someone with no model yet |
| `min_zorlo` | lowest Zorlo version that can run it (e.g. a new architecture); older versions list it under "needs a newer Zorlo" |

### The agent app (`ui/`, `src/cli/agent.cpp`)

`zorlo serve` opens a coding agent at `/` (the plain chat page stays at `/simple`). The browser runs the loop:
the model (OpenAI API with `tools`) asks for tool calls, the page sends each one to `POST /agent/tool`, and the
server runs it, so permissions are enforced there:

| tool | group | what it does |
|---|---|---|
| `list_dir`, `read_file`, `preview_url` | read | folders, numbered file lines, the address of a page in the project |
| `write_file`, `edit_file` | write | whole files, or one exact replacement |
| `run_command` | command | Windows PowerShell in the project folder, output + exit code, process tree stopped at the timeout (120 s, max 1800) |
| `web_search`, `fetch_url` | web | DuckDuckGo (no key) or Brave Search (key in Settings); page text with scripts and tags removed |
| `load_skill` | skills | `<skills folder>\<name>\SKILL.md` (front matter `name:` / `description:`), always allowed |
| `mcp__<server>__<tool>` | mcp | MCP connectors over stdio, configured like Claude Desktop (`mcpServers`: command, args, env) |

Also `search_files` (text or regex across the project, `glob` filter; read group), `list_dir` with `depth` up to 4,
and `todo_write`, which the page handles itself (the Tasks panel). The app has three modes per chat: Chat (no
tools), Plan (read-only tools, ends with a plan and a "Run this plan" button) and Agent. It also has: edit any
sent message and resend from there, regenerate, copy; attachments by button, drag and drop or paste (text and
code go into the message, other files are saved into `<project>\uploads` via `POST /agent/upload`); `@path`
mentions that attach project files; slash commands (`/new /plan /agent /chat /undo /clear /files /preview
/export /settings /help`); a context meter; Markdown with tables and code highlighting; and a side panel with
Tasks, Files (`GET /agent/files`, viewer via `GET /agent/file`), Changes (every write/edit with its diff and
Undo) and a live Preview (iframe, phone/desktop width) that reloads when the agent changes a page.
Every `write_file`/`edit_file` keeps the previous version under `%APPDATA%\Zorlo\backups`; `POST /agent/undo
{undo_id}` restores it (or deletes a file the agent created). The first of `ZORLO.md`, `AGENTS.md`, `CLAUDE.md`
in the project folder is added to the prompt as project rules. A tool call the model writes with invalid JSON is
repaired when possible, otherwise the model is asked to send it again (twice at most).

Settings live in `%APPDATA%\Zorlo\settings.json` (`ZORLO_SETTINGS` overrides): project folder (any folder, picked
with the Windows folder dialog), Full permission, or per group Allow / Ask me / Block. Block always wins; paths
outside the project folder always ask unless Full permission is on. Websites the agent writes open at
`/preview/<path>`. `ZORLO_MCP_DEBUG=1` prints what connectors send.

Safety: each launch makes a random token that only the served page contains; `/agent/*` needs it, a
`Host` of 127.0.0.1/localhost (stops DNS rebinding) and no foreign `Origin`. The agent is off when serving on
another address. The UI is plain files in `ui/` next to `zorlo.exe` (`ZORLO_UI_DIR` overrides).

Tested on the dev laptop: a request to create `hello.html` and give the preview link took about 3.5 minutes
(write_file after approval, then preview_url, then the answer); every tool, the approval rules, the token checks
and an MCP connector were exercised directly.

To add a model: upload its pack to a new repo, add an entry to `release/models.json`, upload that file to the
catalog dataset, and copy the entry into `BUILTIN_CATALOG` for the next release. To update a model, upload the
changed files to its repo: every user gets them the next time they start it. Testing: `ZORLO_CATALOG` (https URL
or local file), `ZORLO_MODELS_DIR`, `ZORLO_NO_BROWSER=1`.

Then point any OpenAI client at `http://127.0.0.1:8080/v1` (model id `qwen3-30b-a3b.zorlo`, any API key):
`/v1/chat/completions` (streaming or not; `temperature`, `top_p`, `top_k`, `seed`, `max_tokens`, `stop`,
`tools`, `tool_choice`, `chat_template_kwargs`),
`/v1/models`, `/health`. Defaults follow Qwen3's recommended sampling (temperature 0.6, top_p 0.95, top_k 20);
`temperature: 0` is greedy and matches llama.cpp exactly.

The expert cache sizes itself from free RAM (`--cache-mb 0`, the default), leaving 512 MB for the OS, and
re-checks every 16 tokens: if another app takes memory the cache shrinks and gives it back, and grows again
when memory frees up. `--cache-mb N` pins it to a fixed size instead.


## Anthropic Messages API

`zorlo serve` also answers the Anthropic Messages API: `POST /v1/messages`, streaming or not, and
`POST /v1/messages/count_tokens`. A request is mapped onto the same chat path as `/v1/chat/completions`,
the way llama-server does it:

- `system` (string or text blocks), text, `tool_use`, `tool_result` and `thinking` blocks, `tools`
  (`input_schema`), `tool_choice`, `stop_sequences`, `max_tokens`, `temperature`/`top_p`/`top_k`, and Zorlo's
  `mode`.
- Replies come back as `thinking`, `text` and `tool_use` blocks with `stop_reason` `end_turn` / `tool_use` /
  `max_tokens`, or as the standard SSE events.
- The per-request hash in Claude Code's `x-anthropic-billing-header` is blanked, so the prompt start stays
  the same and its saved KV is reused.
- While a long prompt is read, or a request waits for a slot, the stream sends `ping` events. Claude Code
  drops a stream that stays silent for 5 minutes, and its huge prompt takes ~10 minutes cold.

Use it from Claude Code:

```
zorlo serve models\qwen3-30b-a3b.zorlo --port 8080 -c 20480
```
```
set ANTHROPIC_BASE_URL=http://127.0.0.1:8080
set ANTHROPIC_AUTH_TOKEN=zorlo
set ANTHROPIC_MODEL=zorlo
set ANTHROPIC_SMALL_FAST_MODEL=zorlo
set API_TIMEOUT_MS=3600000
claude
```

Claude Code's system prompt and tool list are ~12-15k tokens, so give the server `-c 20480` or more
(~96 KB of RAM per context token on Qwen3-30B-A3B). The first request reads it cold (~10 min on the reference PC);
after that it comes from the saved KV (Phase 7) in about a second, also across restarts.

Checked against llama-server's behaviour with an Anthropic client (Python, greedy): a plain reply, a tool call
(the same thinking and `get_weather` call as llama-server's reference when evaluated cold), a `tool_result`
round trip, the same call streamed (`message_start`, block start/delta/stop, `message_delta`,
`message_stop`), and `count_tokens` (213, the reference prompt's length).

End to end with Claude Code 2.1.309 (`claude -p "Create a file named hello.txt containing the word hi, then
reply with the single word done." --permission-mode acceptEdits`, server `-c 20480 --cache-mb 700 --mode fast`,
3.8 GB RAM free): `hello.txt` contains `hi`, and the result is `done` in 2 turns.

| turn | prompt | read from saved KV | prefill | generated |
|---|---|---|---|---|
| 1 (thinks, then calls Write) | 14,381 tokens | 9,574 from disk (0.56 s) | 357 s | 502 tokens at 0.88 tok/s |
| 2 (after the tool result) | 17,409 tokens | 14,883 in RAM | 239 s | 136 tokens at 0.67 tok/s |

It works, but it is slow on the reference PC: 23 minutes for the task. At a 15-17k context with a 700 MB expert cache,
each token reads a lot from the SSD, and Qwen3 thinks for hundreds of tokens before acting. With more free RAM
(a bigger expert cache) both prefill and decode get much faster.

## Measurement tools (no compiler needed)

Measure your SSD. This writes a temporary test file (default 4 GB) and deletes it afterwards:

```
python tools/diskbench.py .bench
```

Simulate the expert cache on a routing trace (`.npy`, int array `[tokens, layers, top_k]`):

```
python tools/cachesim.py trace.npy --budget-gb 1.5 3 5 --expert-mb 2.6
```

Tests:

```
python -m unittest discover -s tests
```
