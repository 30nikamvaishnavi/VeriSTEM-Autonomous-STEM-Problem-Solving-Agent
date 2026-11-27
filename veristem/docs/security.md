# VeriSTEM Security Architecture

## 1. Isolated Python Sandbox (`app/tools/python_sandbox.py`)
- **AST Pre-Execution Filtering**: Rejects `eval`, `exec`, `compile`, `open`, `__import__`, `getattr`/`setattr`, dunder attribute access (`__*`), and any module outside `{math, cmath, decimal, fractions, statistics, json, numpy, scipy, sympy}`.
- **Process Isolation**: Executes inside `python -I` (isolated mode, ignores user site-packages and `PYTHON*` env vars) inside an ephemeral temporary directory deleted immediately after completion.
- **Resource Limits**: Enforces `RLIMIT_CPU`, `RLIMIT_AS` (memory cap), `RLIMIT_FSIZE`, and a hard `subprocess.run(timeout=...)` wall-clock timeout.
- **Network Disablement**: Replaces `socket.socket` inside the child runner so outbound network connections fail immediately.

## 2. API & Secret Hygiene
- `GEMINI_API_KEY` is strictly server-side; never exposed to browser clients.
- `VERISTEM_API_KEY` verification uses constant-time `hmac.compare_digest`.
- Structured logs pass through a regex secret redactor before emission.
- Sliding-window rate limiting and `max_problem_chars` prevent resource exhaustion.
