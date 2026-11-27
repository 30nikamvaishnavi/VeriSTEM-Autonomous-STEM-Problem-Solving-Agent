# ADR-002: Subprocess Isolation for Python Execution

- **Status**: Accepted
- **Context**: Allowing an LLM to generate arbitrary Python expressions inside the FastAPI worker process introduces remote code execution and availability risks.
- **Decision**: Execute generated Python snippets exclusively in an isolated child process (`python -I`) after AST validation, with `resource` CPU/RAM/file caps, disabled sockets, stripped environment variables, and a hard timeout.
- **Consequences**: Untrusted code cannot access server secrets, filesystem paths, or external networks, and runaway loops terminate safely.
