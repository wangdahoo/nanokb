# AGENTS.md

Guidance for OpenCode agents working in this repo. This is an OpenCode plugin
port of the Claude Code `golden-hoop-spell` plugin — pure TypeScript, loaded by
OpenCode as a plugin (no build step, no Python runtime dep).

## Critical Rules

- **Language policy** (from `AGENTS.md`, applies to all agents/subagents): Chinese for human-readable output — conversation, docs, commit messages, TODO/FIXME, task/plan descriptions.English for code identifiers, log/error strings, and LLM-facing prompts. When spawning subagents, include the instruction: `使用中文回复和撰写所有文档/commit message。代码标识符、日志、错误信息用英文。`