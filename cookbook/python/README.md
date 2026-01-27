# GitHub Copilot SDK Cookbook — Python

This folder hosts short, practical recipes for using the GitHub Copilot SDK with Python. Each recipe is concise, copy‑pasteable, and points to fuller examples and tests.

## Recipes

- [Error Handling](error-handling.md): Handle errors gracefully including connection failures, timeouts, and cleanup.
- [Multiple Sessions](multiple-sessions.md): Manage multiple independent conversations simultaneously.
- [Managing Local Files](managing-local-files.md): Organize files by metadata using AI-powered grouping strategies.
- [PR Visualization](pr-visualization.md): Generate interactive PR age charts using GitHub MCP Server.
- [Persisting Sessions](persisting-sessions.md): Save and resume sessions across restarts.

## PoR interception (minimal)

Examples in this folder include a lightweight Proof-of-Resonance (PoR) interception layer. The SDK treats the explicit abstention control signal `[[POR_ABSTAIN]]` as a non-error outcome, so you can log it and continue gracefully.

## Contributing

Add a new recipe by creating a markdown file in this folder and linking it above. Follow repository guidance in [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Status

This README is a scaffold; recipe files are placeholders until populated.
