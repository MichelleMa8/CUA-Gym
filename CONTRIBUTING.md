# Contributing to CUA-Gym

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/cua-gym/cua-gym
cd cua-gym
pip install -e ".[dev]"
pre-commit install
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check .
ruff format .
```

Pre-commit hooks run these automatically on each commit.

## Adding a New Domain Skill

The highest-impact contribution is adding skill files for new application domains. A skill file teaches the Generator and Discriminator agents how to programmatically create, manipulate, and verify domain-specific file/state formats.

1. Create `.claude/skills/<domain>/SKILL.md` — follow the structure of an existing skill (e.g., `libreoffice-calc/SKILL.md`)
2. The skill should cover: (a) how to set up initial state, (b) how to apply changes, (c) how to verify state in `reward.py`
3. Use the `/create-skill <domain>` slash command in Claude Code as a starting point

## Pull Requests

- Fork the repo and create a branch from `main`
- Keep PRs focused — one logical change per PR
- For new domain skills, include at least one example task that the skill enables
- For filter/pipeline changes, run `python filter/run_critic_benchmark.py` and include the before/after accuracy numbers

## Reporting Issues

Please use the GitHub issue templates:
- **Bug report**: unexpected behavior with reproduction steps
- **Feature request**: new domain, new capability, or pipeline improvement

## Questions

Open a GitHub Discussion for questions about the pipeline, the dataset, or reproducing paper results.
