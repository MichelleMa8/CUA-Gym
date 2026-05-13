<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-red" /></a>
  <a href="https://huggingface.co/datasets/cua-gym/cua-gym"><img src="https://img.shields.io/badge/🤗%20Dataset-CUA--Gym-yellow" /></a>
  <a href="https://huggingface.co/collections/cua-gym"><img src="https://img.shields.io/badge/🤗%20Models-CUA--Gym-yellow" /></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green" /></a>
</p>

# CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use Agents

> **NeurIPS 2026** | [Paper](https://arxiv.org/abs/XXXX.XXXXX) · [Dataset](https://huggingface.co/datasets/cua-gym/cua-gym) · [Models](https://huggingface.co/collections/cua-gym)

CUA-Gym is a scalable agentic pipeline that synthesizes verifiable RLVR training data for computer-use agents (CUAs). Given a topic specification, it jointly produces task instructions, environment states, and reward functions as verified tuples — using coding agents to handle the engineering work that has previously required human experts.

## News

- **2026-xx**: Code, dataset, and models released.
- **2026-xx**: CUA-Gym accepted at NeurIPS 2026.

## Overview

Training computer-use agents with reinforcement learning requires a consistent triple of **(task instruction, executable environment, verifiable reward)**. Hand-authoring even one such triple takes hours of expert effort; CUA-Gym automates this at scale.

**The pipeline runs three coordinated agents:**

```
Topic Spec
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Generator (setup-gen agent)                                    │
│  • Constructs initial environment state (initial_setup.py)      │
│  • Constructs golden environment state  (golden_patch.py)       │
└──────────────────────────┬──────────────────────────────────────┘
                           │  files on VM only (information barrier)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Discriminator (reward-gen agent)                               │
│  • Reads task description only — cannot see Generator's code    │
│  • Independently explores the VM and writes reward.py           │
│  • Tests: reward(golden) = 1.0  AND  reward(initial) = 0.0     │
└──────────────────────────┬──────────────────────────────────────┘
                           │  pass / fail
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Orchestrator                                                   │
│  • Drives Generator ↔ Discriminator through ≤5 rounds          │
│  • Collects verified tuples → output/final/<task_id>/           │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
              LLM Majority-Vote Filter  (filter/)
                           │
                           ▼
                   Verified RLVR Tuple
         (instruction, initial_setup, golden_patch, reward)
```

**Five agreement conditions** must hold simultaneously before a tuple is accepted:

| # | Condition |
|---|-----------|
| 1 | `initial_setup.py` executes without errors on the VM |
| 2 | `golden_patch.py` executes without errors on the VM |
| 3 | `reward(golden_state) == 1.0` |
| 4 | `reward(initial_state) == 0.0` |
| 5 | No forbidden reward patterns (hardcoded values, fragile perception) |

**The CUA-Gym dataset** contains **32,112** verified RLVR tuples spanning **110 environments** (16 desktop apps + 94 synthesized mock web apps).

## Results

Models trained with GSPO on CUA-Gym:

| Model | OSWorld-Verified | WebArena |
|-------|-----------------|---------|
| CUA-Gym-A3B (Qwen3.5-35B-A3B) | **62.1%** | — |
| CUA-Gym-A17B (Qwen3.5-397B-A17B) | **70.2%** | **56.0** |

Both checkpoints set state-of-the-art among open-source CUAs at their respective scales, and transfer to the held-out WebArena benchmark without any browser-specific training.

## Installation

```bash
git clone https://github.com/cua-gym/cua-gym
cd cua-gym
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
# edit .env with your OPENAI_API_KEY, ALIYUN_* keys, etc.
source .env
```

## Quickstart

### 1. Generate tasks for a domain

Invoke the `task-gen` agent from the CUA-Gym directory via Claude Code:

```
Generate 50 LibreOffice Calc tasks covering formatting and formula operations,
ranging from easy to hard.
```

Output: `output/task_generation/<topic>.json`

### 2. Run the adversarial co-generation loop

```bash
python scripts/batch_orchestrator.py \
  --task-id calc_fmt_001 \
  output/task_generation/calc_formatting.json
```

Or process a full topic in batch:

```bash
python scripts/batch_orchestrator.py \
  output/task_generation/calc_formatting.json
```

Verified tuples land in `output/final/<task_id>/`.

### 3. Run the majority-vote filter

```bash
export OPENAI_API_KEY=sk-...
python filter/majority_vote_filter.py \
  --tasks-dir output/final \
  --votes 3 \
  --model gpt-4o \
  --write
```

### 4. Download the pre-built dataset

```bash
huggingface-cli download cua-gym/cua-gym --repo-type dataset --local-dir data/
```

## Repository Structure

```
cua-gym/
├── .claude/
│   ├── agents/
│   │   ├── orchestrator.md     # Orchestrator agent prompt
│   │   ├── setup-gen.md        # Generator agent prompt
│   │   ├── reward-gen.md       # Discriminator agent prompt
│   │   └── task-gen.md         # Task generation agent prompt
│   ├── commands/
│   │   └── create-skill.md     # /create-skill slash command
│   └── skills/                 # Per-domain skill files (SKILL.md)
│       ├── libreoffice-calc/
│       ├── libreoffice-writer/
│       ├── mock_websites/
│       └── ...
├── filter/
│   ├── majority_vote_filter.py # LLM majority-vote filter (paper §A.4)
│   ├── critic_prompt.py        # Critic system + user prompt
│   ├── run_critic_benchmark.py # Evaluate critic quality
│   └── benchmark_tasks.json    # Hand-labeled critic benchmark
├── scripts/
│   ├── batch_orchestrator.py   # Batch task processing
│   └── env_cli.py              # VM interaction CLI
├── utils/
│   ├── env.py                  # VM provisioning (Aliyun ECS)
│   ├── llm_utils.py            # LLM API wrappers with caching
│   ├── reward_judge.py         # Reward execution and scoring
│   └── logger.py               # Structured logging
├── .env.example                # Required environment variables
├── pyproject.toml              # Package metadata and dependencies
└── LICENSE                     # Apache 2.0
```

## Supported Environments

**Desktop applications (16):** LibreOffice Calc, Writer, Impress · GIMP · VLC · OpenShot · Blender · VS Code · Chrome · PDF · Penpot · Excalidraw · Overleaf CE · Grafana · Draw.io · Ubuntu OS

**Mock web applications (94):** Productivity (Notion, Airtable, Monday, Asana, Trello, Jira), Communication (Slack, Discord, Teams, Gmail, Outlook), Business (Salesforce, HubSpot, Shopify, Stripe, QuickBooks), Cloud (AWS Console, Azure, Postman, WandB, GitHub, GitLab), Healthcare (Epic Health, PACS), and more — all grounded in O\*NET occupational taxonomies and the Anthropic Economic Index.

## Adding a New Domain Skill

Domain skills give the Generator and Discriminator agents concrete API knowledge. To add a new skill:

```
/create-skill <domain-name>
```

The command researches the domain's Python API, extracts patterns from production experience, and writes `.claude/skills/<domain>/SKILL.md`.

## Reproducing Paper Results

See [`docs/reproduce.md`](docs/reproduce.md) for exact commands to reproduce all tables and figures in the paper, including hardware requirements and expected metric ranges.

## Citation

```bibtex
@inproceedings{cuagym2026,
  title     = {CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use Agents},
  author    = {Anonymous Author(s)},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```

## License

Code: [Apache 2.0](LICENSE)  
Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
