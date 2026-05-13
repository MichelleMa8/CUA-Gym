<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-CUA--Gym-yellow)](https://huggingface.co/datasets/cua-gym/cua-gym)
[![Models](https://img.shields.io/badge/🤗%20Models-CUA--Gym-yellow)](https://huggingface.co/collections/cua-gym)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

</div>

---

<p align="center">
<a href="https://arxiv.org/abs/XXXX.XXXXX"><b>Paper</b></a> |
<a href="https://huggingface.co/datasets/cua-gym/cua-gym"><b>Dataset</b></a> |
<a href="https://huggingface.co/collections/cua-gym"><b>Models</b></a>
</p>

# CUA-Gym

CUA-Gym is a scalable pipeline for synthesizing verifiable RLVR training data for computer-use agents (CUAs). Given a topic, it jointly produces task instructions, environment states, and reward functions as verified triples — using coding agents to handle the engineering work previously requiring human experts.

## News

- [2026-05-13] We released the full pipeline, dataset and models of CUA-Gym 🔥🔥🔥

## About

Training computer-use agents with reinforcement learning requires a consistent triple of **(task instruction, executable environment, verifiable reward)**. Hand-authoring even one such triple takes hours; CUA-Gym automates this at scale.

<p align="center">
  <img src="figures/main_figure.png" alt="CUA-Gym pipeline" width="100%"/>
</p>

**Pipeline.** Three coordinated agents run per task:

- **Generator** (`setup-gen`): constructs the initial and golden environment states (`initial_setup.py`, `golden_patch.py`)
- **Discriminator** (`reward-gen`): writes `reward.py` from the task description alone, without access to Generator's code (information barrier)
- **Orchestrator**: drives the two through iterative rounds until `reward(golden)=1.0` and `reward(initial)=0.0` both hold under execution

**Filtering.** Verified tuples pass through an LLM majority-vote filter (`filter/majority_vote_filter.py`) that rejects tasks where the reward is fragile, ambiguous, or inconsistent. Teacher rollouts provide a second filter stage.

**Environments.** CUA-Gym covers 110 environments: 16 desktop applications and 94 synthesized mock web applications grounded in real-world software-use distributions.

**Dataset.** The resulting [CUA-Gym dataset](https://huggingface.co/datasets/cua-gym/cua-gym) contains **32,112** verified RLVR training tuples.

**Comparison with existing CUA RLVR datasets:**

| Dataset | Platform | Data size | Env. size | Reward | Open |
|---------|----------|----------:|----------:|--------|:----:|
| GUI-Genesis | Mobile | 969 | 1 | Programmatic | No |
| WebArena-Infinity | Web | 1,260 | 10 | Programmatic | Yes |
| InfiniteWeb | Web | 600 | — | Programmatic | No★ |
| UltraCUA | Desktop | 17,000 | 9 | Programmatic | No★ |
| Gym-Anything | Desktop | 7,277 | 193 | VLM | Yes |
| **CUA-Gym** | **Desktop + Web** | **32,122** | **110** | **Programmatic** | **Yes** |

★ partial release.

## Results

| Model | OSWorld-Verified | WebArena |
|-------|:----------------:|:--------:|
| *Claude Sonnet 4.6* | 72.9 | 65.6 |
| *Claude Opus 4.7* | 78.0 | — |
| *GPT-5.5* | 78.7 | — |
| *EvoCUA-8B* | 46.1 | — |
| *EvoCUA-32B* | 56.7 | — |
| *Kimi-K2.6* | 73.1 | — |
| Qwen3.5-35B-A3B (base) | 54.5 | 40.8 |
| Qwen3.5-397B-A17B (base) | 62.2 | 54.0 |
| **CUA-Gym-A3B** | **62.1** | **44.5** |
| **CUA-Gym-A17B** | **70.2** | **56.0** |

Both models set state-of-the-art among open-source CUAs at their respective scales. CUA-Gym-A3B matches the much larger A17B base at ~10× fewer active parameters.

## Getting Started

**Install**

```bash
git clone https://github.com/BowenBryanWang/CUA-Gym
cd CUA-Gym
pip install -e ".[dev]"
cp .env.example .env  # fill in OPENAI_API_KEY and ALIYUN_* credentials
```

**Generate tasks for a domain**

Invoke the `task-gen` agent from the CUA-Gym directory in Claude Code:

```
Generate 50 LibreOffice Calc tasks covering formatting and formula operations.
```

Output: `output/task_generation/<topic>.json`

**Run the adversarial co-generation loop**

```bash
python scripts/batch_orchestrator.py output/task_generation/calc_formatting.json
```

Verified tuples land in `output/final/<task_id>/`.

**Run the majority-vote filter**

```bash
export OPENAI_API_KEY=sk-...
python filter/majority_vote_filter.py \
  --tasks-dir output/final \
  --votes 3 \
  --model gpt-4o \
  --write
```

**Download the pre-built dataset**

```bash
huggingface-cli download cua-gym/cua-gym --repo-type dataset --local-dir data/
```

## Supported Environments

**Desktop (16):** LibreOffice Calc · Writer · Impress · GIMP · VLC · OpenShot · Blender · VS Code · Chrome · PDF · Penpot · Excalidraw · Overleaf CE · Grafana · Draw.io · Ubuntu OS

**Mock Web Apps (94):** Notion · Airtable · Monday · Asana · Jira · Trello · Slack · Discord · Teams · Gmail · Outlook · Salesforce · HubSpot · Shopify · Stripe · QuickBooks · AWS Console · Azure · Postman · WandB · GitHub · GitLab · Epic Health · PACS · and more — grounded in O\*NET occupational taxonomies and the Anthropic Economic Index

## CUA-Gym-Hub

[CUA-Gym-Hub](https://github.com/BowenBryanWang/CUA-Gym-Hub) (included as `hub/`) is a suite of **94 self-contained mock web applications** used as RL training environments. Each mock implements a realistic UI backed by a unified HTTP API for state injection, inspection, reset, and diffing — making it fully programmatic and reproducible.

```bash
# Run a mock app locally
cd hub/notion_mock && npm install && npm run dev
# State API available at http://localhost:5173/api/state
```

Mock apps span 7 categories: Communication & Social · Productivity · Development & Cloud · E-commerce & Travel · Finance & Enterprise · Analytics & Marketing · and more. See [hub/README.md](hub/README.md) for the full list and API reference.

## Repository Structure

```
CUA-Gym/
├── hub/                 # CUA-Gym-Hub submodule — 94 mock web apps
├── .claude/
│   ├── agents/          # orchestrator, setup-gen, reward-gen, task-gen prompts
│   ├── commands/        # /create-skill slash command
│   └── skills/          # per-domain skill files (openpyxl, PIL, Flask APIs, etc.)
├── filter/
│   ├── majority_vote_filter.py   # LLM majority-vote filter
│   ├── critic_prompt.py          # critic system + user prompt
│   ├── run_critic_benchmark.py   # evaluate critic quality
│   └── benchmark_tasks.json      # hand-labeled critic benchmark
├── scripts/
│   ├── batch_orchestrator.py     # batch task processing
│   └── env_cli.py                # VM interaction CLI
├── utils/
│   ├── env.py           # VM provisioning (Aliyun ECS)
│   ├── llm_utils.py     # LLM API wrappers with caching
│   ├── reward_judge.py  # reward execution and scoring
│   └── logger.py        # structured logging
├── .env.example
├── pyproject.toml
└── LICENSE
```

## Citation

```bibtex
@article{cuagym2026,
  title   = {CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use Agents},
  author  = {},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Code: [Apache 2.0](LICENSE) · Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
