---
name: mock_websites_reward
description: "How to write reward.py scripts that verify mock web application state for CUA-Gym tasks. For reward-gen agent."
user-invocable: false
---

# Mock Websites — Reward Script Guide

This skill teaches **reward-gen** how to write `reward.py` scripts that verify task completion against mock web application state. Unlike file-based domains, there are no local artifacts — state is fetched via HTTP.

---

## 1. Reading the Session ID

The sid was written by `initial_setup.py` to `/tmp/task_web_sid`. Fail early if not found.

```python
import sys

try:
    with open('/tmp/task_web_sid') as f:
        sid = f.read().strip()
    if not sid:
        raise ValueError('sid is empty')
except Exception as e:
    print(f'CRITICAL: Cannot read sid from /tmp/task_web_sid: {e}')
    print('REWARD: 0.0')
    sys.exit(0)
```

---

## 2. Fetching State from the Mock Server

```python
import requests

BASE_URL = 'https://cua-gym-<name>.xlang.ai'  # e.g., cua-gym-slack.xlang.ai

try:
    resp = requests.get(f'{BASE_URL}/go?sid={sid}', timeout=15)
    resp.raise_for_status()
    data = resp.json()
except Exception as e:
    print(f'CRITICAL: Cannot fetch state from {BASE_URL}/go?sid={sid}: {e}')
    print('REWARD: 0.0')
    sys.exit(0)

initial_state = data.get('initial_state', {})
current_state = data.get('current_state', {})
state_diff = data.get('state_diff', {})
```

**Key insight**: `initial_state` is the pre-task snapshot. `current_state` is what the agent (or golden_patch) produced. Your reward script scores how well `current_state` matches the expected post-task state.

---

## 3. Reward Script Template (Programmatic Verification)

For tasks with clearly defined, checkable success criteria:

```python
"""
Reward Script: <task_description>
Task ID: <task_id>
Domain: mock_websites
Mock: <mock_name>
Scoring: <brief rubric>
"""
import json
import sys

import requests

# --- Read sid ---
try:
    with open('/tmp/task_web_sid') as f:
        sid = f.read().strip()
except Exception:
    print('REWARD: 0.0')
    sys.exit(0)

BASE_URL = 'https://cua-gym-<name>.xlang.ai'

# --- Fetch state ---
try:
    data = requests.get(f'{BASE_URL}/go?sid={sid}', timeout=15).json()
except Exception:
    print('REWARD: 0.0')
    sys.exit(0)

initial = data.get('initial_state', {})
current = data.get('current_state', {})

def verify_task():
    total_score = 0.0

    # Component 1: <description> (X.X points)
    try:
        # Example: check if a new message was sent in #general
        initial_msgs = initial.get('messages', {}).get('general', [])
        current_msgs = current.get('messages', {}).get('general', [])
        if len(current_msgs) > len(initial_msgs):
            new_msgs = current_msgs[len(initial_msgs):]
            if any('hello' in m.get('content', '').lower() for m in new_msgs):
                print(f'PASS: New message containing "hello" found ({0.5} pts)')
                total_score += 0.5
            else:
                print(f'FAIL: New messages exist but none contain "hello"')
        else:
            print(f'FAIL: No new messages in #general')
    except Exception as e:
        print(f'ERROR: Component 1 — {e}')

    # Component 2: <description> (X.X points)
    try:
        # ... another check ...
        pass
    except Exception as e:
        print(f'ERROR: Component 2 — {e}')

    final_score = min(total_score, 1.0)
    print(f'\nScore: {total_score}/1.0')
    print(f'REWARD: {final_score}')
    return final_score

verify_task()
```

---

## 4. LLM-as-Judge (Constrained Usage)

For task components where success cannot be verified programmatically (e.g., semantic equivalence like "SD-USA" ≈ "San Diego", or subjective quality like "write a professional reply"), use the pre-deployed LLM judge helper.

### Budget Rule

- **≥ 60%** of total score MUST come from programmatic checks (§3 pattern)
- **≤ 40%** MAY use LLM judge via `call_llm_judge()`
- Every LLM judge call MUST have a `# JUSTIFICATION:` comment

### Usage Pattern

```python
import json
import sys

import requests

# --- Read sid ---
try:
    with open('/tmp/task_web_sid') as f:
        sid = f.read().strip()
except Exception:
    print('REWARD: 0.0')
    sys.exit(0)

BASE_URL = 'https://cua-gym-<name>.xlang.ai'

# --- Fetch state ---
try:
    data = requests.get(f'{BASE_URL}/go?sid={sid}', timeout=15).json()
except Exception:
    print('REWARD: 0.0')
    sys.exit(0)

initial = data.get('initial_state', {})
current = data.get('current_state', {})

# --- Import LLM judge helper (pre-deployed by orchestrator to /tmp/) ---
sys.path.insert(0, '/tmp')
from reward_judge import call_llm_judge

def verify_task():
    total_score = 0.0

    # Component 1 (0.4 pts) — PROGRAMMATIC: message count check
    try:
        initial_msgs = initial.get('messages', {}).get('general', [])
        current_msgs = current.get('messages', {}).get('general', [])
        if len(current_msgs) > len(initial_msgs):
            print(f'PASS: New message(s) found in #general (0.4 pts)')
            total_score += 0.4
        else:
            print(f'FAIL: No new messages in #general')
    except Exception as e:
        print(f'ERROR: Component 1 — {e}')

    # Component 2 (0.3 pts) — PROGRAMMATIC: sender is correct user
    try:
        initial_msgs = initial.get('messages', {}).get('general', [])
        current_msgs = current.get('messages', {}).get('general', [])
        new_msgs = current_msgs[len(initial_msgs):]
        if new_msgs and new_msgs[-1].get('sender') == 'user_1':
            print(f'PASS: Message sent by correct user (0.3 pts)')
            total_score += 0.3
        else:
            print(f'FAIL: Message not sent by user_1')
    except Exception as e:
        print(f'ERROR: Component 2 — {e}')

    # Component 3 (0.3 pts) — LLM JUDGE: message content quality
    # JUSTIFICATION: Task asks agent to "write a professional greeting".
    # No single correct phrasing exists — semantic evaluation needed.
    try:
        initial_msgs = initial.get('messages', {}).get('general', [])
        current_msgs = current.get('messages', {}).get('general', [])
        new_msgs = current_msgs[len(initial_msgs):]
        if new_msgs:
            llm_score = call_llm_judge(
                task_instruction='Write a professional greeting in #general',
                success_criteria='The message is a professional, appropriate greeting',
                state_excerpt=json.dumps(new_msgs[-1]),
            )
            total_score += 0.3 * llm_score
            print(f'LLM JUDGE: Component 3 — {0.3 * llm_score:.2f} pts')
        else:
            print(f'FAIL: Component 3 — no message to evaluate')
    except Exception as e:
        print(f'ERROR: Component 3 — {e}')

    final_score = min(total_score, 1.0)
    print(f'\nScore: {total_score}/1.0')
    print(f'REWARD: {final_score}')
    return final_score

verify_task()
```

### FORBIDDEN — Do NOT use raw OpenAI SDK

```python
# FORBIDDEN — bypasses locked-down model/temperature/system_prompt
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(model='gpt-4o-mini', ...)
```

Always use `from reward_judge import call_llm_judge`. The helper is deployed to `/tmp/reward_judge.py` by the orchestrator with fixed parameters you cannot override.

---

## 5. Composite Scoring (Web + Other Checks)

For hybrid tasks that involve both web state and local file changes:

```python
# Weight allocation
WEB_WEIGHT = 0.7
FILE_WEIGHT = 0.3

# Web state score
web_score = verify_web_state()  # uses pattern from §3

# File score
file_score = verify_local_file()  # standard file verification

final = web_score * WEB_WEIGHT + file_score * FILE_WEIGHT
print(f'REWARD: {min(final, 1.0)}')
```

---

## 6. Multi-Mock Reward

When the task involves multiple mocks, fetch `/go` from each and score the combined state:

```python
mocks = {
    'slack': 'https://cua-gym-slack.xlang.ai',
    'notion': 'https://cua-gym-notion.xlang.ai',
}

states = {}
for name, url in mocks.items():
    try:
        data = requests.get(f'{url}/go?sid={sid}', timeout=15).json()
        states[name] = data
    except Exception as e:
        print(f'ERROR: Cannot fetch {name}: {e}')
        print('REWARD: 0.0')
        sys.exit(0)

# Score each mock's state independently, then combine
slack_score = verify_slack(states['slack'])
notion_score = verify_notion(states['notion'])
final = slack_score * 0.5 + notion_score * 0.5
print(f'REWARD: {min(final, 1.0)}')
```

---

## 7. Writing success_criteria (for LLM Judge)

Good success criteria are:
- **Specific**: "A message with content containing 'quarterly report' was posted in #marketing"
- **Observable**: Reference state fields the judge can check in the JSON
- **Negative-aware**: "No messages were deleted from #general" (if relevant)

Bad success criteria:
- **Vague**: "The task was completed" — gives the LLM nothing to verify
- **Implementation-focused**: "The POST request succeeded" — describes how, not what
- **Uncheckable**: "The user felt satisfied" — not observable in state

---

## 8. State Schemas

To understand the state structure for each mock, read the schema file:

```
Read: .claude/skills/mock_websites/schemas/<mock_name>.md
```

The schema documents all required top-level keys, entity shapes, and which fields change for specific user actions. Use it to design accurate verification checks.

**Available schemas:** asana_mock, aws_console_mock, discord_mock, docusign_mock, github_mock, gitlab_mock, gmail_mock, jira_mock, linkedin_mock, notion_mock, reddit_mock, salesforce_mock, slack_mock, trello_mock, twitter_mock, youtube_mock.

For mocks without a schema file, fetch default state via `GET /go?sid=nonexistent` to discover the structure.

---

## 9. Information Barrier Reminder

As the reward-gen agent (discriminator), you MUST NOT:
- Read `initial_setup.py` or `golden_patch.py`
- Derive verification logic from setup-gen's implementation
- Use the initial_state from `/go` to "cheat" (e.g., hardcoding expected values from the golden state)

Your reward script must be derivable purely from `task_config.json` (task description + success criteria). Explore the VMs to understand what changed, but design scoring based on task requirements.

---

## 10. REWARD: X.X Format

The **last printed line** of reward.py MUST be `REWARD: X.X` where X.X is a float between 0.0 and 1.0.

```python
# Always end with this pattern
print(f'REWARD: {final_score}')
```

The pipeline parses this line to extract the reward value. Any other format will cause evaluation failure.

---

## 11. Error Handling Checklist

Your reward.py MUST handle these failure modes gracefully (print `REWARD: 0.0` and exit):

| Failure | Cause | Handling |
|---------|-------|----------|
| `/tmp/task_web_sid` not found | initial_setup.py didn't run | `REWARD: 0.0` |
| sid is empty | File exists but empty | `REWARD: 0.0` |
| `/go?sid=<sid>` returns 4xx/5xx | Server error or wrong URL | `REWARD: 0.0` |
| `/go?sid=<sid>` times out | Server unreachable | `REWARD: 0.0` |
| `current_state` is None | No state was injected for this sid | `REWARD: 0.0` |
| `initial_state == current_state` | No changes made (agent did nothing) | `REWARD: 0.0` |
| LLM judge API fails | OpenAI key missing or rate limited | `REWARD: 0.0` |
| JSON parse error | Malformed response | `REWARD: 0.0` |

**Never let reward.py crash without printing `REWARD: X.X`.** Wrap everything in try/except.
