#!/usr/bin/env python3
"""
Collect trajectories using Holo-3.1-35B-A3B on CUA-Gym task bundles.

Usage:
    # Run tasks selected by select_tasks.py
    python scripts/run_trajectories.py --run-dir data/run-20260620_185032

    # Override vLLM endpoint (env var or CLI flag)
    python scripts/run_trajectories.py --run-dir ... --model-url http://nlpgpu06:8000
    HOLO_BASE_URL=http://nlpgpu06:8000/v1 python scripts/run_trajectories.py --run-dir ...

    # Override tasks source directory
    python scripts/run_trajectories.py --run-dir ... --tasks-dir /path/to/cua_gym_tasks

Output layout (inside --run-dir):
    trajectory/
      {app}/
        {task_id}/
          images/
            0000.png
            0001.png
            ...
          actions.json
          task_summary.json   # {"trajectory_id": ..., "reward": ...}
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.env import Env

DEFAULT_TASKS_DIR = Path(
    "/lcars/home/q/qianranm/research/GUI/CUA-Gym/data/cua_gym_all/cua_gym_tasks"
)

HOLO_BASE_URL = os.getenv("HOLO_BASE_URL", "http://localhost:8000/v1")
HOLO_MODEL = os.getenv("HOLO_MODEL", "holo-3.1")
MAX_STEPS = int(os.getenv("MAX_STEPS", "15"))


# ---------------------------------------------------------------------------
# App normalization
# ---------------------------------------------------------------------------

def canonical_app(app_type: str | None) -> str:
    if not app_type:
        return "unknown"
    if "," in app_type:
        return "multi_apps"
    return app_type.strip()


# ---------------------------------------------------------------------------
# Holo interaction
# ---------------------------------------------------------------------------

def call_holo(instruction: str, screenshot_b64: str, history: list) -> dict:
    messages = history + [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"Task: {instruction}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
        ],
    }]
    resp = requests.post(
        f"{HOLO_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('HOLO_API_KEY', 'token')}"},
        json={"model": HOLO_MODEL, "messages": messages, "max_tokens": 512},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


def execute_action(env: Env, action_text: str):
    """Parse Holo's action output and execute it on the VM.

    TODO: implement based on Holo-3.1-35B-A3B's action schema.
    See https://huggingface.co/Hcompany/Holo-3.1-35B-A3B for the output format.

    Examples:
        # If Holo outputs pyautogui-style Python code:
        # env.run_python(action_text)

        # If Holo outputs a shell command:
        # env.execute(action_text)
    """
    raise NotImplementedError(
        "execute_action() must be implemented based on Holo's action schema. "
        "Check the model card at https://huggingface.co/Hcompany/Holo-3.1-35B-A3B"
    )


# ---------------------------------------------------------------------------
# Task setup / scoring
# ---------------------------------------------------------------------------

def setup_env(env: Env, task_dir: Path):
    """Run the task's initial_setup script on the VM."""
    setup_py = task_dir / "initial_setup.py"
    if setup_py.exists():
        env.run_python(setup_py.read_text())
        return

    for ext in ["sh", "xlsx", "docx", "pptx"]:
        setup_file = task_dir / f"initial_setup.{ext}"
        if setup_file.exists():
            env.upload(str(setup_file), f"/home/user/initial_setup.{ext}")
            if ext == "sh":
                env.execute(f"bash /home/user/initial_setup.{ext}")
            return

    raise FileNotFoundError(f"No initial_setup file found in {task_dir}")


def score_env(env: Env, task_dir: Path) -> float:
    """Upload reward_judge.py and run reward.py on the VM, return the score."""
    repo_root = Path(__file__).parent.parent
    env.upload(str(repo_root / "utils" / "reward_judge.py"), "/tmp/reward_judge.py")
    reward_code = (task_dir / "reward.py").read_text()
    result = env.run_python(reward_code)
    try:
        return float(result.get("output", "0").strip())
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_trajectory(
    task_id: str,
    app: str,
    actions: list[dict],
    screenshots_b64: list[str],
    reward: float,
    traj_root: Path,
) -> None:
    """Write images/, actions.json, task_summary.json under traj_root/app/task_id/."""
    task_out = traj_root / app / task_id

    # images/
    images_dir = task_out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for i, b64 in enumerate(screenshots_b64):
        img_bytes = base64.b64decode(b64)
        (images_dir / f"{i:04d}.png").write_bytes(img_bytes)

    # actions.json
    (task_out / "actions.json").write_text(
        json.dumps(actions, ensure_ascii=False, indent=2)
    )

    # task_summary.json
    (task_out / "task_summary.json").write_text(
        json.dumps({"trajectory_id": task_id, "reward": reward}, indent=2)
    )


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_one_task(task_dir: Path, traj_root: Path) -> float:
    """Run a single task and persist trajectory to disk. Returns the reward score."""
    task_json = json.loads((task_dir / "task.json").read_text())
    instruction = task_json.get("instruction", task_json.get("task_instruction", ""))
    task_id = task_dir.name
    app = canonical_app(task_json.get("app_type"))

    # Skip if already done
    summary_path = traj_root / app / task_id / "task_summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        print(f"  [SKIP] {task_id}  (reward={existing.get('reward')})")
        return existing.get("reward", 0.0)

    env_config_path = f"/tmp/{task_id}_env.json"
    actions: list[dict] = []
    screenshots_b64: list[str] = []
    reward = 0.0

    env = Env.create(task_id=task_id)
    try:
        env.save_config(env_config_path)
        setup_env(env, task_dir)

        history = []
        for step in range(MAX_STEPS):
            screenshot_bytes = env.screenshot()
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode() if screenshot_bytes else ""
            screenshots_b64.append(screenshot_b64)

            action_msg = call_holo(instruction, screenshot_b64, history)
            action_text = action_msg.get("content", "")

            actions.append({"step": step, "action": action_text})
            history.append(action_msg)
            execute_action(env, action_text)

        reward = score_env(env, task_dir)

    finally:
        Env.delete_instance(env_config_path)

    save_trajectory(task_id, app, actions, screenshots_b64, reward, traj_root)
    return reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect trajectories with Holo-3.1-35B-A3B")
    parser.add_argument(
        "--run-dir", required=True, metavar="DIR",
        help="Run directory produced by select_tasks.py (e.g. data/run-20260620_185032). "
             "Reads task_selection/task_ids.json; writes trajectory/ here.",
    )
    parser.add_argument(
        "--tasks-dir", default=None, metavar="DIR",
        help=f"Source of CUA-Gym task bundles (default: {DEFAULT_TASKS_DIR})",
    )
    parser.add_argument(
        "--model-url", default=None, metavar="URL",
        help="Base URL of the vLLM endpoint, e.g. http://nlpgpu06:8000. "
             "'/v1' is appended automatically if absent. "
             "Overrides the HOLO_BASE_URL environment variable.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")

    task_ids_path = run_dir / "task_selection" / "task_ids.json"
    if not task_ids_path.exists():
        sys.exit(f"task_ids.json not found: {task_ids_path}")

    if args.model_url:
        global HOLO_BASE_URL
        base = args.model_url.rstrip("/")
        HOLO_BASE_URL = base if base.endswith("/v1") else f"{base}/v1"

    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else DEFAULT_TASKS_DIR
    if not tasks_dir.exists():
        sys.exit(f"Tasks directory not found: {tasks_dir}")

    task_ids: list[str] = json.loads(task_ids_path.read_text())
    traj_root = run_dir / "trajectory"
    traj_root.mkdir(parents=True, exist_ok=True)

    print(f"Run dir    : {run_dir}")
    print(f"Tasks dir  : {tasks_dir}")
    print(f"Model URL  : {HOLO_BASE_URL}")
    print(f"Tasks to run: {len(task_ids)}")
    print()

    done = skipped = failed = 0
    for task_id in task_ids:
        task_dir = tasks_dir / task_id
        if not task_dir.exists():
            print(f"  [MISS] {task_id}  (task dir not found)")
            failed += 1
            continue

        print(f"  [RUN]  {task_id}")
        try:
            reward = run_one_task(task_dir, traj_root)
            print(f"  [DONE] {task_id}  reward={reward:.4f}")
            done += 1
        except Exception as exc:
            print(f"  [FAIL] {task_id}  {exc}")
            failed += 1

    print()
    print(f"Finished — done={done}  skipped={skipped}  failed={failed}")
    print(f"Trajectories saved to: {traj_root}")


if __name__ == "__main__":
    main()
