#!/usr/bin/env python3
"""Majority-vote LLM filter for CUA-Gym training tasks.

Each task is evaluated by the critic LLM independently N times (default N=3).
The final verdict is determined by majority vote:
  - reject     if > N/2 votes say 'reject'
  - modify_query if > N/2 votes say 'modify_query' (and not already rejected)
  - keep       otherwise

This implements the majority-vote filter stage described in the CUA-Gym paper,
which rejects ~3,100 tasks before the teacher-rollout filtering stage.

Task directory layout expected under --tasks-dir:
    <tasks-dir>/
      <task-id>/
        config.json          # contains 'instruction', written back with filter fields
        initial_setup.py
        golden_patch.py
        reward.py

Filter fields written to config.json:
  - reject (bool)            true if majority vote says reject
  - train_poor_fit (bool)    true if majority training_pool_fit == 'poor'
  - revised_query (str)      present only when majority says modify_query

Usage:
    export OPENAI_API_KEY='sk-...'

    # Dry run (count pending tasks):
    python majority_vote_filter.py --tasks-dir output/final --dry-run

    # Run with 3 votes per task, write results:
    python majority_vote_filter.py --tasks-dir output/final --votes 3 --write

    # Filter a specific app type only:
    python majority_vote_filter.py --tasks-dir output/final --votes 3 --write \\
        --app-type libreoffice_calc

    # Use a different model:
    python majority_vote_filter.py --tasks-dir output/final --votes 3 --write \\
        --model claude-opus-4-7
"""
import argparse
import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from critic_prompt import SYSTEM_PROMPT, build_user_prompt

_DEFAULT_TASKS_DIR = Path(__file__).resolve().parent.parent / 'output' / 'final'


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_task_bundle(task_dir: Path) -> dict[str, str]:
    config = json.loads((task_dir / 'config.json').read_text())
    return {
        'task_id': task_dir.name,
        'query': config.get('instruction', ''),
        'setup': (task_dir / 'initial_setup.py').read_text(errors='ignore')
                 if (task_dir / 'initial_setup.py').exists() else '',
        'golden_setup': (task_dir / 'golden_patch.py').read_text(errors='ignore')
                        if (task_dir / 'golden_patch.py').exists() else '',
        'reward': (task_dir / 'reward.py').read_text(errors='ignore')
                  if (task_dir / 'reward.py').exists() else '',
    }


def is_already_filtered(config: dict) -> bool:
    return 'reject' in config or 'train_poor_fit' in config


# ---------------------------------------------------------------------------
# LLM call (single vote)
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


async def call_critic_once(
    client: AsyncOpenAI,
    model: str,
    sem: asyncio.Semaphore,
    bundle: dict[str, str],
    max_retries: int = 2,
) -> dict[str, Any] | None:
    prompt = build_user_prompt(**bundle)
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': prompt},
    ]
    for attempt in range(max_retries + 1):
        try:
            async with sem:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=1.0,   # non-zero so N votes are independent
                    response_format={'type': 'json_object'},
                    timeout=180,
                )
            return _parse_json(response.choices[0].message.content or '')
        except Exception as exc:
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            print(f'[WARN] critic call failed for {bundle["task_id"]}: {exc!r:.120}',
                  flush=True)
            return None


# ---------------------------------------------------------------------------
# Majority vote aggregation
# ---------------------------------------------------------------------------

def aggregate_votes(votes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate N critic votes into a single majority-vote decision."""
    valid = [v for v in votes if v is not None]
    n = len(valid)
    if n == 0:
        return {'verdict': 'keep', 'train_poor_fit': False, 'revised_query': ''}

    verdict_counts: Counter = Counter(v.get('verdict', 'keep') for v in valid)
    best_verdict, best_count = verdict_counts.most_common(1)[0]

    # Majority threshold: strictly more than half
    majority = best_count > n / 2
    final_verdict = best_verdict if majority else 'keep'

    # training_pool_fit: majority 'poor' → mark poor
    fit_counts: Counter = Counter(v.get('training_pool_fit', 'good') for v in valid)
    train_poor_fit = fit_counts.get('poor', 0) > n / 2

    # revised_query: only if majority says modify_query; pick most common non-empty rewrite
    revised_query = ''
    if final_verdict == 'modify_query':
        rewrites = [
            v.get('revised_query', '').strip()
            for v in valid
            if v.get('verdict') == 'modify_query' and v.get('revised_query', '').strip()
        ]
        if rewrites:
            revised_query = Counter(rewrites).most_common(1)[0][0]

    return {
        'verdict': final_verdict,
        'train_poor_fit': train_poor_fit,
        'revised_query': revised_query,
        'vote_breakdown': dict(verdict_counts),
        'valid_votes': n,
        'total_votes': len(votes),
    }


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------

async def process_task(
    client: AsyncOpenAI,
    model: str,
    sem: asyncio.Semaphore,
    task_dir: Path,
    n_votes: int,
    write: bool,
    tracker: 'ProgressTracker',
) -> dict[str, Any]:
    try:
        bundle = load_task_bundle(task_dir)
    except Exception as exc:
        await tracker.record(error=True)
        return {'task_id': task_dir.name, 'error': str(exc)}

    # Cast N independent votes concurrently
    vote_coros = [call_critic_once(client, model, sem, bundle) for _ in range(n_votes)]
    votes = await asyncio.gather(*vote_coros)

    agg = aggregate_votes(list(votes))

    if write:
        config_path = task_dir / 'config.json'
        config = json.loads(config_path.read_text())
        config['reject'] = (agg['verdict'] == 'reject')
        config['train_poor_fit'] = agg['train_poor_fit']
        if agg['revised_query']:
            config['revised_query'] = agg['revised_query']
        else:
            config.pop('revised_query', None)
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')

    await tracker.record(
        reject=agg['verdict'] == 'reject',
        poor_fit=agg['train_poor_fit'],
        revised=bool(agg['revised_query']),
    )
    return {'task_id': task_dir.name, **agg}


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.done = self.errors = self.rejected = self.poor_fit = self.revised = 0
        self.start = time.time()
        self._lock = asyncio.Lock()

    async def record(self, *, reject: bool = False, poor_fit: bool = False,
                     revised: bool = False, error: bool = False):
        async with self._lock:
            self.done += 1
            if error:
                self.errors += 1
            else:
                self.rejected  += int(reject)
                self.poor_fit  += int(poor_fit)
                self.revised   += int(revised)
            if self.done % 50 == 0 or self.done == self.total:
                elapsed = time.time() - self.start
                rate = self.done / elapsed if elapsed else 0
                eta = (self.total - self.done) / rate if rate else 0
                print(
                    f'[{self.done}/{self.total}] '
                    f'rejected={self.rejected} poor_fit={self.poor_fit} '
                    f'revised={self.revised} err={self.errors} '
                    f'| {rate:.1f}/s ETA {eta:.0f}s',
                    flush=True,
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description='Majority-vote LLM filter for CUA-Gym tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else '',
    )
    parser.add_argument(
        '--tasks-dir', type=Path, default=_DEFAULT_TASKS_DIR,
        help='Directory of per-task folders (default: output/final)',
    )
    parser.add_argument(
        '--model', default='gpt-4o',
        help='LLM model for the critic (default: gpt-4o)',
    )
    parser.add_argument(
        '--votes', type=int, default=3, metavar='N',
        help='Number of independent critic votes per task (default: 3)',
    )
    parser.add_argument(
        '--concurrency', type=int, default=32,
        help='Max concurrent LLM calls (default: 32)',
    )
    parser.add_argument(
        '--write', action='store_true',
        help='Write filter results back to each task\'s config.json',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-run even if the task already has filter fields',
    )
    parser.add_argument(
        '--app-type', default='',
        help='Only process tasks whose config app_type matches (e.g. libreoffice_calc)',
    )
    parser.add_argument(
        '--limit', type=int, default=0,
        help='Cap the number of tasks to process (0 = all pending)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Count pending tasks and exit without running the critic',
    )
    args = parser.parse_args()

    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key and not args.dry_run:
        parser.error('OPENAI_API_KEY environment variable is not set')

    tasks_dir = args.tasks_dir
    if not tasks_dir.exists():
        parser.error(f'--tasks-dir does not exist: {tasks_dir}')

    # Collect pending tasks
    all_dirs = sorted(
        d for d in tasks_dir.iterdir()
        if d.is_dir() and (d / 'config.json').exists()
    )
    pending: list[Path] = []
    for d in all_dirs:
        try:
            cfg = json.loads((d / 'config.json').read_text())
        except Exception:
            continue
        if args.app_type and cfg.get('app_type') != args.app_type:
            continue
        if not args.force and is_already_filtered(cfg):
            continue
        pending.append(d)

    desc = f'app_type={args.app_type}' if args.app_type else 'all'
    print(f'Tasks total={len(all_dirs)}, filter={desc}, pending={len(pending)}', flush=True)

    if args.dry_run:
        return

    if args.limit:
        pending = pending[:args.limit]
        print(f'Capped to {len(pending)} tasks (--limit)', flush=True)

    if not pending:
        print('Nothing to do.')
        return

    print(
        f'Starting majority-vote filter: model={args.model}, votes={args.votes}, '
        f'concurrency={args.concurrency}, write={args.write}',
        flush=True,
    )

    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(args.concurrency)
    tracker = ProgressTracker(len(pending))

    results = await asyncio.gather(*(
        process_task(client, args.model, sem, d, args.votes, args.write, tracker)
        for d in pending
    ))

    elapsed = time.time() - tracker.start
    summary = {
        'model':        args.model,
        'votes_per_task': args.votes,
        'app_type':     args.app_type or 'all',
        'total':        len(pending),
        'rejected':     tracker.rejected,
        'poor_fit':     tracker.poor_fit,
        'revised':      tracker.revised,
        'errors':       tracker.errors,
        'elapsed_s':    round(elapsed, 1),
        'tasks_per_s':  round(len(pending) / elapsed, 2) if elapsed else 0,
    }
    print('\n=== DONE ===', flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
