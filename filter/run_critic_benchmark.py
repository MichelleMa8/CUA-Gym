import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from critic_prompt import SYSTEM_PROMPT, build_user_prompt

_REPO_ROOT  = Path(__file__).resolve().parent.parent
FINAL_DIR   = _REPO_ROOT / 'output' / 'final'
BENCH_PATH  = Path(__file__).resolve().parent / 'benchmark_tasks.json'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'


def read_text(path: Path) -> str:
    return path.read_text(errors='ignore')


def load_task_bundle(task_id: str) -> dict[str, str]:
    task_dir = FINAL_DIR / task_id
    config = json.loads((task_dir / 'config.json').read_text())
    query = config.get('instruction', '')
    return {
        'task_id': task_id,
        'query': query,
        'setup': read_text(task_dir / 'initial_setup.py'),
        'golden_setup': read_text(task_dir / 'golden_patch.py'),
        'reward': read_text(task_dir / 'reward.py'),
    }


def parse_json_maybe(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        raise


async def call_one(client: AsyncOpenAI, model: str, sem: asyncio.Semaphore, sample: dict[str, Any]) -> dict[str, Any]:
    task = load_task_bundle(sample['task_id'])
    prompt = build_user_prompt(**task)
    req = {
        'model': model,
        'temperature': 0,
        'response_format': {"type": "json_object"},
    }
    if 'claude' in model:
        req['system'] = SYSTEM_PROMPT
        req['messages'] = [{"role": "user", "content": prompt}]
    else:
        req['messages'] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    async with sem:
        stream = await client.chat.completions.create(stream=True, **req)
        parts = []
        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                if getattr(delta, 'content', None):
                    parts.append(delta.content)
            except Exception:
                pass
        content = ''.join(parts)
    parsed = parse_json_maybe(content)
    return {
        'task_id': sample['task_id'],
        'expected_verdict': sample['expected_verdict'],
        'expected_severity': sample['expected_severity'],
        'why': sample['why'],
        'raw_response': content,
        'parsed': parsed,
    }


def grade(result: dict[str, Any]) -> dict[str, Any]:
    parsed = result['parsed']
    verdict_ok = parsed.get('verdict') == result['expected_verdict']
    severity_ok = parsed.get('severity') == result['expected_severity']
    revised_ok = True
    if parsed.get('verdict') == 'modify_query':
        revised_ok = bool((parsed.get('revised_query') or '').strip())
    else:
        revised_ok = not bool((parsed.get('revised_query') or '').strip())
    return {
        **result,
        'verdict_ok': verdict_ok,
        'severity_ok': severity_ok,
        'revised_query_shape_ok': revised_ok,
        'overall_ok': verdict_ok and severity_ok and revised_ok,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen-max')
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bench = json.loads(BENCH_PATH.read_text())
    if args.limit:
        bench = bench[:args.limit]

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    tasks = [call_one(client, args.model, sem, sample) for sample in bench]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    graded = []
    for item in raw_results:
        if isinstance(item, Exception):
            graded.append({'error': repr(item), 'overall_ok': False})
            continue
        graded.append(grade(item))

    summary = {
        'model': args.model,
        'total': len(graded),
        'success': sum(1 for x in graded if x.get('overall_ok')),
        'verdict_accuracy': sum(1 for x in graded if x.get('verdict_ok')),
        'severity_accuracy': sum(1 for x in graded if x.get('severity_ok')),
        'shape_accuracy': sum(1 for x in graded if x.get('revised_query_shape_ok')),
        'results': graded,
    }
    out = RESULTS_DIR / f'benchmark_{args.model}.json'
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, ensure_ascii=False, indent=2))
    print(f'Wrote {out}')


if __name__ == '__main__':
    asyncio.run(main())
