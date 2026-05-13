"""
CUA-Gym Reward LLM Judge — deployed to VM by orchestrator.

This file is uploaded to /tmp/reward_judge.py on each VM before reward.py runs.
It provides a locked-down call_llm_judge() function with fixed model, temperature,
and system prompt. reward.py imports this instead of using raw OpenAI SDK.

DO NOT modify the model, temperature, or system prompt from reward.py.
These parameters are intentionally locked to prevent reward-gen from
introducing variability or gaming the judge.
"""

import base64
import json
import os
import re


def _load_env_file(path="/tmp/reward_env"):
    """Load KEY=VALUE env file written by orchestrator. Sets os.environ."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())
    except FileNotFoundError:
        pass


# Bootstrap: load credentials from orchestrator-deployed env file
_load_env_file()


# --- Fixed parameters (not configurable by reward.py) ---

_MODEL = "claude-sonnet-4-5-20250929"
_VISION_MODEL = "qwen-vl-max"  # Vision-capable model (DashScope native; Claude proxy lacks vision)
_TEMPERATURE = 0.0
_MAX_RETRIES = 5

_SYSTEM_PROMPT = """\
You are a precise task completion evaluator for computer-use agent training.

Your job: determine whether a web application's current state matches the expected
outcome described in the success criteria. Score on a 0.0-1.0 scale.

Rules:
- Semantic equivalence is acceptable (e.g., "SD-USA" ≈ "San Diego, USA")
- Minor formatting differences are acceptable (extra spaces, capitalization)
- Missing information or wrong information scores 0.0 for that criterion
- Partial completion gets proportional partial credit
- Be strict: do not give credit for vaguely related content

Respond with ONLY a JSON object:
{"score": <float 0.0-1.0>, "reasoning": "<brief explanation of each criterion>"}
"""


def call_llm_judge(
    task_instruction: str,
    success_criteria: str,
    state_excerpt: str,
    max_tokens: int = 300,
) -> float:
    """
    Call LLM to judge semantic equivalence or subjective quality.

    Args:
        task_instruction: The original task given to the agent
        success_criteria: Specific criteria to evaluate (be precise!)
        state_excerpt: JSON string of the relevant state slice to evaluate
        max_tokens: Max response length (default 300)

    Returns:
        float between 0.0 and 1.0
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("LLM_JUDGE_ERROR: No API key found (OPENAI_API_KEY or MSH_API_KEY)")
        return 0.0

    client = OpenAI(api_key=api_key, base_url=base_url)

    user_prompt = f"""TASK: {task_instruction}

SUCCESS CRITERIA:
{success_criteria}

STATE TO EVALUATE:
{state_excerpt[:6000]}

Score 0.0-1.0 based on how well the state meets the success criteria.
Respond with JSON only: {{"score": <float>, "reasoning": "<brief>"}}"""

    last_error = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=_TEMPERATURE,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content.strip()

            # Parse score from JSON response
            try:
                result = json.loads(text)
                score = float(result["score"])
            except (json.JSONDecodeError, KeyError, ValueError):
                match = re.search(r'"?score"?\s*:\s*([\d.]+)', text)
                score = float(match.group(1)) if match else 0.0

            score = max(0.0, min(1.0, score))
            print(f"LLM_JUDGE: score={score} | {text[:200]}")
            return score

        except Exception as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                import time
                time.sleep(1.0 * (attempt + 1))
                continue

    print(f"LLM_JUDGE_ERROR: All {_MAX_RETRIES + 1} attempts failed. Last error: {last_error}")
    return 0.0


# --- Vision Judge (for multimedia / image-based tasks) ---

_VISION_SYSTEM_PROMPT = """\
You are a precise visual task completion evaluator for computer-use agent training.

Your job: compare a BEFORE image (initial state) and an AFTER image (result after agent acted),
then judge whether the described task was correctly completed. Score on a 0.0-1.0 scale.

Rules:
- Focus on whether the SPECIFIC operation described in the task was applied
- Compare BEFORE and AFTER to detect what changed — the change should match the task
- Minor quality differences (compression artifacts, slight color shifts) are acceptable
- The operation must be clearly visible and correct, not just partially applied
- If the AFTER image is identical to BEFORE (nothing happened), score 0.0
- If the AFTER image shows the wrong operation, score 0.0
- Partial completion gets proportional partial credit

Respond with ONLY a JSON object:
{"score": <float 0.0-1.0>, "reasoning": "<brief explanation of what you observe>"}
"""


def _encode_image_file(image_path: str) -> str:
    """Read an image file and return base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _detect_media_type(image_path: str) -> str:
    """Detect MIME type from file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/png")


def _extract_video_frames(video_path: str, num_frames: int = 3) -> list:
    """Extract key frames from a video file as base64 PNG strings.

    Returns list of (base64_str, "image/png") tuples.
    Requires cv2 (opencv-python) on the VM.
    """
    import cv2
    import tempfile

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Sample evenly: first, middle, last (and more if num_frames > 3)
    indices = [int(total_frames * i / (num_frames + 1)) for i in range(1, num_frames + 1)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            cv2.imwrite(tmp.name, frame)
            frames.append((_encode_image_file(tmp.name), "image/png"))
            os.unlink(tmp.name)
    cap.release()
    return frames


def call_vision_judge(
    task_instruction: str,
    initial_image: str,
    result_image: str,
    success_criteria: str = "",
    max_tokens: int = 400,
) -> float:
    """
    Call multimodal LLM to judge visual task completion by comparing
    the initial (before) image with the result (after) image.

    Args:
        task_instruction: The original task given to the agent
        initial_image: File path to the BEFORE image (on VM)
        result_image: File path to the AFTER image (on VM)
        success_criteria: Optional extra criteria beyond the task instruction
        max_tokens: Max response length

    Returns:
        float between 0.0 and 1.0
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("VISION_JUDGE_ERROR: No API key found")
        return 0.0

    # Validate files exist
    for path, label in [(initial_image, "initial"), (result_image, "result")]:
        if not os.path.isfile(path):
            print(f"VISION_JUDGE_ERROR: {label} image not found: {path}")
            return 0.0

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Build image content blocks
    initial_b64 = _encode_image_file(initial_image)
    result_b64 = _encode_image_file(result_image)
    initial_mime = _detect_media_type(initial_image)
    result_mime = _detect_media_type(result_image)

    criteria_text = f"\n\nADDITIONAL SUCCESS CRITERIA:\n{success_criteria}" if success_criteria else ""

    user_content = [
        {"type": "text", "text": f"TASK: {task_instruction}{criteria_text}\n\nBEFORE image (initial state):"},
        {"type": "image_url", "image_url": {
            "url": f"data:{initial_mime};base64,{initial_b64}",
        }},
        {"type": "text", "text": "AFTER image (result after agent acted):"},
        {"type": "image_url", "image_url": {
            "url": f"data:{result_mime};base64,{result_b64}",
        }},
        {"type": "text", "text": 'Score 0.0-1.0 based on whether the task was correctly completed.\nRespond with JSON only: {"score": <float>, "reasoning": "<brief>"}'},
    ]

    last_error = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=_VISION_MODEL,
                messages=[
                    {"role": "system", "content": _VISION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=_TEMPERATURE,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content.strip()

            try:
                result = json.loads(text)
                score = float(result["score"])
            except (json.JSONDecodeError, KeyError, ValueError):
                match = re.search(r'"?score"?\s*:\s*([\d.]+)', text)
                score = float(match.group(1)) if match else 0.0

            score = max(0.0, min(1.0, score))
            print(f"VISION_JUDGE: score={score} | {text[:200]}")
            return score

        except Exception as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                import time
                time.sleep(1.0 * (attempt + 1))
                continue

    print(f"VISION_JUDGE_ERROR: All {_MAX_RETRIES + 1} attempts failed. Last error: {last_error}")
    return 0.0


def call_video_vision_judge(
    task_instruction: str,
    initial_video: str,
    result_video: str,
    success_criteria: str = "",
    num_frames: int = 3,
    max_tokens: int = 500,
) -> float:
    """
    Call multimodal LLM to judge video task completion by comparing
    key frames extracted from BEFORE and AFTER videos.

    Args:
        task_instruction: The original task given to the agent
        initial_video: File path to the BEFORE video (on VM)
        result_video: File path to the AFTER video (on VM)
        success_criteria: Optional extra criteria
        num_frames: Number of frames to extract from each video (default 3)
        max_tokens: Max response length

    Returns:
        float between 0.0 and 1.0
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("VISION_JUDGE_ERROR: No API key found")
        return 0.0

    for path, label in [(initial_video, "initial"), (result_video, "result")]:
        if not os.path.isfile(path):
            print(f"VISION_JUDGE_ERROR: {label} video not found: {path}")
            return 0.0

    initial_frames = _extract_video_frames(initial_video, num_frames)
    result_frames = _extract_video_frames(result_video, num_frames)

    if not initial_frames or not result_frames:
        print("VISION_JUDGE_ERROR: Failed to extract frames from video(s)")
        return 0.0

    client = OpenAI(api_key=api_key, base_url=base_url)

    criteria_text = f"\n\nADDITIONAL SUCCESS CRITERIA:\n{success_criteria}" if success_criteria else ""

    # Build content: initial frames, then result frames
    user_content = [
        {"type": "text", "text": f"TASK: {task_instruction}{criteria_text}\n\nBEFORE video key frames (initial state):"},
    ]
    for b64, mime in initial_frames:
        user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    user_content.append({"type": "text", "text": "AFTER video key frames (result after agent acted):"})
    for b64, mime in result_frames:
        user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    user_content.append({"type": "text", "text": 'Score 0.0-1.0 based on whether the task was correctly completed.\nRespond with JSON only: {"score": <float>, "reasoning": "<brief>"}'})

    last_error = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=_VISION_MODEL,
                messages=[
                    {"role": "system", "content": _VISION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=_TEMPERATURE,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content.strip()

            try:
                result = json.loads(text)
                score = float(result["score"])
            except (json.JSONDecodeError, KeyError, ValueError):
                match = re.search(r'"?score"?\s*:\s*([\d.]+)', text)
                score = float(match.group(1)) if match else 0.0

            score = max(0.0, min(1.0, score))
            print(f"VIDEO_VISION_JUDGE: score={score} | {text[:200]}")
            return score

        except Exception as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                import time
                time.sleep(1.0 * (attempt + 1))
                continue

    print(f"VIDEO_VISION_JUDGE_ERROR: All {_MAX_RETRIES + 1} attempts failed. Last error: {last_error}")
    return 0.0
