#!/usr/bin/env python3
"""
CLI wrapper for VM environment operations.

Subagents invoke this via Bash tool to interact with the remote VM.
Connects directly to VM IP (no SSH tunnel needed when running server-side).
Falls back to SSH tunnel if ssh_gateway is present in env_config.json (legacy).

Usage:
    python3 scripts/env_cli.py --config env_config.json <command> [args]

Commands:
    execute   "ls -la"                    Execute shell command
    run-python script.py                  Run Python script (use - for stdin)
    run-bash   script.sh [--workdir /p]   Run bash script
    upload     local.py /remote/path      Upload file to VM
    download   /remote/path local.txt     Download file from VM
    screenshot output.png                 Save VM screenshot
    screen-size                           Print screen dimensions
    launch     "libreoffice --calc"       Launch app (non-blocking)
    dir-tree   /home/user                 List directory
    create     --output env_config.json   Create a new VM (orchestrator only)
    delete                                Delete VM from config (orchestrator only)
"""

import argparse
import importlib.util
import json
import sys
import os

# Import utils/env.py directly to avoid pulling in utils/__init__.py
# (which may require openai and other heavy deps not needed for CLI operations)
_env_path = os.path.join(os.path.dirname(__file__), "..", "utils", "env.py")
_spec = importlib.util.spec_from_file_location("utils.env", _env_path)
_env_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_env_mod)
Env = _env_mod.Env
EnvError = _env_mod.EnvError


def cmd_execute(env, args):
    result = env.execute(args.command_str)
    stdout = result.get("output", result.get("stdout", ""))
    stderr = result.get("error", result.get("stderr", ""))
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
    return result.get("returncode", 0) or 0


def cmd_run_python(env, args):
    if args.script == "-":
        script_content = sys.stdin.read()
    else:
        with open(args.script) as f:
            script_content = f.read()
    result = env.run_python(script_content)
    stdout = result.get("output", result.get("stdout", ""))
    stderr = result.get("error", result.get("stderr", ""))
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
    return result.get("returncode", 0) or 0


def cmd_run_bash(env, args):
    if args.script == "-":
        script_content = sys.stdin.read()
    else:
        with open(args.script) as f:
            script_content = f.read()
    result = env.run_bash(script_content, working_dir=args.workdir)
    stdout = result.get("output", result.get("stdout", ""))
    stderr = result.get("error", result.get("stderr", ""))
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
    return result.get("returncode", 0) or 0


def cmd_upload(env, args):
    env.upload(args.local_path, args.remote_path)
    print(f"Uploaded {args.local_path} -> {args.remote_path}")
    return 0


def cmd_download(env, args):
    data = env.download(args.remote_path)
    if data is None:
        print(f"Failed to download {args.remote_path}", file=sys.stderr)
        return 1
    with open(args.local_path, "wb") as f:
        f.write(data)
    print(f"Downloaded {args.remote_path} -> {args.local_path} ({len(data)} bytes)")
    return 0


def cmd_screenshot(env, args):
    data = env.screenshot()
    if data is None:
        print("Failed to get screenshot", file=sys.stderr)
        return 1
    with open(args.output, "wb") as f:
        f.write(data)
    print(f"Screenshot saved to {args.output} ({len(data)} bytes)")
    return 0


def cmd_screen_size(env, args):
    result = env.get_screen_size()
    width = result.get("width", "?")
    height = result.get("height", "?")
    print(f"{width}x{height}")
    return 0


def cmd_launch(env, args):
    env.launch(args.command_str)
    print(f"Launched: {args.command_str}")
    return 0


def cmd_dir_tree(env, args):
    result = env.get_directory_tree(args.path)
    print(json.dumps(result, indent=2))
    return 0


def cmd_create(args):
    """Create a new VM and save config. Does NOT need --config (it creates one)."""
    env = Env.create(task_id=args.task_id)
    env.save_config(args.output)
    print(f"VM created: {env.config.instance_id} ({env.config.vm_ip})")
    print(f"Config saved to {args.output}")
    env.close()
    return 0


def cmd_delete(args):
    """Delete VM from saved config."""
    Env.delete_instance(args.config)
    print(f"VM deleted (config: {args.config})")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="CLI for VM environment operations",
        prog="env_cli.py",
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to env_config.json (required for most commands)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # execute
    p = subparsers.add_parser("execute", help="Execute shell command on VM")
    p.add_argument("command_str", metavar="command", help="Shell command to execute")

    # run-python
    p = subparsers.add_parser("run-python", help="Run Python script on VM")
    p.add_argument("script", help="Path to Python script (use - for stdin)")

    # run-bash
    p = subparsers.add_parser("run-bash", help="Run bash script on VM")
    p.add_argument("script", help="Path to bash script (use - for stdin)")
    p.add_argument("--workdir", default=None, help="Working directory on VM")

    # upload
    p = subparsers.add_parser("upload", help="Upload file to VM")
    p.add_argument("local_path", help="Local file path")
    p.add_argument("remote_path", help="Remote file path on VM")

    # download
    p = subparsers.add_parser("download", help="Download file from VM")
    p.add_argument("remote_path", help="Remote file path on VM")
    p.add_argument("local_path", help="Local file path to save to")

    # screenshot
    p = subparsers.add_parser("screenshot", help="Save VM screenshot")
    p.add_argument("output", help="Output file path (PNG)")

    # screen-size
    subparsers.add_parser("screen-size", help="Print screen dimensions")

    # launch
    p = subparsers.add_parser("launch", help="Launch app on VM (non-blocking)")
    p.add_argument("command_str", metavar="command", help="Command to launch")

    # dir-tree
    p = subparsers.add_parser("dir-tree", help="List directory on VM")
    p.add_argument("path", help="Directory path on VM")

    # create (no --config needed)
    p = subparsers.add_parser("create", help="Create a new VM and save config")
    p.add_argument("--output", "-o", default="env_config.json",
                   help="Output config path (default: env_config.json)")
    p.add_argument("--task-id", default=None, help="Task ID to tag the VM")

    # delete
    subparsers.add_parser("delete", help="Delete VM from config")

    return parser


COMMAND_HANDLERS = {
    "execute": cmd_execute,
    "run-python": cmd_run_python,
    "run-bash": cmd_run_bash,
    "upload": cmd_upload,
    "download": cmd_download,
    "screenshot": cmd_screenshot,
    "screen-size": cmd_screen_size,
    "launch": cmd_launch,
    "dir-tree": cmd_dir_tree,
}


def main():
    parser = build_parser()
    args = parser.parse_args()

    # create/delete are special — they don't need an existing env
    if args.command == "create":
        return cmd_create(args)
    if args.command == "delete":
        if not args.config:
            print("Error: --config is required for delete", file=sys.stderr)
            return 1
        return cmd_delete(args)

    # All other commands need --config
    if not args.config:
        print("Error: --config is required", file=sys.stderr)
        return 1

    try:
        env = Env.from_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    handler = COMMAND_HANDLERS.get(args.command)
    if not handler:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    try:
        return handler(env, args)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EnvError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
