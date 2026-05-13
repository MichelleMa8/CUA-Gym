"""
VM Environment management for CUA-Gym.

Architecture (default — server-side):
  cpfs02 server → Aliyun ECS VM (private IP:5000) — direct connection, no SSH tunnel

Architecture (legacy — remote Mac):
  Local Mac → SSH Gateway ($SSH_GATEWAY_HOST) → Aliyun ECS VM (private IP)
  HTTP requests flow through an SSH tunnel: localhost:local_port → vm_private_ip:5000

Usage:
  # Orchestrator: create VM and save config (uses LocalExecutor by default)
  env = Env.create(task_id="my_task")
  env.save_config("env_config.json")

  # Subagent: reconnect from config (direct connection when no ssh_gateway in config)
  env = Env.from_config("env_config.json")
  result = env.execute("ls -la")
  data = env.screenshot()

  # Cleanup
  Env.delete_instance("env_config.json")
"""

import base64
import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import requests

# Ensure localhost traffic bypasses all proxies (http_proxy, https_proxy, all_proxy).
# VM communication goes through SSH tunnel at localhost:<port>, and proxy interference
# causes 502 errors or SOCKS dependency failures.
_no_proxy = os.environ.get("no_proxy", "")
if "localhost" not in _no_proxy:
    os.environ["no_proxy"] = f"localhost,127.0.0.1,{_no_proxy}".rstrip(",")

logger = logging.getLogger("cua_gym.env")


class EnvError(Exception):
    """Exception raised by Env operations."""
    pass


@dataclass
class EnvConfig:
    """Serializable VM connection config. Saved as env_config.json."""
    vm_ip: str
    server_port: int = 5000
    instance_id: Optional[str] = None
    region: Optional[str] = None
    image_id: Optional[str] = None
    created_at: Optional[str] = None
    task_id: Optional[str] = None
    provider_name: str = "aliyun"
    ssh_gateway: Optional[dict] = None   # Legacy: {host, port, user, osworld_dir, venv_path}
    local_port: Optional[int] = None     # Legacy: local end of SSH tunnel

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> "EnvConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EnvConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _default_ssh_gateway() -> dict:
    return {
        "host": os.getenv("SSH_GATEWAY_HOST", ""),
        "port": int(os.getenv("SSH_GATEWAY_PORT", "22")),
        "user": os.getenv("SSH_GATEWAY_USER", "root"),
        "osworld_dir": os.getenv(
            "SSH_GATEWAY_OSWORLD_DIR",
            "~/OSWorld-RL",
        ),
        "venv_path": os.getenv("SSH_GATEWAY_VENV", "~/.venvs/osworld-py312"),
    }


def _default_local_executor() -> dict:
    return {
        "osworld_dir": os.getenv(
            "OSWORLD_DIR",
            "~/OSWorld-RL",
        ),
        "venv_path": os.getenv("OSWORLD_VENV", "~/.venvs/osworld-py312"),
    }


def _collect_aliyun_env_vars() -> dict:
    keys = [
        "ALIYUN_ACCESS_KEY_ID", "ALIYUN_ACCESS_KEY_SECRET", "ALIYUN_REGION",
        "ALIYUN_IMAGE_ID", "ALIYUN_INSTANCE_TYPE", "ALIYUN_VSWITCH_ID",
        "ALIYUN_SECURITY_GROUP_ID", "ALIYUN_RESOURCE_GROUP_ID",
        "ALIYUN_USE_PRIVATE_IP",
    ]
    return {k: os.environ[k] for k in keys if k in os.environ}


# ---------------------------------------------------------------------------
# Remote scripts (executed on the gateway server via SSH)
# ---------------------------------------------------------------------------

_VM_CREATE_SCRIPT = r'''
import os, sys, time, json, urllib.request

from alibabacloud_ecs20140526.client import Client as EcsClient
from alibabacloud_ecs20140526 import models as ecs_models
from alibabacloud_tea_openapi import models as open_api_models

region = os.environ.get("ALIYUN_REGION", "ap-southeast-1")

api_cfg = open_api_models.Config(
    access_key_id=os.environ["ALIYUN_ACCESS_KEY_ID"],
    access_key_secret=os.environ["ALIYUN_ACCESS_KEY_SECRET"],
)
api_cfg.endpoint = f"ecs.{region}.aliyuncs.com"
client = EcsClient(api_cfg)

sys_disk = ecs_models.RunInstancesRequestSystemDisk(size="80", category="cloud_essd")

# Try primary instance type first, then fallbacks if NoStock
primary_type = os.environ.get("ALIYUN_INSTANCE_TYPE", "ecs.g8a.xlarge")
fallback_types = [primary_type, "ecs.g7.xlarge", "ecs.g6.xlarge", "ecs.c8a.xlarge", "ecs.c7.xlarge", "ecs.c6.xlarge", "ecs.r7.xlarge", "ecs.g8i.xlarge", "ecs.c8i.xlarge", "ecs.g8ae.xlarge", "ecs.g6e.xlarge", "ecs.u1-c1m4.xlarge"]
# Deduplicate while preserving order
seen = set()
instance_types = []
for t in fallback_types:
    if t not in seen:
        seen.add(t)
        instance_types.append(t)

resp = None
for itype in instance_types:
    req = ecs_models.RunInstancesRequest(
        region_id=region,
        image_id=os.environ["ALIYUN_IMAGE_ID"],
        instance_type=itype,
        security_group_id=os.environ["ALIYUN_SECURITY_GROUP_ID"],
        v_switch_id=os.environ["ALIYUN_VSWITCH_ID"],
        instance_charge_type="PostPaid",
        internet_max_bandwidth_out=10,
        system_disk=sys_disk,
        amount=1,
    )
    rg = os.environ.get("ALIYUN_RESOURCE_GROUP_ID")
    if rg:
        req.resource_group_id = rg
    try:
        resp = client.run_instances(req)
        print(f"INSTANCE_TYPE_USED={itype}", flush=True)
        break
    except Exception as e:
        if "NoStock" in str(e):
            print(f"NoStock for {itype}, trying next...", flush=True)
            continue
        raise
else:
    print("ERROR=all_instance_types_out_of_stock")
    sys.exit(1)

instance_id = resp.body.instance_id_sets.instance_id_set[0]
print(f"INSTANCE_ID={instance_id}", flush=True)

# Wait for Running state
for i in range(60):
    time.sleep(5)
    d = ecs_models.DescribeInstancesRequest(
        region_id=region,
        instance_ids=json.dumps([instance_id]),
    )
    dr = client.describe_instances(d)
    insts = dr.body.instances.instance
    if insts and insts[0].status == "Running":
        inst = insts[0]
        break
    print(f"WAIT status={insts[0].status if insts else 'unknown'}", flush=True)
else:
    print("ERROR=timeout_waiting_for_running")
    sys.exit(1)

# Get IP
use_private = os.environ.get("ALIYUN_USE_PRIVATE_IP", "1") == "1"
ip = ""
if use_private and inst.vpc_attributes and inst.vpc_attributes.private_ip_address:
    ips = inst.vpc_attributes.private_ip_address.ip_address
    ip = ips[0] if ips else ""
else:
    if inst.public_ip_address:
        ips = inst.public_ip_address.ip_address
        ip = ips[0] if ips else ""
    if not ip and hasattr(inst, "eip_address") and inst.eip_address:
        ip = inst.eip_address.ip_address or ""

if not ip:
    print("ERROR=no_ip_found")
    sys.exit(1)

print(f"VM_IP={ip}", flush=True)

# Wait for Flask server (port 5000)
for i in range(30):
    time.sleep(10)
    try:
        r = urllib.request.urlopen(f"http://{ip}:5000/screenshot", timeout=10)
        if r.status == 200:
            print("SERVER_READY=true", flush=True)
            break
    except Exception as e:
        print(f"WAIT server attempt {i+1}: {e}", flush=True)
else:
    print("SERVER_READY=false", flush=True)

print("CREATION_COMPLETE", flush=True)
'''

_VM_DELETE_SCRIPT = r'''
import os, sys, time, json

from alibabacloud_ecs20140526.client import Client as EcsClient
from alibabacloud_ecs20140526 import models as ecs_models
from alibabacloud_tea_openapi import models as open_api_models

region = os.environ.get("ALIYUN_REGION", "ap-southeast-1")
instance_id = os.environ["INSTANCE_ID"]

api_cfg = open_api_models.Config(
    access_key_id=os.environ["ALIYUN_ACCESS_KEY_ID"],
    access_key_secret=os.environ["ALIYUN_ACCESS_KEY_SECRET"],
)
api_cfg.endpoint = f"ecs.{region}.aliyuncs.com"
client = EcsClient(api_cfg)

for attempt in range(3):
    try:
        req = ecs_models.DeleteInstanceRequest(instance_id=instance_id, force=True)
        client.delete_instance(req)
        print(f"DELETED={instance_id}", flush=True)
        break
    except Exception as e:
        print(f"DELETE attempt {attempt+1} failed: {e}", flush=True)
        if attempt < 2:
            time.sleep(15)
else:
    print(f"ERROR=delete_failed_after_retries")
    sys.exit(1)
'''


# ---------------------------------------------------------------------------
# SSHGateway — manages SSH tunnel + remote commands through jump host
# ---------------------------------------------------------------------------

class SSHGateway:
    """SSH gateway to company server for VM lifecycle and port forwarding."""

    def __init__(self, host: str, port: int, user: str,
                 osworld_dir: str, venv_path: str = None):
        self.host = host
        self.port = port
        self.user = user
        self.osworld_dir = osworld_dir
        self.venv_path = venv_path or "~/.venvs/osworld-py312"
        self._tunnel_proc: Optional[subprocess.Popen] = None

    def _ssh_base(self) -> list:
        return [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=30",
            "-p", str(self.port),
            f"{self.user}@{self.host}",
        ]

    def run_remote(self, command: str, timeout: int = 300) -> subprocess.CompletedProcess:
        """Execute a shell command on the remote server."""
        return subprocess.run(
            self._ssh_base() + [command],
            capture_output=True, text=True, timeout=timeout,
        )

    def run_remote_python(self, script: str, timeout: int = 300,
                          env_vars: dict = None) -> subprocess.CompletedProcess:
        """Run a Python script on the remote server via stdin (avoids shell quoting issues)."""
        env_prefix = ""
        if env_vars:
            parts = []
            for k, v in env_vars.items():
                safe_v = str(v).replace("'", "'\\''")
                parts.append(f"{k}='{safe_v}'")
            env_prefix = " ".join(parts) + " "

        # Pipe script via stdin to avoid exec/base64 quoting issues in bash
        venv_python = f"{self.venv_path}/bin/python3"
        shell_cmd = f"cd {self.osworld_dir} && {env_prefix}{venv_python} -"
        return subprocess.run(
            self._ssh_base() + [shell_cmd],
            input=script, capture_output=True, text=True, timeout=timeout,
        )

    def create_vm(self, env_vars: dict = None, timeout: int = 600) -> tuple:
        """Create VM on remote server. Returns (instance_id, vm_ip)."""
        logger.info("Creating VM via SSH gateway...")
        result = self.run_remote_python(_VM_CREATE_SCRIPT, timeout=timeout, env_vars=env_vars)

        if result.returncode != 0:
            raise EnvError(
                f"VM creation failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        vm_ip = instance_id = None
        for line in result.stdout.splitlines():
            if line.startswith("VM_IP="):
                vm_ip = line.split("=", 1)[1].strip()
            elif line.startswith("INSTANCE_ID="):
                instance_id = line.split("=", 1)[1].strip()

        if not vm_ip or not instance_id:
            raise EnvError(f"Could not parse VM info from output:\n{result.stdout}")

        logger.info(f"VM created: {instance_id} ({vm_ip})")
        return instance_id, vm_ip

    def delete_vm(self, instance_id: str, env_vars: dict = None) -> None:
        """Delete VM instance on remote server."""
        logger.info(f"Deleting VM {instance_id} via SSH gateway...")
        extra = dict(env_vars or {})
        extra["INSTANCE_ID"] = instance_id
        result = self.run_remote_python(_VM_DELETE_SCRIPT, timeout=120, env_vars=extra)

        if result.returncode != 0:
            raise EnvError(
                f"VM deletion failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        logger.info(f"VM deleted: {instance_id}")

    def start_tunnel(self, vm_ip: str, remote_port: int = 5000,
                     local_port: int = None) -> int:
        """Start SSH port-forward tunnel. Returns the local port."""
        if self._tunnel_proc and self._tunnel_proc.poll() is None:
            self.stop_tunnel()

        local_port = local_port or _find_free_port()
        cmd = [
            "ssh", "-N",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=30",
            "-o", "ExitOnForwardFailure=yes",
            "-L", f"{local_port}:{vm_ip}:{remote_port}",
            "-p", str(self.port),
            f"{self.user}@{self.host}",
        ]
        self._tunnel_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

        # Wait for tunnel to accept connections
        for _ in range(30):
            time.sleep(0.5)
            if self._tunnel_proc.poll() is not None:
                stderr = self._tunnel_proc.stderr.read().decode()
                raise EnvError(f"SSH tunnel failed to start: {stderr}")
            try:
                with socket.create_connection(("localhost", local_port), timeout=1):
                    logger.info(f"SSH tunnel ready: localhost:{local_port} -> {vm_ip}:{remote_port}")
                    return local_port
            except (ConnectionRefusedError, OSError):
                continue

        # Tunnel process alive but can't connect yet — VM server may still be starting
        logger.warning(f"SSH tunnel started but connection not yet verified (localhost:{local_port})")
        return local_port

    def stop_tunnel(self) -> None:
        if self._tunnel_proc and self._tunnel_proc.poll() is None:
            self._tunnel_proc.terminate()
            try:
                self._tunnel_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._tunnel_proc.kill()
            self._tunnel_proc = None


# ---------------------------------------------------------------------------
# LocalExecutor — runs VM lifecycle scripts locally (no SSH needed)
# ---------------------------------------------------------------------------

class LocalExecutor:
    """Runs VM lifecycle Python scripts locally in the OSWorld-RL venv.

    Used when CUA-Gym runs directly on the server that can reach Aliyun VM
    private IPs without SSH tunneling.
    """

    def __init__(self, osworld_dir: str, venv_path: str = None):
        self.osworld_dir = osworld_dir
        self.venv_path = venv_path or "~/.venvs/osworld-py312"

    def run_python(self, script: str, timeout: int = 300,
                   env_vars: dict = None) -> subprocess.CompletedProcess:
        """Run a Python script locally via stdin."""
        python_bin = os.path.expanduser(f"{self.venv_path}/bin/python3")
        env = os.environ.copy()
        if env_vars:
            env.update({k: str(v) for k, v in env_vars.items()})
        return subprocess.run(
            [python_bin, "-"],
            input=script,
            capture_output=True, text=True, timeout=timeout,
            cwd=self.osworld_dir,
            env=env,
        )

    def create_vm(self, env_vars: dict = None, timeout: int = 600) -> tuple:
        """Create VM locally. Returns (instance_id, vm_ip)."""
        logger.info("Creating VM locally...")
        result = self.run_python(_VM_CREATE_SCRIPT, timeout=timeout, env_vars=env_vars)

        if result.returncode != 0:
            raise EnvError(
                f"VM creation failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        vm_ip = instance_id = None
        for line in result.stdout.splitlines():
            if line.startswith("VM_IP="):
                vm_ip = line.split("=", 1)[1].strip()
            elif line.startswith("INSTANCE_ID="):
                instance_id = line.split("=", 1)[1].strip()

        if not vm_ip or not instance_id:
            raise EnvError(f"Could not parse VM info from output:\n{result.stdout}")

        logger.info(f"VM created: {instance_id} ({vm_ip})")
        return instance_id, vm_ip

    def delete_vm(self, instance_id: str, env_vars: dict = None) -> None:
        """Delete VM instance locally."""
        logger.info(f"Deleting VM {instance_id} locally...")
        extra = dict(env_vars or {})
        extra["INSTANCE_ID"] = instance_id
        result = self.run_python(_VM_DELETE_SCRIPT, timeout=120, env_vars=extra)

        if result.returncode != 0:
            raise EnvError(
                f"VM deletion failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        logger.info(f"VM deleted: {instance_id}")


# ---------------------------------------------------------------------------
# Env — main interface for VM operations
# ---------------------------------------------------------------------------

class Env:
    """
    VM environment interface.

    - Env.create()      → creates VM via LocalExecutor (default) or SSHGateway (legacy)
    - Env.from_config() → reconnects to VM; direct connection or SSH tunnel from config
    - Env.connect()     → direct connection by IP (no gateway needed)
    """

    def __init__(self, config: EnvConfig, gateway: SSHGateway = None):
        self.config = config
        self._gateway = gateway
        self._closed = False

    @property
    def _base_url(self) -> str:
        """HTTP base URL — goes through tunnel when gateway is active."""
        if self._gateway and self._gateway._tunnel_proc:
            return f"http://localhost:{self.config.local_port}"
        return f"http://{self.config.vm_ip}:{self.config.server_port}"

    # --- Factory methods ---

    @classmethod
    def create(cls, task_id: str = None, env_vars: dict = None,
               ssh_gateway: dict = None, local_executor: dict = None) -> "Env":
        """Create a new VM. Uses LocalExecutor by default (direct connection, no SSH).

        Pass ssh_gateway dict to use legacy SSH gateway mode (e.g., running from Mac).
        """
        if env_vars is None:
            env_vars = _collect_aliyun_env_vars()

        if ssh_gateway is not None:
            # Legacy: SSH gateway mode
            gw = SSHGateway(**ssh_gateway)
            instance_id, vm_ip = gw.create_vm(env_vars=env_vars)
            local_port = gw.start_tunnel(vm_ip)
            config = EnvConfig(
                vm_ip=vm_ip,
                server_port=5000,
                instance_id=instance_id,
                region=env_vars.get("ALIYUN_REGION"),
                image_id=env_vars.get("ALIYUN_IMAGE_ID"),
                created_at=datetime.now().isoformat(),
                task_id=task_id,
                provider_name="aliyun",
                ssh_gateway=ssh_gateway,
                local_port=local_port,
            )
            return cls(config=config, gateway=gw)

        # Default: local execution, direct VM connection (no SSH tunnel)
        if local_executor is None:
            local_executor = _default_local_executor()
        executor = LocalExecutor(**local_executor)
        instance_id, vm_ip = executor.create_vm(env_vars=env_vars)
        config = EnvConfig(
            vm_ip=vm_ip,
            server_port=5000,
            instance_id=instance_id,
            region=env_vars.get("ALIYUN_REGION"),
            image_id=env_vars.get("ALIYUN_IMAGE_ID"),
            created_at=datetime.now().isoformat(),
            task_id=task_id,
            provider_name="aliyun",
        )
        return cls(config=config)

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "Env":
        """Reconnect to existing VM from env_config.json.

        Connects directly to VM IP when no ssh_gateway in config (default for server-side).
        Falls back to SSH tunnel if ssh_gateway is present (legacy Mac usage).
        """
        config = EnvConfig.load(config_path)
        gateway = None

        if config.ssh_gateway:
            # Legacy: SSH tunnel mode
            gateway = SSHGateway(**config.ssh_gateway)
            config.local_port = gateway.start_tunnel(config.vm_ip, config.server_port)

        return cls(config=config, gateway=gateway)

    @classmethod
    def connect(cls, vm_ip: str, server_port: int = 5000) -> "Env":
        """Direct connection by IP. No SSH gateway needed."""
        return cls(config=EnvConfig(vm_ip=vm_ip, server_port=server_port))

    # --- HTTP helpers ---

    # Bypass ALL local proxies for tunnel traffic (localhost → SSH tunnel → VM).
    # Without this, requests to localhost get routed through http_proxy/all_proxy and fail.
    _NO_PROXY = {"proxies": {"http": None, "https": None, "all": None}}

    def _get(self, path: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", 90)
        kwargs.setdefault("proxies", self._NO_PROXY["proxies"])
        return requests.get(f"{self._base_url}{path}", **kwargs)

    def _post_json(self, path: str, data: dict, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", 90)
        kwargs.setdefault("proxies", self._NO_PROXY["proxies"])
        return requests.post(
            f"{self._base_url}{path}",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data),
            **kwargs,
        )

    # --- Core operations ---

    def screenshot(self) -> Optional[bytes]:
        """Take a VM screenshot."""
        try:
            resp = self._get("/screenshot")
            return resp.content if resp.status_code == 200 else None
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    def execute(self, command: str) -> dict:
        """Execute a shell command on the VM. Returns {output, error, returncode}."""
        try:
            resp = self._post_json("/setup/execute", {"command": command, "shell": True})
            return resp.json() if resp.status_code == 200 else {"error": resp.text}
        except Exception as e:
            return {"error": str(e)}

    def run_python(self, script: Union[str, Path]) -> dict:
        """Run a Python script on the VM (base64-encoded via /setup/execute)."""
        if isinstance(script, (str, Path)) and os.path.isfile(str(script)):
            with open(script) as f:
                script = f.read()
        encoded = base64.b64encode(script.encode()).decode()
        cmd = f"python3 -c \"import base64; exec(base64.b64decode('{encoded}').decode())\""
        return self.execute(cmd)

    def run_bash(self, script: Union[str, Path], timeout: int = 30,
                 working_dir: str = None) -> dict:
        """Run a bash script on the VM."""
        if isinstance(script, (str, Path)) and os.path.isfile(str(script)):
            with open(script) as f:
                script = f.read()
        payload = {"script": script, "timeout": timeout}
        if working_dir:
            payload["working_dir"] = working_dir
        try:
            resp = self._post_json("/run_bash_script", payload)
            return resp.json() if resp.status_code == 200 else {"error": resp.text}
        except Exception as e:
            return {"error": str(e)}

    def upload(self, local_path: Union[str, Path], remote_path: str) -> None:
        """Upload a local file to the VM."""
        local_path = Path(local_path)
        if not local_path.exists():
            raise EnvError(f"Local file not found: {local_path}")
        with open(local_path, "rb") as f:
            resp = requests.post(
                f"{self._base_url}/setup/upload",
                files={"file_data": (local_path.name, f)},
                data={"file_path": remote_path},
                timeout=120,
                proxies=self._NO_PROXY["proxies"],
            )
        if resp.status_code != 200:
            raise EnvError(f"Upload failed ({resp.status_code}): {resp.text}")

    def download(self, remote_path: str) -> Optional[bytes]:
        """Download a file from the VM. Uses form-data POST (not JSON)."""
        try:
            resp = requests.post(
                f"{self._base_url}/file",
                data={"file_path": remote_path},
                timeout=120,
                proxies=self._NO_PROXY["proxies"],
            )
            return resp.content if resp.status_code == 200 else None
        except Exception:
            return None

    def launch(self, command: str) -> None:
        """Launch an application on the VM (non-blocking)."""
        self._post_json(
            "/setup/launch",
            {"command": command.split(), "shell": False},
            timeout=30,
        )

    def get_screen_size(self) -> dict:
        """Get VM screen dimensions."""
        try:
            resp = self._post_json("/screen_size", {}, timeout=10)
            return resp.json() if resp.status_code == 200 else {}
        except Exception:
            return {}

    def get_accessibility_tree(self) -> Optional[str]:
        """Get VM accessibility tree."""
        try:
            resp = self._get("/accessibility")
            return resp.text if resp.status_code == 200 else None
        except Exception:
            return None

    def get_directory_tree(self, path: str) -> dict:
        """List directory contents on the VM."""
        try:
            resp = self._post_json("/list_directory", {"path": path}, timeout=30)
            return resp.json() if resp.status_code == 200 else {}
        except Exception:
            return {}

    # --- Lifecycle ---

    def save_config(self, config_path: Union[str, Path]) -> None:
        """Save connection config to JSON for subagents."""
        self.config.save(config_path)

    def close(self) -> None:
        """Stop SSH tunnel. Does NOT delete the VM (use delete_instance for that)."""
        if self._closed:
            return
        self._closed = True
        if self._gateway:
            self._gateway.stop_tunnel()

    @staticmethod
    def delete_instance(config_path: Union[str, Path], env_vars: dict = None) -> None:
        """Delete VM instance using saved config. Used by orchestrator for cleanup."""
        config = EnvConfig.load(config_path)
        if not config.instance_id:
            logger.warning("No instance_id in config — skipping VM cleanup")
            return

        if env_vars is None:
            env_vars = _collect_aliyun_env_vars()

        if config.ssh_gateway:
            # Legacy: SSH gateway mode
            gw = SSHGateway(**config.ssh_gateway)
            gw.delete_vm(config.instance_id, env_vars=env_vars)
        else:
            # Default: local execution
            executor = LocalExecutor(**_default_local_executor())
            executor.delete_vm(config.instance_id, env_vars=env_vars)

    def __enter__(self) -> "Env":
        return self

    def __exit__(self, *args) -> None:
        self.close()
