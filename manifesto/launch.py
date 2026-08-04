"""Build the per-pod shell script that prepares the environment and starts vLLM."""

from __future__ import annotations

import json
import shlex
from typing import Any

from .dp_ports import RolePorts
from .parallelism import parallel_layout
from .spec import DeploymentSpec, RoleSpec


def _flag_name(name: str) -> str:
    if "." in name:
        return "--" + name
    return "--" + name.replace("_", "-")


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _format_arg(name: str, value: Any) -> list[str]:
    if value is None:
        return []
    flag = _flag_name(name)
    if "." in name:
        return [f"{flag}={shlex.quote(_format_value(value))}"]
    if isinstance(value, bool):
        return [flag] if value else []
    if isinstance(value, (dict, list)):
        return [flag, shlex.quote(_format_value(value))]
    return [flag, shlex.quote(str(value))]


def _command_lines(parts: list[str | list[str]], *, indent: str = "") -> list[str]:
    if not parts:
        return []
    rendered = [" ".join(part) if isinstance(part, list) else part for part in parts]
    lines = [f"{indent}{rendered[0]} \\"]
    lines.extend(f"{indent}  {part} \\" for part in rendered[1:-1])
    lines.append(f"{indent}  {rendered[-1]}")
    return lines


def build_launch_script(
    spec: DeploymentSpec,
    role: RoleSpec,
    ports: RolePorts,
    *,
    log_dir: str | None,
    trace_dir: str | None = None,
    vllm_env: str | None,
    persistent_cache: bool = False,
    vllm_args: dict[str, Any] | None = None,
    external_dp: bool = False,
    multi_port_external_dp: bool = False,
    distributed_dp: bool = False,
    vllm_raw_args: list[str] | None = None,
) -> str:
    layout = parallel_layout(role)
    cleanup_cache = persistent_cache and spec.cache.cleanup_on_crash
    lines = ["set -euo pipefail"]
    if log_dir:
        lines += [
            f"LOG_DIR={shlex.quote(log_dir)}",
            'mkdir -p "$LOG_DIR"',
            'LOG_FILE="$LOG_DIR/${HOSTNAME}_$(date +%Y%m%d-%H%M%S).log"',
            'exec > >(tee -a "$LOG_FILE") 2>&1',
            'echo "=== Pod $HOSTNAME started at $(date -Iseconds) ==="',
            "",
        ]
    if trace_dir:
        lines += [
            f"mkdir -p {shlex.quote(trace_dir)}",
            "",
        ]
    if cleanup_cache:
        lines += [
            'CRASH_MARKER="${VLLM_CACHE_ROOT}/.manifesto-running-${HOSTNAME}-${MANIFESTO_POD_UID}"',
            "cleanup_compile_caches() {",
            "  echo '=== Clearing JIT and compilation caches after crash ==='",
            '  if [ -n "${VLLM_CACHE_ROOT:-}" ] && [ -d "$VLLM_CACHE_ROOT" ]; then',
            '    find "$VLLM_CACHE_ROOT" -type d -name torch_compile_cache -prune -exec rm -rf -- {} + 2>/dev/null || true',
            "  fi",
            "  for CACHE_PATH in \\",
            '    "${FLASHINFER_CACHE_DIR:-}" \\',
            '    "${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-}" \\',
            '    "${TRITON_CACHE_DIR:-}" \\',
            '    "${TORCHINDUCTOR_CACHE_DIR:-}" \\',
            '    "${TILELANG_CACHE_DIR:-}"',
            "  do",
            '    case "$CACHE_PATH" in ""|/) continue ;; esac',
            '    rm -rf -- "$CACHE_PATH"',
            "  done",
            "}",
            "on_exit() {",
            "  STATUS=$?",
            "  trap - EXIT",
            "  set +e",
            '  if [ "$STATUS" -ne 0 ]; then',
            "    cleanup_compile_caches",
            "  fi",
            '  rm -f -- "$CRASH_MARKER"',
            '  exit "$STATUS"',
            "}",
            'mkdir -p "$VLLM_CACHE_ROOT"',
            'if [ -e "$CRASH_MARKER" ]; then',
            "  echo '=== Previous container terminated without exiting; treating it as a crash ==='",
            "  cleanup_compile_caches",
            "fi",
            'touch "$CRASH_MARKER"',
            "trap on_exit EXIT",
            "",
        ]
    if vllm_env:
        lines += [
            'if [ ! -d "${MANIFESTO_VLLM_ENV}" ]; then',
            '  echo "Error: vllm-envs worktree not found at ${MANIFESTO_VLLM_ENV}" >&2',
            "  exit 1",
            "fi",
            'if [ ! -f "${MANIFESTO_VLLM_ENV}/.venv/bin/activate" ]; then',
            '  echo "Error: vllm-envs environment is incomplete: ${MANIFESTO_VLLM_ENV}/.venv/bin/activate is missing" >&2',
            "  exit 1",
            "fi",
            'echo "Using vllm-envs worktree at ${MANIFESTO_VLLM_ENV}"',
            'source "${MANIFESTO_VLLM_ENV}/.venv/bin/activate"',
            "",
        ]
    else:
        lines += [
            "if [ -f /opt/vllm/bin/activate ]; then",
            "  source /opt/vllm/bin/activate",
            "fi",
            "",
        ]
    hooks = [*spec.runtime.pre_launch, *role.pre_launch]
    if hooks:
        lines += [
            "echo '=== Running pre-launch hooks ==='",
            *hooks,
            "",
        ]

    if role.parallelism.dp_enabled:
        lines += [
            f"DP_SIZE_LOCAL={layout.dp_local_size}",
            f"DP_SIZE={layout.dp_world_size}",
        ]
        if not distributed_dp and role.lws.size > 1:
            lines.append("START_RANK=$(( LWS_WORKER_INDEX * DP_SIZE_LOCAL ))")
        elif not distributed_dp:
            lines.append("START_RANK=0")
    if layout.cross_node_tp:
        lines += ["HEADLESS_ARGS=()"]
        if distributed_dp and external_dp:
            lines += [
                f"TP_NODES={layout.tp_node_count}",
                'if (( LWS_WORKER_INDEX % TP_NODES != 0 )); then',
            ]
        else:
            lines.append('if [ "$LWS_WORKER_INDEX" -gt 0 ]; then')
        lines += ["  HEADLESS_ARGS=(--headless)", "fi"]
    single_rank = not role.parallelism.dp_enabled or distributed_dp

    base_args: list[str | list[str]] = [
        "vllm",
        "serve",
        shlex.quote(spec.model.id),
        [
            "--port",
            str(ports.backend[0])
            if multi_port_external_dp or single_rank
            else "$PORT",
        ],
        ["--tensor-parallel-size", str(layout.tp_world_size)],
    ]
    if not multi_port_external_dp:
        device_ids = (
            ",".join(str(index) for index in range(layout.tp_local_size))
            if single_rank
            else "$GPUS"
        )
        base_args[3:3] = [["--device-ids", device_ids]]
    if role.parallelism.ep:
        base_args.append("--enable-expert-parallel")
    if layout.cross_node_tp:
        base_args += [
            ["--nnodes", str(role.lws.size)],
            ["--node-rank", "$LWS_WORKER_INDEX"],
            ["--master-addr", '"${LWS_LEADER_ADDRESS}"'],
            '"${HEADLESS_ARGS[@]}"',
        ]
    if distributed_dp:
        base_args += [
            ["--data-parallel-size", "$DP_SIZE"],
            ["--data-parallel-size-local", "1"],
            ["--data-parallel-address", '"${LWS_LEADER_ADDRESS}"'],
            ["--data-parallel-rpc-port", "5555"],
        ]
        if external_dp:
            base_args.append("--data-parallel-external-lb")
    elif multi_port_external_dp:
        dp_address = "${LWS_LEADER_ADDRESS}" if role.lws.size > 1 else "127.0.0.1"
        base_args += [
            ["--data-parallel-size", "$DP_SIZE"],
            ["--data-parallel-start-rank", "$START_RANK"],
            ["--data-parallel-size-local", "$DP_SIZE_LOCAL"],
            ["--data-parallel-address", dp_address],
            ["--data-parallel-rpc-port", "5555"],
            "--data-parallel-multi-port-external-lb",
            ["--data-parallel-supervisor-port", "8100"],
        ]
    elif role.parallelism.dp_enabled:
        dp_address = "${LWS_LEADER_ADDRESS}" if role.lws.size > 1 else "127.0.0.1"
        base_args += [
            ["--data-parallel-size", "$DP_SIZE"],
            ["--data-parallel-rank", "$RANK"],
            ["--data-parallel-size-local", "1"],
            ["--data-parallel-address", dp_address],
            ["--data-parallel-rpc-port", "5555"],
        ]
    if role.kv_transfer_config:
        base_args.append(["--kv_transfer_config", shlex.quote(json.dumps(role.kv_transfer_config, separators=(",", ":")))])
    if spec.model.served_name:
        base_args.append(["--served-model-name", shlex.quote(spec.model.served_name)])
    for name, value in (vllm_args or role.vllm_args).items():
        if arg := _format_arg(name, value):
            base_args.append(arg)
    base_args.extend(vllm_raw_args if vllm_raw_args is not None else role.vllm_raw_args)

    if single_rank:
        if lines[-1]:
            lines.append("")
        lines += _command_lines([*(() if cleanup_cache else ("exec",)), *base_args])
        return "\n".join(lines)

    if multi_port_external_dp:
        lines.append("")
        if persistent_cache:
            lines += [
                f"FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${{FLASH_ATTENTION_CUTE_DSL_CACHE_DIR}}/{role.name} \\",
                f"TILELANG_CACHE_DIR=${{TILELANG_CACHE_DIR}}/{role.name} \\",
            ]
        lines += _command_lines([*(() if cleanup_cache else ("exec",)), *base_args])
        return "\n".join(lines)

    lines += [
        "",
        "for R in $(seq 0 $((DP_SIZE_LOCAL - 1))); do",
        f"  GPU_START=$((R * {layout.tp_local_size}))",
        f"  GPUS=$(seq -s, $GPU_START $((GPU_START + {layout.tp_local_size} - 1)))",
        "  RANK=$((START_RANK + R))",
        f"  PORTS=({' '.join(str(port) for port in ports.backend)})",
        "  PORT=${PORTS[$R]}",
    ]

    if persistent_cache:
        lines += [
            "  VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}/rank${RANK} \\",
            "  FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR}/rank${RANK} \\",
            f"  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${{FLASH_ATTENTION_CUTE_DSL_CACHE_DIR}}/{role.name}_rank${{RANK}} \\",
            f"  TILELANG_CACHE_DIR=${{TILELANG_CACHE_DIR}}/{role.name}_rank${{RANK}} \\",
        ]
    lines += [
        *_command_lines([*base_args, "&"], indent="  "),
        "done",
        "",
        "wait -n",
        "kill $(jobs -p) 2>/dev/null || true",
        "exit 1",
    ]
    return "\n".join(lines)
