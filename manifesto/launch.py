"""Build the per-pod shell script that prepares the environment and starts vLLM."""

from __future__ import annotations

import json
import shlex
from typing import Any

from .resolve import CACHE_DIRS, ResolvedRole
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
    resolved: ResolvedRole,
    *,
    respect_visible_devices: bool = False,
) -> str:
    layout = resolved.layout
    ports = resolved.ports
    log_dir = resolved.log_dir
    trace_dir = resolved.trace_dir
    vllm_env = resolved.vllm_env
    persistent_cache = resolved.persistent_cache
    external_dp = resolved.features.external_dp
    multi_port_external_dp = ports.rank_count > 1
    headless_workers = resolved.has_headless_nodes
    cleanup_cache = persistent_cache and spec.cache.cleanup_on_crash
    lines = ["set -euo pipefail"]
    if persistent_cache:
        # A deployment shares its cache prefix across pods. Scope writable JIT
        # caches before crash cleanup so one pod cannot remove another's files.
        for name in CACHE_DIRS:
            lines.append(f'export {name}="${{{name}}}/${{HOSTNAME}}"')
        lines.append("")
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
    hooks = [*spec.runtime.pre_launch, *role.pre_launch]
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
        if hooks:
            lines += [
                'MANIFESTO_VLLM_PYTHON="${MANIFESTO_VLLM_ENV}/.venv/bin/python"',
                "",
            ]
    else:
        lines += [
            "if [ -f /opt/vllm/bin/activate ]; then",
            "  source /opt/vllm/bin/activate",
            "fi",
            "",
        ]
        if hooks:
            lines += [
                'MANIFESTO_VLLM_EXECUTABLE="$(command -v vllm)"',
                'MANIFESTO_VLLM_PYTHON="$(command -v python3 || true)"',
                'if IFS= read -r MANIFESTO_VLLM_SHEBANG < "$MANIFESTO_VLLM_EXECUTABLE"; then',
                '  if [[ "$MANIFESTO_VLLM_SHEBANG" =~ ^\\#\\!/usr/bin/env[[:space:]]+(python([0-9]+([.][0-9]+)*)?)([[:space:]].*)?$ ]]; then',
                '    MANIFESTO_VLLM_PYTHON="$(command -v "${BASH_REMATCH[1]}" || true)"',
                '  elif [[ "$MANIFESTO_VLLM_SHEBANG" =~ ^\\#\\!([^[:space:]]*/python([0-9]+([.][0-9]+)*)?)([[:space:]].*)?$ ]]; then',
                '    MANIFESTO_VLLM_PYTHON="${BASH_REMATCH[1]}"',
                "  fi",
                "fi",
                "",
            ]
    if hooks:
        lines += [
            'if [ ! -x "$MANIFESTO_VLLM_PYTHON" ]; then',
            '  echo "Error: unable to resolve the Python interpreter for vLLM" >&2',
            "  exit 1",
            "fi",
            "export MANIFESTO_VLLM_PYTHON",
            "if ! command -v python >/dev/null 2>&1; then",
            '  python() { "$MANIFESTO_VLLM_PYTHON" "$@"; }',
            "fi",
            "",
            "echo '=== Running pre-launch hooks ==='",
            *hooks,
            "",
        ]

    if role.parallelism.dp_enabled:
        lines += [
            f"DP_SIZE_LOCAL={layout.dp_local_size}",
            f"DP_SIZE={layout.dp_world_size}",
        ]
        if layout.nodes_per_dp_rank == 1:
            lines.append(
                "START_RANK=$(( LWS_WORKER_INDEX * DP_SIZE_LOCAL ))"
                if layout.node_count > 1 else "START_RANK=0"
            )
    if headless_workers:
        api_pattern = "|".join(str(node) for node in resolved.api_nodes)
        lines += [
            "HEADLESS_ARGS=()",
            'case "$LWS_WORKER_INDEX" in',
            f"  {api_pattern}) ;;",
            "  *)",
            "    HEADLESS_ARGS=(--headless)",
        ]
        if layout.nodes_per_dp_rank == 1:
            # vLLM infers ranks from --node-rank when a DP rank spans nodes.
            # Otherwise only headless nodes need start-rank; on an API node
            # this option would switch vLLM to hybrid load balancing.
            lines.append('    HEADLESS_ARGS+=(--data-parallel-start-rank "$START_RANK")')
        lines += ["    ;;", "esac"]

    base_args: list[str | list[str]] = [
        "vllm",
        "serve",
        shlex.quote(spec.model.id),
        ["--port", str(ports.backend[0])],
        ["--tensor-parallel-size", str(layout.tp_world_size)],
    ]
    if layout.pp_world_size > 1:
        base_args.append(["--pipeline-parallel-size", str(layout.pp_world_size)])
    if not role.parallelism.dp_enabled and not respect_visible_devices:
        device_ids = ",".join(str(index) for index in range(layout.gpus_per_node))
        base_args[3:3] = [["--device-ids", device_ids]]
    if role.parallelism.ep:
        base_args.append("--enable-expert-parallel")
    if layout.nodes_per_dp_rank > 1:
        base_args += [
            ["--nnodes", str(role.lws.size)],
            ["--node-rank", "$LWS_WORKER_INDEX"],
            ["--master-addr", '"${LWS_LEADER_ADDRESS}"'],
        ]
    if headless_workers:
        base_args.append('${HEADLESS_ARGS[@]+"${HEADLESS_ARGS[@]}"}')
    if role.parallelism.dp_enabled:
        dp_address = '"${LWS_LEADER_ADDRESS}"' if layout.node_count > 1 else "127.0.0.1"
        base_args += [
            ["--data-parallel-size", "$DP_SIZE"],
            ["--data-parallel-size-local", "$DP_SIZE_LOCAL"],
            ["--data-parallel-address", dp_address],
            ["--data-parallel-rpc-port", "5555"],
        ]
        if multi_port_external_dp:
            base_args += [
                ["--data-parallel-start-rank", "$START_RANK"],
                "--data-parallel-multi-port-external-lb",
                ["--data-parallel-supervisor-port", "8100"],
            ]
        elif external_dp:
            if layout.nodes_per_dp_rank > 1:
                base_args.append("--data-parallel-external-lb")
            else:
                base_args.append(["--data-parallel-rank", "$START_RANK"])
    if role.kv_transfer_config:
        base_args.append(["--kv_transfer_config", shlex.quote(json.dumps(role.kv_transfer_config, separators=(",", ":")))])
    if spec.model.revision:
        base_args.append(["--revision", shlex.quote(spec.model.revision)])
    if spec.model.served_name:
        base_args.append(["--served-model-name", shlex.quote(spec.model.served_name)])
    for name, value in resolved.vllm_args.items():
        if arg := _format_arg(name, value):
            base_args.append(arg)
    base_args.extend(resolved.vllm_raw_args)

    if lines[-1]:
        lines.append("")
    if multi_port_external_dp and persistent_cache:
        lines += [
            f"FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${{FLASH_ATTENTION_CUTE_DSL_CACHE_DIR}}/{role.name} \\",
            f"TILELANG_CACHE_DIR=${{TILELANG_CACHE_DIR}}/{role.name} \\",
        ]
    # vLLM starts local DP engines and assigns their devices. Preserve the
    # scheduler/container GPU visibility instead of slicing it in a shell loop.
    lines += _command_lines([*(() if cleanup_cache else ("exec",)), *base_args])
    return "\n".join(lines)
