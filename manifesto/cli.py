"""Command-line entrypoints for rendering and managing Manifesto deployments."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from .cluster import load_cluster
from .e2e import E2E_IMAGE, e2e
from .instance import Instance
from .overrides import load_routing_profile, load_spec_data
from .render import render
from .resolve import resolve_role
from .spec import load_spec
from .workflow import (
    COMPLETION_KUBECTL_TIMEOUT,
    RuntimeConfig,
    WorkflowError,
    apply_runtime_overrides,
    apply_file,
    bootstrap,
    catalog_entries,
    config_home,
    delete_file,
    deploy,
    diff_file,
    kubectl_limits,
    load_cluster_with_overrides,
    load_dotenv,
    load_runtime_cluster,
    ready,
    render_bootstrap_manifest,
    render_manifest,
    render_to_file,
    resolve_cluster,
    resolve_model,
    resolve_routing,
    resolve_user,
    servers,
    stop,
)
from .workload import load_workload, render_workload, workload_settings


def _render(args: argparse.Namespace, *, routing_only: bool = False) -> int:
    config = RuntimeConfig.from_args(args)
    sys.stdout.write(render_manifest(args, config, routing_only=routing_only))
    return 0


def _render_workload(args: argparse.Namespace) -> int:
    workload = load_workload(args.spec)
    settings = (
        workload_settings(load_cluster(args.cluster)) if args.cluster else None
    )
    yaml.safe_dump_all(
        render_workload(
            workload,
            settings=settings,
            accelerator=args.accelerator,
        ),
        sys.stdout,
        sort_keys=False,
        explicit_start=True,
    )
    return 0


def _explain(args: argparse.Namespace) -> int:
    """Print resolution provenance without adding metadata to rendered YAML."""

    config = RuntimeConfig.from_args(args)
    cluster = load_runtime_cluster(config, args)
    spec = load_spec(resolve_model(args.spec), cluster)
    apply_runtime_overrides(spec, args, config)
    instance = Instance(user=config.user, release=spec.release)
    roles = []
    for role in spec.roles:
        resolved = resolve_role(spec, instance, cluster, role)
        environment = dict(sorted(resolved.env_provenance.items()))
        environment.update(
            {
                contribution.name: contribution.source
                for contribution in resolved.features.field_ref_env
            }
        )
        roles.append(
            {
                "name": role.name,
                "workload": str(resolved.features.workload_kind),
                "features": sorted(str(feature) for feature in resolved.features.enabled),
                "backends": sorted(resolved.features.backends),
                "fabric_profile": resolved.fabric_profile,
                "environment": environment,
                "resource_claims": resolved.resource_claims,
            }
        )
    sys.stdout.write(yaml.safe_dump({"roles": roles}, sort_keys=False))
    return 0


def _add_gpu_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpu", "--accelerator", dest="accelerator")


def _duration_minutes(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([mh]?)", value.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError("use a positive duration such as 15m or 2h")
    amount = int(match.group(1))
    return amount * 60 if match.group(2) == "h" else amount


def _add_context_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--context", help="kubectl context to use")


def _add_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("spec")
    _add_gpu_arg(parser)
    _add_context_arg(parser)
    parser.add_argument("--cluster")
    parser.add_argument("--namespace")
    parser.add_argument("--user")
    parser.add_argument("--routing-profile")
    parser.add_argument("--vllm-env")
    parser.add_argument("--user-root")
    parser.add_argument("--log-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--pre-launch", action="append", default=[])
    idle_group = parser.add_mutually_exclusive_group()
    idle_group.add_argument(
        "--idle-timeout",
        dest="idle_timeout_minutes",
        type=_duration_minutes,
        metavar="DURATION",
        help="override idle shutdown timeout (for example 15m or 2h)",
    )
    idle_group.add_argument(
        "--no-idle-shutdown",
        action="store_true",
        help="disable automatic idle shutdown",
    )


def _render_file(args: argparse.Namespace) -> int:
    print(render_to_file(args))
    return 0


def _render_bootstrap(args: argparse.Namespace) -> int:
    sys.stdout.write(render_bootstrap_manifest(args))
    return 0


def _add_file_args(parser: argparse.ArgumentParser) -> None:
    _add_context_arg(parser)
    parser.add_argument("--namespace")
    parser.add_argument("--user")
    parser.add_argument("--output")


def _add_cluster_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cluster")
    _add_gpu_arg(parser)
    parser.add_argument("--user")
    parser.add_argument("--user-root")
    parser.add_argument("--log-root")
    parser.add_argument("--cache-root")


def _add_ready_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("spec")
    _add_context_arg(parser)
    parser.add_argument("--cluster")
    parser.add_argument("--namespace")
    parser.add_argument("--user")
    parser.add_argument("--gateway-timeout", type=int, default=120)


def _completion(args: argparse.Namespace) -> int:
    scripts = {"bash": _BASH_COMPLETION, "zsh": _ZSH_COMPLETION, "fish": _FISH_COMPLETION}
    sys.stdout.write(scripts[args.shell])
    return 0


def _config_home(_args: argparse.Namespace) -> int:
    print(config_home())
    return 0


def _config_list(args: argparse.Namespace) -> int:
    entries = catalog_entries(args.catalog)
    if args.output == "name":
        for entry in entries:
            print(entry.name)
        return 0
    if args.output == "json":
        print(
            json.dumps(
                [
                    {
                        "name": entry.name,
                        "source": entry.source,
                        "path": str(entry.path),
                        "shadows": str(entry.shadows) if entry.shadows else None,
                    }
                    for entry in entries
                ],
                indent=2,
            )
        )
        return 0

    rows = [("NAME", "SOURCE", "PATH")]
    rows.extend((entry.name, entry.source, str(entry.path)) for entry in entries)
    widths = [max(len(row[index]) for row in rows) for index in range(3)]
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip())
    return 0


def _config_resolve(args: argparse.Namespace) -> int:
    if args.catalog == "models":
        resolved = resolve_model(args.name)
    elif args.catalog == "clusters":
        resolved = resolve_cluster(args.name)
    else:
        resolved = resolve_routing(args.name)
    if not Path(resolved).is_file():
        raise WorkflowError(
            f"Unknown {_catalog_label(args.catalog)} config {args.name!r}. "
            f"Run 'manifesto config list {args.catalog}' to see available names.",
            code=2,
        )
    print(Path(resolved).resolve())
    return 0


def _config_name(value: str) -> str:
    path = Path(value)
    return path.with_suffix("").as_posix() if path.suffix.lower() in {".yaml", ".yml"} else value


def _catalog_label(catalog: str) -> str:
    return {"models": "model", "clusters": "cluster", "routing": "routing profile"}[catalog]


def _user_config_path(catalog: str, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise WorkflowError(f"Config name must be a relative catalog name: {name!r}", code=2)
    if relative.suffix.lower() not in {".yaml", ".yml"}:
        relative = Path(f"{relative}.yaml")
    return config_home() / catalog / relative


def _validate_config_file(catalog: str, path: Path) -> None:
    if catalog == "models":
        load_spec(path)
    elif catalog == "clusters":
        load_cluster(path)
    else:
        load_routing_profile(path)


def _copy_model_with_parents(source: Path, destination: Path, *, child: bool = True) -> None:
    with source.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"spec must be a YAML mapping: {source}")
    parent_ref = data.get("extends")
    if parent_ref is not None:
        if not isinstance(parent_ref, str):
            raise ValueError(f"extends must be a relative path string: {source}")
        parent_path = Path(parent_ref)
        if parent_path.is_absolute() or ".." in parent_path.parts:
            raise ValueError(f"extends must stay within the model catalog: {source}")
        parent_source = (source.parent / parent_ref).resolve()
        parent_destination = destination.parent / parent_ref
        if not parent_destination.exists():
            _copy_model_with_parents(parent_source, parent_destination, child=False)
    if child or not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _config_edit(args: argparse.Namespace) -> int:
    name = _config_name(args.name)
    entries = {entry.name: entry for entry in catalog_entries(args.catalog)}
    entry = entries.get(name)
    destination = _user_config_path(args.catalog, args.name)

    if entry is not None and entry.source == "user":
        destination = entry.path
    elif entry is not None:
        _validate_config_file(args.catalog, entry.path)
        if args.catalog == "models":
            _copy_model_with_parents(entry.path, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry.path, destination)
        print(f"Created user config from bundled entry: {destination}")
    elif not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch()
        print(f"Created new user config: {destination}")

    editor = os.environ.get("EDITOR", "vi")
    result = subprocess.run([*shlex.split(editor), str(destination)])
    if result.returncode:
        return result.returncode
    _validate_config_file(args.catalog, destination)
    print(f"Valid {_catalog_label(args.catalog)} config: {destination}")
    return 0


def _portable_config(catalog: str, source: Path) -> dict:
    _validate_config_file(catalog, source)
    if catalog == "models":
        return load_spec_data(source)
    with source.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a YAML mapping: {source}")
    return data


def _write_config(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise WorkflowError(f"Refusing to overwrite {path}; pass --force to replace it.", code=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _config_import(args: argparse.Namespace) -> int:
    source = Path(args.file).expanduser()
    data = _portable_config(args.catalog, source)
    destination = _user_config_path(args.catalog, args.name or source.stem)
    _write_config(destination, yaml.safe_dump(data, sort_keys=False), force=args.force)
    print(destination)
    return 0


def _config_export(args: argparse.Namespace) -> int:
    if args.catalog == "models":
        resolved = resolve_model(args.name)
    elif args.catalog == "clusters":
        resolved = resolve_cluster(args.name)
    else:
        resolved = resolve_routing(args.name)
    source = Path(resolved)
    if not source.is_file():
        raise WorkflowError(
            f"Unknown {_catalog_label(args.catalog)} config {args.name!r}. "
            f"Run 'manifesto config list {args.catalog}' to see available names.",
            code=2,
        )
    content = yaml.safe_dump(_portable_config(args.catalog, source), sort_keys=False)
    if args.output:
        destination = Path(args.output).expanduser()
        _write_config(destination, content, force=args.force)
        print(destination)
    else:
        sys.stdout.write(content)
    return 0


def _config_validate(args: argparse.Namespace) -> int:
    load_dotenv()
    cluster_path = resolve_cluster(args.cluster)
    cluster = load_cluster_with_overrides(cluster_path, args)
    if args.spec is None:
        print(f"Valid cluster: {Path(cluster_path).resolve()}")
        return 0

    model_path = resolve_model(args.spec)
    spec = load_spec(model_path, cluster)
    if args.accelerator:
        spec.accelerator = args.accelerator
    objects = render(spec, user=resolve_user(args.user), cluster=cluster)
    print(f"Valid cluster: {Path(cluster_path).resolve()}")
    print(f"Valid model:   {Path(model_path).resolve()}")
    print(f"Renders:       {len(objects)} Kubernetes objects")
    return 0


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["Invalid configuration:"]
    for error in exc.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in error["loc"])
        prefix = f"{location}: " if location else ""
        lines.append(f"  {prefix}{error['msg']}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manifesto")
    sub = parser.add_subparsers(dest="command", required=True)

    render_parser = sub.add_parser("render", help="render Kubernetes manifests")
    render_sub = render_parser.add_subparsers(dest="render_command", required=True)

    render_manifest_parser = render_sub.add_parser(
        "manifest", help="render a full Kubernetes manifest to stdout"
    )
    _add_render_args(render_manifest_parser)
    render_manifest_parser.set_defaults(func=lambda args: _render(args, routing_only=False))

    render_routing_parser = render_sub.add_parser(
        "routing", help="render routing-only Kubernetes YAML to stdout"
    )
    _add_render_args(render_routing_parser)
    render_routing_parser.set_defaults(func=lambda args: _render(args, routing_only=True))

    render_file_parser = render_sub.add_parser(
        "file", help="render a full manifest to the workflow file"
    )
    _add_render_args(render_file_parser)
    render_file_parser.add_argument("-o", "--output")
    render_file_parser.set_defaults(func=_render_file)

    render_bootstrap_parser = render_sub.add_parser(
        "bootstrap", help="render namespace prerequisites to stdout"
    )
    render_bootstrap_parser.add_argument("--cluster")
    _add_context_arg(render_bootstrap_parser)
    render_bootstrap_parser.add_argument("--namespace")
    render_bootstrap_parser.set_defaults(func=_render_bootstrap)

    render_workload_parser = render_sub.add_parser(
        "workload",
        help="render a controller-neutral Job, Deployment, or LeaderWorkerSet",
    )
    render_workload_parser.add_argument("spec")
    render_workload_parser.add_argument("--cluster")
    render_workload_parser.add_argument("--accelerator")
    render_workload_parser.set_defaults(func=_render_workload)

    explain_parser = sub.add_parser(
        "explain", help="explain resolved role features and their contributions"
    )
    _add_render_args(explain_parser)
    explain_parser.set_defaults(func=_explain)

    file_parser = sub.add_parser("file", help="manage the saved workflow manifest")
    file_sub = file_parser.add_subparsers(dest="file_command", required=True)

    diff_parser = file_sub.add_parser("diff", help="kubectl diff the workflow file")
    _add_file_args(diff_parser)
    diff_parser.set_defaults(func=diff_file)

    apply_parser = file_sub.add_parser("apply", help="kubectl apply the workflow file")
    _add_file_args(apply_parser)
    apply_parser.set_defaults(func=apply_file)

    delete_parser = file_sub.add_parser("delete", help="kubectl delete objects from the workflow file")
    _add_file_args(delete_parser)
    delete_parser.add_argument("--now", action="store_true")
    delete_parser.set_defaults(func=delete_file)

    deploy_parser = sub.add_parser(
        "deploy",
        help="deploy a model or specialized Kubernetes resources",
        usage=(
            "%(prog)s SPEC [OPTIONS]\n"
            "       %(prog)s routing SPEC [OPTIONS]\n"
            "       %(prog)s bootstrap [OPTIONS]"
        ),
        description=(
            "Deploy a model with 'manifesto deploy SPEC'. "
            "Use the routing and bootstrap subcommands for specialized updates."
        ),
    )
    deploy_sub = deploy_parser.add_subparsers(dest="deploy_command", required=True)

    deploy_manifest_parser = deploy_sub.add_parser(
        "manifest", help="legacy compatibility alias for deploying a model"
    )
    _add_render_args(deploy_manifest_parser)
    deploy_manifest_parser.set_defaults(func=lambda args: deploy(args, routing_only=False))

    deploy_routing_parser = deploy_sub.add_parser(
        "routing", help="render and apply routing objects only"
    )
    _add_render_args(deploy_routing_parser)
    deploy_routing_parser.set_defaults(func=lambda args: deploy(args, routing_only=True))

    deploy_bootstrap_parser = deploy_sub.add_parser(
        "bootstrap", help="apply namespace prerequisites declared by the cluster profile"
    )
    deploy_bootstrap_parser.add_argument("--cluster")
    _add_context_arg(deploy_bootstrap_parser)
    deploy_bootstrap_parser.add_argument("--namespace")
    deploy_bootstrap_parser.set_defaults(func=bootstrap)

    servers_parser = sub.add_parser("servers", help="list live Manifesto servers in the namespace")
    _add_context_arg(servers_parser)
    servers_parser.add_argument("--namespace")
    servers_parser.add_argument("--instance")
    servers_parser.add_argument("--output", choices=["table", "name", "json"], default="table")
    servers_parser.set_defaults(func=servers)

    stop_parser = sub.add_parser("stop", help="discover and delete a live Manifesto server")
    stop_parser.add_argument("spec", nargs="?")
    stop_parser.add_argument("--instance")
    _add_context_arg(stop_parser)
    stop_parser.add_argument("--namespace")
    stop_parser.add_argument("--user")
    # Retain legacy render flags as accepted no-ops now that stop discovers live state.
    for legacy_flag in ("--cluster", "--user-root", "--log-root", "--cache-root"):
        stop_parser.add_argument(legacy_flag, help=argparse.SUPPRESS)
    stop_parser.add_argument("--pre-launch", action="append", default=[], help=argparse.SUPPRESS)
    stop_parser.add_argument("--now", action="store_true")
    stop_parser.set_defaults(func=stop)

    ready_parser = sub.add_parser("ready", help="wait for model pods and gateway readiness")
    _add_ready_args(ready_parser)
    ready_parser.set_defaults(func=ready)

    test_parser = sub.add_parser("test", help="run integration tests")
    test_sub = test_parser.add_subparsers(dest="test_command", required=True)
    e2e_parser = test_sub.add_parser(
        "e2e", help="run a fresh-namespace dev and inference integration test"
    )
    e2e_parser.add_argument("spec")
    e2e_parser.add_argument("--cluster")
    _add_context_arg(e2e_parser)
    e2e_parser.add_argument("--namespace")
    e2e_parser.add_argument("--user")
    e2e_parser.add_argument("--routing-profile")
    _add_gpu_arg(e2e_parser)
    e2e_parser.add_argument("--image", default=E2E_IMAGE)
    e2e_parser.add_argument("--timeout", type=int, default=300)
    e2e_parser.add_argument("--gateway-timeout", type=int, default=120)
    e2e_parser.add_argument("--vllm-env")
    e2e_parser.add_argument("--keep-namespace", action="store_true")
    e2e_parser.set_defaults(func=e2e)

    completion_parser = sub.add_parser("completion", help="print shell completion setup")
    completion_parser.add_argument("shell", choices=["bash", "zsh", "fish"])
    completion_parser.set_defaults(func=_completion)

    config_parser = sub.add_parser("config", help="manage model and cluster configs")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)

    config_home_parser = config_sub.add_parser("home", help="print the user config directory")
    config_home_parser.set_defaults(func=_config_home)

    config_list_parser = config_sub.add_parser("list", help="list effective catalog entries")
    config_list_parser.add_argument("catalog", choices=["models", "clusters", "routing"])
    config_list_parser.add_argument("--output", choices=["table", "name", "json"], default="table")
    config_list_parser.set_defaults(func=_config_list)

    config_resolve_parser = config_sub.add_parser("resolve", help="resolve a catalog name to a file")
    config_resolve_parser.add_argument("catalog", choices=["models", "clusters", "routing"])
    config_resolve_parser.add_argument("name")
    config_resolve_parser.set_defaults(func=_config_resolve)

    config_edit_parser = config_sub.add_parser("edit", help="open a user config in $EDITOR")
    config_edit_parser.add_argument("catalog", choices=["models", "clusters", "routing"])
    config_edit_parser.add_argument("name")
    config_edit_parser.set_defaults(func=_config_edit)

    config_import_parser = config_sub.add_parser("import", help="validate and import a portable config")
    config_import_parser.add_argument("catalog", choices=["models", "clusters", "routing"])
    config_import_parser.add_argument("file")
    config_import_parser.add_argument("--name", help="destination catalog name (defaults to the filename)")
    config_import_parser.add_argument("--force", action="store_true")
    config_import_parser.set_defaults(func=_config_import)

    config_export_parser = config_sub.add_parser("export", help="export a portable flattened config")
    config_export_parser.add_argument("catalog", choices=["models", "clusters", "routing"])
    config_export_parser.add_argument("name")
    config_export_parser.add_argument("-o", "--output")
    config_export_parser.add_argument("--force", action="store_true")
    config_export_parser.set_defaults(func=_config_export)

    config_validate_parser = config_sub.add_parser(
        "validate", help="validate a cluster or a model and cluster together"
    )
    config_validate_parser.add_argument("spec", nargs="?")
    _add_cluster_args(config_validate_parser)
    config_validate_parser.set_defaults(func=_config_validate)

    return parser


_DEPLOY_SUBCOMMANDS = {"manifest", "routing", "bootstrap"}


def _normalize_deploy_argv(argv: list[str]) -> list[str]:
    """Map the noun-free deploy form onto the legacy manifest subcommand."""

    if (
        len(argv) > 1
        and argv[0] == "deploy"
        and argv[1] not in _DEPLOY_SUBCOMMANDS
        and argv[1] not in {"-h", "--help"}
    ):
        return ["deploy", "manifest", *argv[1:]]
    return argv


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if argv and argv[0] == "__complete":
        for candidate in _completion_candidates(parser, argv[1:]):
            print(candidate)
        return 0

    try:
        argv = _normalize_deploy_argv(argv)
        args = parser.parse_args(argv)
        return args.func(args)
    except WorkflowError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    except ValidationError as exc:
        print(_format_validation_error(exc), file=sys.stderr)
        return 2
    except (ValueError, yaml.YAMLError) as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"File not found: {exc.filename}", file=sys.stderr)
        return 2


def _completion_context(
    parser: argparse.ArgumentParser, tokens: list[str]
) -> tuple[argparse.ArgumentParser, tuple[str, ...], int]:
    """Follow completed command words through the argparse tree."""

    path: list[str] = []
    positional_count = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        subcommands = next(
            (action for action in parser._actions if isinstance(action, argparse._SubParsersAction)),
            None,
        )
        if subcommands is not None and token in subcommands.choices:
            parser = subcommands.choices[token]
            path.append(token)
            positional_count = 0
            index += 1
            continue

        option_name = token.split("=", 1)[0]
        option = parser._option_string_actions.get(option_name)
        if option is not None:
            if "=" not in token and option.nargs != 0:
                index += 2
            else:
                index += 1
            continue
        if token != "--":
            positional_count += 1
        index += 1
    return parser, tuple(path), positional_count


def _completion_files(prefix: str, *, yaml_only: bool = False) -> list[str]:
    expanded = os.path.expanduser(prefix)
    path = Path(expanded)
    directory = path.parent if path.name else path
    display_directory = Path(prefix).parent if Path(prefix).name else Path(prefix)
    name_prefix = path.name
    try:
        children = sorted(directory.iterdir(), key=lambda child: child.name)
    except OSError:
        return []

    candidates = []
    for child in children:
        if not child.name.startswith(name_prefix):
            continue
        if yaml_only and not child.is_dir() and child.suffix.lower() not in {".yaml", ".yml"}:
            continue
        candidate = str(display_directory / child.name)
        if candidate.startswith("./"):
            candidate = candidate[2:]
        if child.is_dir():
            candidate += "/"
        candidates.append(candidate)
    return candidates


def _completion_argument(tokens: list[str], *option_names: str) -> str | None:
    """Return the last value supplied for one of the named options."""

    value = None
    for index, token in enumerate(tokens):
        option_name, separator, inline_value = token.partition("=")
        if option_name not in option_names:
            continue
        if separator:
            value = inline_value
        elif index + 1 < len(tokens):
            value = tokens[index + 1]
    return value


def _completion_catalog(catalog: str, prefix: str) -> list[str]:
    return [entry.name for entry in catalog_entries(catalog)] + _completion_files(
        prefix, yaml_only=True
    )


def _completion_accelerators(tokens: list[str]) -> list[str]:
    cluster_name = _completion_argument(tokens, "--cluster")
    try:
        load_dotenv()
        cluster = load_cluster(resolve_cluster(cluster_name))
    except (OSError, WorkflowError, ValueError, ValidationError, yaml.YAMLError):
        return []
    return sorted(cluster.accelerators.profiles)


def _completion_option_value(
    parser: argparse.ArgumentParser, tokens: list[str], current: str
) -> tuple[argparse.Action | None, str, str]:
    if current.startswith("-") and "=" in current:
        option_name, prefix = current.split("=", 1)
        return parser._option_string_actions.get(option_name), prefix, f"{option_name}="
    if tokens:
        option = parser._option_string_actions.get(tokens[-1])
        if option is not None and option.nargs != 0:
            return option, current, ""
    return None, current, ""


def _completion_live_instances(tokens: list[str]) -> list[str]:
    namespace = _completion_argument(tokens, "--namespace")
    output = io.StringIO()
    args = argparse.Namespace(namespace=namespace, instance=None, output="name")
    try:
        # Tab completion must never stall the shell: a slow cluster yields no
        # candidates rather than a multi-second pause.
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            with kubectl_limits(timeout=COMPLETION_KUBECTL_TIMEOUT, retries=0):
                servers(args)
    except (OSError, WorkflowError, ValueError):
        return []
    return output.getvalue().splitlines()


def _completion_values(
    action: argparse.Action, path: tuple[str, ...], tokens: list[str], prefix: str
) -> list[str]:
    if action.choices is not None:
        return [str(choice) for choice in action.choices]
    if action.dest == "instance":
        return _completion_live_instances(tokens)
    if action.dest == "cluster":
        return _completion_catalog("clusters", prefix)
    if action.dest == "spec":
        return _completion_catalog("models", prefix)
    if action.dest == "routing_profile":
        return _completion_catalog("routing", prefix)
    if action.dest == "accelerator":
        return _completion_accelerators(tokens)
    if action.dest == "name" and path[-2:-1] == ("config",):
        catalog = next(
            (token for token in reversed(tokens) if token in {"models", "clusters", "routing"}),
            None,
        )
        return _completion_catalog(catalog, prefix) if catalog else []
    if action.dest in {"file", "output", "vllm_env"}:
        return _completion_files(prefix, yaml_only=action.dest == "file")
    return []


def _completion_candidates(parser: argparse.ArgumentParser, words: list[str]) -> list[str]:
    """Return newline-safe candidates for the shell adapters below."""

    direct_deploy_prefix = words[1] if len(words) == 2 and words[0] == "deploy" else None
    words = _normalize_deploy_argv(words)
    current = words[-1] if words else ""
    tokens = words[:-1] if words else []
    active, path, positional_count = _completion_context(parser, tokens)
    value_action, value_prefix, value_leader = _completion_option_value(active, tokens, current)

    candidates: list[str] = []
    if value_action is not None:
        candidates = [
            f"{value_leader}{value}"
            for value in _completion_values(value_action, path, tokens, value_prefix)
            if value.startswith(value_prefix)
        ]
    elif current.startswith("-"):
        candidates = [
            option
            for action in active._actions
            if action.help is not argparse.SUPPRESS
            for option in action.option_strings
            if option.startswith(current)
        ]
    else:
        positionals = [action for action in active._actions if not action.option_strings]
        action = positionals[positional_count] if positional_count < len(positionals) else None
        if isinstance(action, argparse._SubParsersAction):
            candidates.extend(action.choices)
        elif action is not None:
            candidates.extend(_completion_values(action, path, tokens, current))
        if current == "":
            candidates.extend(
                option
                for action in active._actions
                if action.help is not argparse.SUPPRESS
                for option in action.option_strings
            )

    if direct_deploy_prefix is not None:
        candidates.extend(
            mode
            for mode in ("bootstrap", "routing")
            if mode.startswith(direct_deploy_prefix)
        )

    return sorted(dict.fromkeys(candidate for candidate in candidates if candidate.startswith(current)))


_BASH_COMPLETION = r'''_manifesto_complete() {
  COMPREPLY=()
  while IFS= read -r candidate; do
    COMPREPLY+=("$candidate")
  done < <(manifesto __complete "${COMP_WORDS[@]:1}" 2>/dev/null)
}
complete -o filenames -F _manifesto_complete manifesto
'''


_ZSH_COMPLETION = r'''#compdef manifesto
_manifesto() {
  local -a candidates
  candidates=("${(@f)$(manifesto __complete "${words[@]:1}" 2>/dev/null)}")
  compadd -f -a candidates
}
compdef _manifesto manifesto
'''


_FISH_COMPLETION = r'''function __manifesto_complete
    set -l current (commandline -ct)
    manifesto __complete (commandline -opc)[2..-1] "$current" 2>/dev/null
end
complete -c manifesto -f -a '(__manifesto_complete)'
'''


if __name__ == "__main__":
    raise SystemExit(main())
