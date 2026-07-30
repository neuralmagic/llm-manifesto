# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
this repository.

## Preferences

- Use the `manifesto` CLI directly for rendering and model-server lifecycle
  operations.
- Keep private and operational details out of all public project surfaces. This
  includes code, tests, documentation, generated manifests, logs, benchmark
  results, screenshots, commit messages, issue titles and comments, pull request
  descriptions and comments, and review feedback.
- Never publish credentials or credential material, including API tokens,
  passwords, private keys, certificates, kubeconfigs, cloud configuration,
  registry authentication, or secret values. References to Kubernetes Secret
  names and keys are acceptable only when they are synthetic or intentionally
  public and contain no secret value.
- Do not publish real cluster or context names, API endpoints, namespaces, node
  names or addresses, internal registry locations, queue names, storage or
  resource claim names, network interfaces, user-specific paths, or similar
  environment identifiers. Keep them in private user configuration and use
  clearly synthetic placeholders such as `example.com`, RFC 5737 IP addresses,
  and example resource names.
- Before opening or updating an issue or pull request, inspect the complete diff,
  staged and untracked files, generated output, and commit messages for private
  data. Do not paste raw logs, manifests, command output, or screenshots without
  reviewing and redacting them first.
- If sensitive material is committed or posted, stop sharing it, revoke or rotate
  affected credentials, remove it from all reachable history and public text,
  and report the exposure through the appropriate private security channel.

## Repository Overview

Manifesto renders shareable Kubernetes manifests for llm-d/vLLM deployments.
It supports large GPU deployments, including aggregated and prefill/decode
topologies, without checking concrete cluster inventory into the repository.

The repository contains:

- `manifesto/` - Python renderer implementation and CLI.
- `models/` - Model deployment specs.
- `clusters/` - Cluster profiles.
- The persistent dev pod for building vLLM from source is managed by
  `manifesto dev` using the image configured in `config/images.yaml`.
- `monitoring/` - Namespace-scoped Prometheus and Grafana stack.
- `tests/` - Renderer, validation, and UX regression tests.

## Architecture

Manifesto takes a model spec and a cluster profile, then emits raw Kubernetes
objects:

- Deployment or LeaderWorkerSet model-server workloads, depending on node count.
- InferencePool and endpoint picker deployment.
- Gateway API and HTTPRoute objects.
- Per-pod monitoring sidecars.
- Instance-scoped names, labels, selectors, and cache paths.

Key components:

- **vLLM** - Model server and inference engine.
- **Inference Gateway** - Request scheduler and balancer through Gateway API
  InferencePool.
- **Kubernetes** - Infrastructure orchestrator and workload control plane.
- **LeaderWorkerSet** - Multi-host inference coordination.
- **NIXL** - Fast interconnect library for KV cache transfer.

## Common Commands

Local configuration may provide:

- `HF_TOKEN` - HuggingFace token for model access.
- `KUBECONFIG` - Path to kubeconfig.
- `MANIFESTO_CLUSTER` or `MANIFESTO_CLUSTER_MAP` - Explicit renderer cluster
  profile, or local kube context/cluster to profile mapping.
- `MANIFESTO_NAMESPACE` - Optional namespace override; defaults to the current
  kube context namespace or `default`.

Renderer workflow:

```bash
manifesto render bootstrap
manifesto deploy bootstrap
manifesto render manifest models/qwen/aggregated.yaml
manifesto render file models/deepseek-v4/1P-EP8-1D-EP8.yaml --dev
manifesto file diff
manifesto file apply
manifesto deploy models/deepseek-v4/1P-EP8-1D-EP8.yaml --dev
manifesto ready models/deepseek-v4/1P-EP8-1D-EP8.yaml
manifesto stop
manifesto stop models/deepseek-v4/1P-EP8-1D-EP8.yaml
```

Dev vLLM workflow:

```bash
manifesto dev start
manifesto dev shell
manifesto dev build
manifesto dev build-log
manifesto dev stop
```

## Key Configuration Files

- `pyproject.toml` - Python package metadata and test configuration.
- `config/images.yaml` - Central image catalog for model, llm-d, sidecar, and
  dev images.
- `clusters/example-gb200.yaml` - Synthetic GB200 profile.
- `clusters/example-h200.yaml` - Synthetic H200 profile.
- `models/qwen/aggregated.yaml` - Aggregated Qwen example.
- `models/deepseek-v4/wide-ep-base.yaml` - Shared DeepSeek V4 wide-EP base.
- `models/deepseek-v4/1P-EP8-1D-EP8.yaml` - DeepSeek V4 P/D EP8 example.
- `models/deepseek-v4/3P-EP8-1D-EP16.yaml` - DeepSeek V4 wider decode example.
- `monitoring/` - Prometheus/Grafana Helm values and dashboards.

## Development Workflow

1. Render a manifest with `manifesto render manifest` or `manifesto render file`.
2. Inspect or edit the generated YAML.
3. Apply with `manifesto file apply` or deploy directly with `manifesto deploy`.
4. Wait with `manifesto ready`.
5. Use `manifesto dev start` and `manifesto dev build` for vLLM source iteration.

## Important Notes

- Rendered objects are scoped by `{user}-{release}` so multiple users can share
  a namespace.
- Decode pods may expose multiple vLLM ports when data parallel fanout is
  enabled.
- vLLM API servers can take several minutes to start for large MoE models.
