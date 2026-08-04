# vLLM Development Services

This document scopes the missing build-plane service between disposable coding
agents and persistent `vllm-envs` environments. The working name used below is
`vllm-env-service`; the name is not an API commitment.

## Recommendation

Start with one authenticated REST API that creates Kubernetes Jobs. Do not
start with a custom controller, CRDs, a permanent GPU pod, or changes to
Manifesto.

```text
OpenShell agent
    |
    | HTTPS: create, sync, run, inspect, delete
    v
vllm-env-service API -------- Kubernetes API
    |                              |
    | metadata and locks           | creates
    v                              v
small persistent state       vllm-dev-image Jobs
                                   |
                                   v
                         shared PVC: repo, worktrees,
                         .venvs, build cache, logs
                                   |
                                   v
                         Manifesto model pods
```

This gives agents a narrow interface without Kubernetes credentials and lets
GPU workers scale to zero. A warm builder pool can be added after queue and
startup measurements justify it.

## Service responsibilities

The service owns:

- a configured vLLM clone and per-developer environment roots;
- `ve new`, `ve sync`, `ve status`, `ve rm`, and cache maintenance;
- asynchronous Kubernetes Job creation and log/status collection;
- canonical workspace paths and PVC subpaths;
- serialization of mutations to a worktree;
- authentication, ownership checks, quotas, and audit records; and
- safe deletion checks for dirty or unpushed work.

It does not own:

- Claude Code or Codex process lifecycle;
- agent credentials or OpenShell policy;
- generic Kubernetes access for agents;
- Manifesto deployment lifecycle;
- GitHub pull requests; or
- model-serving health and benchmark orchestration.

## Proposed v1 API

All mutating calls accept an idempotency key and return an asynchronous
operation. Names are logical identifiers; the server derives filesystem paths
and never accepts a client-supplied absolute path.

### Workspaces

`POST /v1/workspaces`

```json
{
  "name": "fix-kernel-selection",
  "ref": "origin/main",
  "branch": "dev/fix-kernel-selection"
}
```

Runs `ve new`, creates the branch, completes the initial sync, and returns:

```json
{
  "operation_id": "op-123",
  "workspace_id": "ws-123"
}
```

`GET /v1/workspaces/{id}` returns owner, ref, branch, lifecycle state,
canonical build/serve path, safe PVC subpath, current commit, last successful
sync, and active operation.

`DELETE /v1/workspaces/{id}` runs the safe `ve rm` path. It returns a conflict
when the worktree is dirty, has unpushed commits, or has an active operation.
Force deletion is operator-only and must be audited. Detecting use by a
Manifesto deployment requires a later lease integration; until then, stopping
the deployment is an explicit operator precondition.

### Synchronization

`POST /v1/workspaces/{id}/sync`

```json
{
  "fresh_venv": false
}
```

Runs `ve sync` at the current worktree HEAD. This endpoint is the remote
replacement for the Git hook skipped inside the agent sandbox with
`VE_NO_SYNC=1`.

### Commands and tests

`POST /v1/workspaces/{id}/runs`

```json
{
  "argv": ["python", "-m", "pytest", "-q", "tests/test_selected.py"],
  "gpu_count": 1,
  "cpu": "8",
  "memory": "64Gi",
  "timeout_seconds": 1800
}
```

The executor activates the workspace `.venv`, sets the worktree as its working
directory, and invokes `argv` without an implicit shell. Bounds come from
server policy, not directly from the caller.

This endpoint is intentionally remote code execution: the worktree itself can
contain executable build and test code. Treat the executor namespace as a
security boundary. Give its service account no Secrets access, mount no host
paths, inject no agent or GitHub credentials, restrict egress, run as a
non-root UID, and enforce resource and wall-clock limits.

### Operations

`GET /v1/operations/{id}` returns queued, running, succeeded, failed, or
cancelled state plus timestamps, workspace ID, Job name, exit status, and a
bounded diagnostic summary.

`GET /v1/operations/{id}/logs` streams or paginates logs. Durable full logs may
be copied to object storage; Kubernetes pod logs alone are not a retention
contract.

`DELETE /v1/operations/{id}` requests cancellation and deletes the Job only
after recording the terminal state.

### Maintenance

Operator-only endpoints or commands expose `ve list`, `ve du`, `ve reap`, and
`ve gc --dry-run`. Destructive cache collection should require a separate
confirmed action. Maintenance is not exposed to agent identities in v1.

## Job execution model

Use one Job per create, sync, run, or delete operation:

- pin `vllm-dev-image` by digest;
- mount the shared PVC at `/mnt/shared`;
- derive developer, repository, cache, and environment paths from trusted
  server configuration;
- label the Job with operation, workspace, owner, and image digest;
- request a GPU for create/sync initially because a cache miss may compile;
- allow zero-GPU runs when explicitly supported by the selected command;
- use `activeDeadlineSeconds`, TTL cleanup, and bounded retries; and
- preserve operation metadata and logs after the Job is removed.

Avoid retrying arbitrary failed commands automatically. Create and sync may be
retried only when their filesystem operations are proven idempotent.

## State, locking, and concurrency

For v1, run one API replica with a small persistent SQL database and use
Kubernetes Leases for crash-tolerant locks. Record workspaces, ownership,
operations, requested resources, Job identity, image digest, timestamps, and
terminal results. Source and environment contents remain on the shared PVC.

Lock at two levels:

1. a repository lock for clone initialization, fetch, worktree add/remove, and
   hook installation; and
2. a workspace lock for sync, run, and delete mutations.

Start conservatively with one mutating operation per workspace and a global
GPU concurrency quota. Concurrent read-only status calls are safe. Increase
cross-workspace build concurrency only after verifying that `vllm-envs` cache
publication and garbage collection are safe under that load.

On API restart, reconcile nonterminal operations against labeled Jobs. Never
infer success only from the presence of a worktree; success requires the Job's
recorded exit status and a valid `.venv/bin/activate`.

## Authentication and authorization

Use short-lived workload identity issued or injected for the OpenShell
sandbox. Bind each identity to a developer or team and enforce:

- workspace names and ownership;
- allowed refs and branch prefixes;
- per-user workspace and concurrent Job quotas;
- maximum GPU, CPU, memory, and timeout values; and
- access only to that identity's operation logs.

The agent receives an API credential, not a kubeconfig. The API validates
logical IDs and returns a server-derived PVC subpath suitable for OpenShell;
it never allows `..`, absolute mount paths, arbitrary PVC names, images, service
accounts, or namespaces from callers.

## Agent client

Provide a small `vllm-env` CLI for humans and agents. It should support JSON
output, waiting with log streaming, cancellation, and stable exit codes:

```text
vllm-env workspace create --name NAME --ref REF --branch BRANCH --wait
vllm-env workspace status NAME
vllm-env workspace sync NAME --wait
vllm-env run NAME [resource flags] -- COMMAND ARG...
vllm-env workspace delete NAME
```

Keep the client independent of `ve`. `ve` remains the local build executor
inside `vllm-dev-image`; `vllm-env` is the remote service client in the agent
image. A thin OpenShell-derived agent image may pin this client and policy, but
it must not inherit from `vllm-dev-image`.

## Changes by repository

### `neuralmagic/vllm-envs`

- Add the service and client, or create a sibling service repository if the
  maintainers want to keep `ve` strictly local.
- Expose machine-readable output and stable exit/error categories where the
  service currently would need to parse human output.
- Confirm concurrency guarantees for cache publication, `ve new`, `ve rm`,
  and garbage collection.
- Add a remote-hook mode or document `VE_NO_SYNC=1` as the supported agent
  behavior.

### `neuralmagic/vllm-dev-image`

- Install pinned `vllm-envs` and `uv`.
- Supply and test the complete vLLM CUDA build toolchain.
- Use a stable non-root UID/GID compatible with the shared volume.
- Provide a simple executor entrypoint suitable for Kubernetes Jobs.
- Publish immutable digests and compatibility metadata.
- Do not install OpenShell, Claude Code, Codex, or their credentials.

### OpenShell agent image or configuration

- Start from the maintained OpenShell agent base.
- Add only the `vllm-env` client if it is not downloaded at session start.
- Add policy for the internal API and selected Git/agent providers.
- Set `VE_NO_SYNC=1` for these workspaces.
- Mount only the assigned workspace subpath when practical.

### `llm-manifesto`

No service implementation belongs here. Manifesto's contract remains
`--vllm-env PATH` or `runtime.vllm_env`: validate the mounted external
environment and activate it before serving.

## Delivery phases

### Phase 0: prove the Job contract

- Manually run `ve new`, `ve sync`, a targeted test, and `ve rm` as separate
  Jobs against the shared PVC.
- Validate UID/GID, reflinks, GPU detection, cache reuse, and builder/runtime
  ABI compatibility.
- Validate an OpenShell sandbox editing the same worktree with
  `VE_NO_SYNC=1`.

### Phase 1: minimum service

- Single API replica, authentication, persistent metadata, and Kubernetes Job
  executor.
- Create, get, sync, run, logs, cancel, and safe delete.
- Per-workspace locking, quotas, audit events, and a JSON-capable CLI.
- End-to-end test from workspace creation through Manifesto startup.

### Phase 2: hardening

- Restart reconciliation, durable logs, metrics, alerts, backup, and restore.
- Network policy, admission controls, image allowlists, and threat-model review.
- Queue visibility, fair sharing, and operator maintenance workflows.
- Compatibility tests across builder and serving image releases.

### Phase 3: optimize from measurements

- Warm CPU or GPU worker pools if Job startup dominates iteration time.
- Smarter GPU requests when sync can prove a cache hit without compiling.
- Build cancellation, priority, preemption, and multi-cluster execution.
- Optional deployment-use registration to make workspace deletion safer.

## Acceptance criteria for v1

- A user with no Kubernetes credentials can create a named environment from a
  Git ref and receive its canonical path.
- A fresh OpenShell Claude Code or Codex sandbox can edit only that worktree.
- Checkout hooks do not build in the agent sandbox.
- The agent can request sync and a GPU test, stream logs, and receive a stable
  success or failure result.
- Two requests cannot mutate one worktree concurrently.
- A successful environment can be deployed by Manifesto without copying it.
- Dirty, unpushed, or actively building worktrees are not deleted by default;
  deployment-aware protection is explicitly deferred to a lease integration.
- GPU executors scale to zero when idle.
- Builder pods contain no agent, GitHub, model, or cluster credentials beyond
  the minimum identity required for their operation.
