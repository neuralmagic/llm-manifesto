# vLLM Development with OpenShell, vllm-envs, and Manifesto

This runbook describes the target workflow for:

1. bootstrapping a fresh Kubernetes cluster through creation of a persistent
   vLLM development environment; and
2. starting a fresh Claude Code or Codex session in OpenShell, backed by a new
   `vllm-envs` worktree that Manifesto can deploy.

The workflow deliberately uses three different execution planes:

| Plane | Runs | Image | GPU |
|---|---|---|---|
| Agent | OpenShell, Claude Code or Codex, Git, a small service client | OpenShell agent image | No, normally |
| Build | `ve new`, `ve sync`, builds, and tests | `vllm-dev-image` | When requested |
| Serve | Manifesto-managed vLLM model pods | ABI-compatible vLLM runtime image | Yes |

The agent is disposable. Source, worktrees, virtual environments, and build
caches are persistent. The `vllm-dev-image` remains a vLLM builder and does not
need OpenShell, Claude Code, or Codex installed in it.

The build plane described here does not exist yet. Its proposed contract and
implementation scope are in
[vLLM Development Services](vllm-development-services.md).

## Storage and compatibility contract

The build service and Manifesto model pods mount the same persistent storage at
the same absolute path. For example:

```text
/mnt/shared/dev/<developer>/vllm-envs/<environment>
```

A completed `vllm-envs` worktree contains vLLM source and its private `.venv`.
Manifesto receives that absolute worktree path through `--vllm-env`.

The agent only needs the selected worktree mounted, preferably as a PVC
subpath at `/sandbox/workspace/vllm`. It does not need the shared caches or a
GPU. The path inside the agent may differ because only the build and serve
planes exchange an environment by absolute path.

The builder and serving image must have compatible Python, PyTorch, CUDA, and
native-extension ABIs. Pinning both to the same `vllm-dev-image` digest is the
simplest initial contract, but a smaller runtime image is also valid when that
compatibility is tested. The OpenShell agent image is independent.

## Required component work

Before this workflow is available end to end:

- implement the development environment service described in the companion
  design;
- install and pin `vllm-envs` and `uv` in `vllm-dev-image`;
- verify that `vllm-dev-image` can perform a cold vLLM build with CUDA, CMake,
  Ninja, and ccache;
- establish a stable UID/GID or `fsGroup` contract for files shared by build
  jobs, agent sandboxes, and model pods;
- publish immutable builder image tags or digests; and
- optionally publish a small agent image derived from the OpenShell base image
  that adds only the development service client and policy. Do not add the
  coding agents to `vllm-dev-image`.

## Shared conventions

The examples use placeholders. Keep real cluster names, credentials, storage
classes, and registry locations in private configuration.

```bash
export DEV_NAMESPACE=vllm-dev
export OPENSHELL_NAMESPACE=openshell-system
export DEV_PVC=vllm-dev-shared
export DEV_ID=alice
export DEV_ROOT=/mnt/shared/dev/$DEV_ID
export VE_ENVS_ROOT=$DEV_ROOT/vllm-envs
export VLLM_ENV_API=https://vllm-envs.example.com
export VLLM_DEV_IMAGE='registry.example.com/team/vllm-dev@sha256:<digest>'
export VLLM_RUNTIME_IMAGE="$VLLM_DEV_IMAGE"
export OPENSHELL_AGENT_IMAGE='registry.example.com/team/vllm-agent@sha256:<digest>'
export OPENSHELL_CHART_VERSION='<pinned-version>'
export AGENT_SANDBOX_VERSION='<pinned-version>'
export MODEL_SPEC=models/qwen/aggregated.yaml
export CLUSTER_PROFILE='<private-cluster-profile>'
```

## Workflow 1: fresh cluster to first development environment

### 1. Verify cluster prerequisites

The cluster needs Kubernetes with RBAC, working NVIDIA GPU scheduling, Helm,
and storage accessible from every build and serving node. Use ReadWriteMany
storage when those workloads may run on different nodes. A reflink-capable
filesystem such as XFS or btrfs makes `vllm-envs` substantially more efficient;
plain copies are the fallback.

The operator workstation needs `kubectl`, `helm`, `jq`, `openshell`, and
`manifesto`:

```bash
kubectl get nodes
kubectl get storageclass
kubectl get nodes \
  -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu
```

### 2. Create persistent development storage

Create the namespace and a site-appropriate PVC:

```bash
kubectl create namespace "$DEV_NAMESPACE"
```

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-dev-shared
  namespace: vllm-dev
spec:
  accessModes: [ReadWriteMany]
  storageClassName: <shared-storage-class>
  resources:
    requests:
      storage: 1Ti
```

Configure the private Manifesto cluster profile to mount it at the canonical
path:

```yaml
storage:
  shared_volume:
    persistentVolumeClaim:
      claimName: vllm-dev-shared
  shared_mount_path: /mnt/shared
```

### 3. Install the build-plane service

Deploy the development environment API and its Job executor in
`$DEV_NAMESPACE`. Configure it with:

- the shared PVC mounted at `/mnt/shared`;
- the immutable `vllm-dev-image` digest;
- one configured vLLM repository and derived per-developer roots;
- a Kubernetes service account that can create and inspect only its own Jobs;
- authentication and per-developer authorization; and
- conservative CPU, memory, GPU, timeout, and concurrency limits.

The service should scale GPU executors to zero: API requests create Kubernetes
Jobs, and only Jobs that may compile or run GPU tests request a GPU. The API
itself does not need one.

Initialize the repository and first environment through the service rather
than an interactive pod. The exact client syntax is proposed, not yet
implemented:

```bash
vllm-env workspace create \
  --name base \
  --ref origin/main \
  --branch dev/base \
  --wait
```

On completion, record the returned canonical path and verify it contains
`.venv/bin/activate`. A failed cold build should remain inspectable through the
operation logs rather than leaving an untracked worktree.

### 4. Install OpenShell

Install the Kubernetes Agent Sandbox controller and OpenShell from reviewed,
pinned releases. Direct sandbox workloads to the development namespace:

```bash
kubectl apply -f \
  "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${AGENT_SANDBOX_VERSION}/manifest.yaml"

kubectl create namespace "$OPENSHELL_NAMESPACE"
helm upgrade --install openshell \
  oci://ghcr.io/nvidia/openshell/helm-chart \
  --version "$OPENSHELL_CHART_VERSION" \
  --namespace "$OPENSHELL_NAMESPACE" \
  --set server.sandboxNamespace="$DEV_NAMESPACE" \
  --set server.sandboxImagePullPolicy=IfNotPresent
```

Follow the current OpenShell Kubernetes or OpenShift documentation for
controller versions, service accounts, TLS, and gateway registration. Then
configure only the providers used by a session:

```bash
openshell provider create --name claude-dev --type claude --from-existing
openshell provider create --name codex-dev --type codex --from-existing
openshell provider create --name github-dev --type github --from-existing
```

Add a provider or short-lived workload identity for the development service.
The agent must not receive Kubernetes credentials.

### 5. Verify the end-to-end contract

Before admitting users, verify that:

- the API can create a worktree and complete a cold `ve sync`;
- an agent sandbox can edit only its assigned worktree and call the API;
- a build Job observes those edits and can run a selected test;
- Manifesto can deploy the same environment using its canonical build-plane
  path; and
- neither the agent nor build Job writes credentials onto the shared PVC.

## Workflow 2: fresh agent session in a new worktree

### 1. Create the worktree before the sandbox

Choose unique names, then ask the service to create and fully sync the
environment:

```bash
export SESSION=fix-kernel-selection
export VLLM_REF=origin/main
export GIT_BRANCH=dev/fix-kernel-selection

vllm-env workspace create \
  --name "$SESSION" \
  --ref "$VLLM_REF" \
  --branch "$GIT_BRANCH" \
  --wait \
  --output json > /tmp/vllm-workspace.json

export VLLM_ENV_PATH="$(jq -r .path /tmp/vllm-workspace.json)"
export VLLM_ENV_SUBPATH="$(jq -r .pvc_subpath /tmp/vllm-workspace.json)"
```

The service runs `ve new` in `vllm-dev-image`, owns the repository/worktree
locks, and returns a path only after `.venv/bin/activate` exists. Creating the
worktree first is required because Kubernetes PVC `subPath` mounts expect the
target to exist when the sandbox starts.

### 2. Start the disposable agent sandbox

Create a private OpenShell driver config that mounts only
`$VLLM_ENV_SUBPATH` at `/sandbox/workspace/vllm`:

```json
{
  "kubernetes": {
    "volumes": [{
      "name": "vllm-worktree",
      "persistent_volume_claim": {
        "claim_name": "vllm-dev-shared",
        "read_only": false
      }
    }],
    "containers": {
      "agent": {
        "volume_mounts": [{
          "name": "vllm-worktree",
          "mount_path": "/sandbox/workspace/vllm",
          "sub_path": "dev/alice/vllm-envs/fix-kernel-selection",
          "read_only": false
        }]
      }
    }
  }
}
```

Generate the `sub_path` value from the service response rather than accepting
an arbitrary user path. Then launch the selected agent from the OpenShell
agent image, without a GPU:

```bash
export OPENSHELL_DRIVER_CONFIG="$(jq -c . openshell-vllm-worktree.json)"
export AGENT=claude
export AGENT_PROVIDER=claude-dev

openshell sandbox create \
  --name "$SESSION" \
  --from "$OPENSHELL_AGENT_IMAGE" \
  --cpu 8 \
  --memory 32Gi \
  --provider "$AGENT_PROVIDER" \
  --provider github-dev \
  --provider vllm-env-api \
  --driver-config-json "$OPENSHELL_DRIVER_CONFIG" \
  --env VE_NO_SYNC=1 \
  --env VLLM_ENV_API="$VLLM_ENV_API" \
  -- /bin/bash -lc 'cd /sandbox/workspace/vllm && exec "$AGENT"'
```

`VE_NO_SYNC=1` is important. `vllm-envs` installs Git hooks that normally sync
after a checkout. In an agent sandbox, the hook must skip that local sync;
the agent requests a build-plane sync after changing commits or build inputs.

The OpenShell image contains agent tooling and its policy. It does not contain
the CUDA build toolchain, and the builder contains no agent credentials.

### 3. Edit, sync, and test

Claude Code or Codex edits and performs ordinary Git operations directly in
the mounted worktree. It delegates environment mutation and execution to the
service:

```bash
vllm-env workspace status "$SESSION"
vllm-env workspace sync "$SESSION" --wait
vllm-env run "$SESSION" --gpu 1 --timeout 30m -- \
  python -m pytest -q tests/test_selected.py
```

Sync before any command that imports vLLM after a checkout or after changing
requirements, Python build metadata, CMake, or native source. The first
implementation may execute every run as a fresh Job. A later warm worker pool
is an optimization, not part of the correctness contract.

### 4. Deploy the worktree with Manifesto

Set the model image to the pinned runtime image, render, inspect, and deploy:

```bash
manifesto render manifest "$MODEL_SPEC" \
  --cluster "$CLUSTER_PROFILE" \
  --namespace "$DEV_NAMESPACE" \
  --vllm-env "$VLLM_ENV_PATH" > /tmp/manifesto-vllm-dev.yaml

kubectl apply --dry-run=server -f /tmp/manifesto-vllm-dev.yaml
manifesto deploy "$MODEL_SPEC" \
  --cluster "$CLUSTER_PROFILE" \
  --namespace "$DEV_NAMESPACE" \
  --vllm-env "$VLLM_ENV_PATH"
manifesto ready "$MODEL_SPEC" \
  --cluster "$CLUSTER_PROFILE" \
  --namespace "$DEV_NAMESPACE"
```

Manifesto validates that the path is absolute and covered by a model-pod
volume mount. Pod startup fails before launching vLLM if the worktree or
`.venv/bin/activate` is missing.

### 5. End the session safely

Commit and push wanted changes. Stop deployments before asking the service to
remove the environment:

```bash
manifesto stop "$MODEL_SPEC" --namespace "$DEV_NAMESPACE"
openshell sandbox delete "$SESSION"
vllm-env workspace delete "$SESSION"
```

Deletion must use `ve rm` and refuse dirty or unpushed work unless an operator
explicitly forces it. Cache reclamation is a separate, auditable maintenance
operation.

## Operational invariants

- OpenShell owns disposable agent lifecycle, policy, and agent credentials.
- The development service owns vLLM repository, worktree, environment, build,
  and cache mutations through `vllm-envs`.
- Agent sandboxes set `VE_NO_SYNC=1` and never run `ve sync` locally.
- Build jobs use `vllm-dev-image`; agent sandboxes use an OpenShell agent image.
- Builder and serving images are ABI-compatible; the agent image is unrelated.
- Manifesto only deploys a pre-existing environment at an explicit path.
- A worktree has at most one mutating build operation at a time.
- Never remove an environment while a build Job or model deployment uses it.
- Never store API keys, registry credentials, kubeconfigs, or agent
  authentication files on the shared PVC.

## Upstream references

- [OpenShell Kubernetes setup](https://docs.nvidia.com/openshell/latest/kubernetes/setup)
- [OpenShell Kubernetes PVC mounts](https://docs.nvidia.com/openshell/latest/reference/sandbox-compute-drivers#kubernetes-driver-config-pvc-mounts)
- [OpenShell providers](https://docs.nvidia.com/openshell/latest/sandboxes/manage-providers)
- [OpenShell base image](https://github.com/NVIDIA/OpenShell-Community/tree/main/sandboxes/base)
- [`vllm-envs`](https://github.com/neuralmagic/vllm-envs)
- [`vllm-dev-image`](https://github.com/neuralmagic/vllm-dev-image)
