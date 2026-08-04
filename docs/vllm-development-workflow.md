# vLLM Development with OpenShell, vllm-envs, and Manifesto

This runbook describes two related workflows:

1. bootstrap a fresh Kubernetes cluster through creation of the first persistent
   vLLM development environment; and
2. start a new Claude Code or Codex session in OpenShell, backed by a new
   `vllm-envs` worktree that Manifesto can deploy directly.

The central contract is one absolute path:

```text
/mnt/shared/dev/<developer>/vllm-envs/<environment>
```

OpenShell and every Manifesto model pod must mount the same storage at
`/mnt/shared`. A `vllm-envs` worktree contains both the vLLM source and its
private `.venv`, so Manifesto only needs the worktree path.

## Status and required image work

This is the target workflow. It depends on an OpenShell-compatible release of
[`neuralmagic/vllm-dev-image`](https://github.com/neuralmagic/vllm-dev-image).
The current image is useful for interactive vLLM development, but does not yet
provide the complete OpenShell agent-image contract.

Before using this runbook end to end, update that image to:

- install pinned versions of `uv`, Claude Code, Codex, Node.js, and the GitHub
  CLI, following the
  [OpenShell base image](https://github.com/NVIDIA/OpenShell-Community/tree/main/sandboxes/base);
- install a pinned revision or release of
  [`vllm-envs`](https://github.com/neuralmagic/vllm-envs) and expose `ve` on
  `PATH`;
- provide `/etc/openshell/policy.yaml` with `/sandbox`, `/tmp`, and
  `/mnt/shared` writable, and with narrowly scoped network access for the chosen
  agent, Git, Python package indexes, vLLM wheels, and vLLM build dependencies;
- provide an unprivileged sandbox user and a writable home at `/sandbox`;
- verify that the image contains the CUDA compiler, headers, CMake, Ninja,
  ccache, and other tools needed for a cold local vLLM extension build;
- make files created on shared storage readable by the UID/GID used by
  Manifesto model pods; and
- publish immutable image tags or digests. Use the same image digest for the
  OpenShell sandbox and the Manifesto model spec to avoid Python, CUDA, and
  native-extension ABI drift.

Do not make the policy permissive merely to get the first build working. Start
with the OpenShell base policy, add the vLLM-specific filesystem and network
requirements, and use OpenShell denial logs to tighten the result.

## Shared conventions

The examples use synthetic values. Set these in the operator shell and keep
real cluster names, credentials, storage classes, and registry locations in
private configuration.

```bash
export DEV_NAMESPACE=vllm-dev
export OPENSHELL_NAMESPACE=openshell-system
export DEV_PVC=vllm-dev-shared
export DEV_ID=alice
export DEV_ROOT=/mnt/shared/dev/$DEV_ID
export VLLM_REPO=$DEV_ROOT/src/vllm
export VE_CACHE_DIR=$DEV_ROOT/cache/vllm-envs
export VE_ENVS_ROOT=$DEV_ROOT/vllm-envs
export UV_CACHE_DIR=$DEV_ROOT/cache/uv
export CCACHE_DIR=$DEV_ROOT/cache/ccache
export VLLM_DEV_IMAGE='registry.example.com/team/vllm-dev@sha256:<digest>'
export OPENSHELL_CHART_VERSION='<pinned-version>'
export AGENT_SANDBOX_VERSION='<pinned-version>'
export MODEL_SPEC=models/qwen/aggregated.yaml
export CLUSTER_PROFILE='<private-cluster-profile>'
```

Use an immutable image digest in real automation. The literal placeholders
above are intentionally not runnable until replaced.

## Workflow 1: fresh cluster to first development environment

### 1. Verify cluster prerequisites

The cluster must have:

- Kubernetes 1.29 or newer with RBAC;
- working GPU scheduling and the NVIDIA device plugin;
- Helm 3;
- a storage backend accessible from every GPU node that may run a sandbox or
  model pod; and
- a ReadWriteMany volume when work spans multiple nodes.

The operator workstation needs `kubectl`, `helm`, `jq`, `openshell`, and
`manifesto` installed and authenticated for the intended cluster.

Check the basics before installing OpenShell:

```bash
kubectl get nodes
kubectl get storageclass
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu
```

`vllm-envs` benefits from a reflink-capable filesystem such as XFS or btrfs.
It falls back to copies when reflinks are unavailable, but environment creation
and cache storage will be more expensive.

### 2. Create the shared development namespace and storage

Create the namespace:

```bash
kubectl create namespace "$DEV_NAMESPACE"
```

Create a PVC appropriate for the cluster. Keep the manifest in private cluster
configuration because the storage class and capacity are site-specific:

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

Wait for it to bind:

```bash
kubectl -n "$DEV_NAMESPACE" get pvc "$DEV_PVC" --watch
```

Configure the private Manifesto cluster profile to mount this claim at the
same absolute path OpenShell will use:

```yaml
storage:
  shared_volume:
    persistentVolumeClaim:
      claimName: vllm-dev-shared
  shared_mount_path: /mnt/shared
```

### 3. Install OpenShell on Kubernetes

OpenShell's Kubernetes path is experimental. Pin releases rather than using
floating development versions.

Install the Kubernetes Agent Sandbox controller from a reviewed, pinned
release manifest, then verify it:

```bash
kubectl apply -f \
  "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${AGENT_SANDBOX_VERSION}/manifest.yaml"
kubectl -n agent-sandbox-system get pods
```

Install OpenShell with a separate control-plane namespace and direct sandbox
workloads to the development namespace:

```bash
kubectl create namespace "$OPENSHELL_NAMESPACE"
helm upgrade --install openshell \
  oci://ghcr.io/nvidia/openshell/helm-chart \
  --version "$OPENSHELL_CHART_VERSION" \
  --namespace "$OPENSHELL_NAMESPACE" \
  --set server.sandboxNamespace="$DEV_NAMESPACE" \
  --set server.sandboxImagePullPolicy=IfNotPresent
kubectl -n "$OPENSHELL_NAMESPACE" rollout status statefulset/openshell
```

If the cluster is OpenShift, follow the current
[OpenShell OpenShift installation guide](https://docs.nvidia.com/openshell/latest/kubernetes/openshift)
instead of copying generic SCC settings into a shared cluster. Apply any
sandbox service-account permissions to the configured sandbox namespace,
`$DEV_NAMESPACE`, rather than assuming it matches the Helm release namespace.

Install and register the workstation CLI using the
[Kubernetes setup guide](https://docs.nvidia.com/openshell/latest/kubernetes/setup),
then confirm the selected gateway:

```bash
openshell status
```

### 4. Configure agent and source-control providers

Create providers from credentials already present in the operator shell. Do
not put credential values in this repository, shell history, sandbox images,
PVCs, or OpenShell driver JSON.

```bash
openshell provider create --name claude-dev --type claude --from-existing
openshell provider create --name codex-dev --type codex --from-existing
openshell provider create --name github-dev --type github --from-existing
```

Only create the providers actually used. Provider attachment and network policy
are separate: a credential does not grant an endpoint or Git write access by
itself.

### 5. Define the shared PVC mount for sandboxes

Save this as a private local file named `openshell-vllm-storage.json`:

```json
{
  "kubernetes": {
    "volumes": [
      {
        "name": "vllm-dev",
        "persistent_volume_claim": {
          "claim_name": "vllm-dev-shared",
          "read_only": false
        }
      }
    ],
    "containers": {
      "agent": {
        "volume_mounts": [
          {
            "name": "vllm-dev",
            "mount_path": "/mnt/shared",
            "read_only": false
          }
        ]
      }
    }
  }
}
```

An explicit mount outside `/sandbox` preserves OpenShell's normal workspace
volume while making the vLLM repository, worktrees, and caches durable.

### 6. Create the bootstrap sandbox

```bash
export OPENSHELL_DRIVER_CONFIG="$(jq -c . openshell-vllm-storage.json)"

openshell sandbox create \
  --name vllm-bootstrap \
  --from "$VLLM_DEV_IMAGE" \
  --gpu 1 \
  --cpu 16 \
  --memory 64Gi \
  --driver-config-json "$OPENSHELL_DRIVER_CONFIG" \
  --env DEV_ROOT="$DEV_ROOT" \
  --env VLLM_REPO="$VLLM_REPO" \
  --env VE_CACHE_DIR="$VE_CACHE_DIR" \
  --env VE_ENVS_ROOT="$VE_ENVS_ROOT" \
  --env UV_CACHE_DIR="$UV_CACHE_DIR" \
  --env CCACHE_DIR="$CCACHE_DIR" \
  -- /bin/bash
```

If the image is private, configure `server.sandboxImagePullSecrets` in the
OpenShell chart. The referenced pull secret must exist in `$DEV_NAMESPACE`.

### 7. Initialize the persistent repository and first environment

Inside the bootstrap sandbox:

```bash
set -euo pipefail
test -w /mnt/shared
mkdir -p "$DEV_ROOT/src" "$VE_CACHE_DIR" "$VE_ENVS_ROOT"
git clone https://github.com/vllm-project/vllm.git "$VLLM_REPO"
cd "$VLLM_REPO"
ve init --name base
source .venv/bin/activate
ve status
python -c 'import vllm; print(vllm.__version__)'
```

If `/mnt/shared` is not writable, stop here. Fix PVC ownership, fsGroup, or the
image's UID/GID contract in cluster configuration; do not work around it with a
world-writable volume.

The first `ve init` can perform cold dependency downloads or a local CUDA
build. Later environments reuse the content-addressed cache under
`$VE_CACHE_DIR`.

Exit and delete only the bootstrap sandbox. The repository, base `.venv`, and
caches remain on the PVC:

```bash
openshell sandbox delete vllm-bootstrap
```

## Workflow 2: fresh agent session in a new vllm-envs worktree

Run this workflow for each task. Each OpenShell sandbox gets one agent and one
new `vllm-envs` worktree, while expensive dependency and build layers are shared.

### 1. Choose the session inputs

```bash
export SESSION=fix-kernel-selection
export VLLM_REF=origin/main
export GIT_BRANCH=dev/fix-kernel-selection
export AGENT=claude                 # claude or codex
export AGENT_PROVIDER=claude-dev    # claude-dev or codex-dev
export VLLM_ENV_PATH=$VE_ENVS_ROOT/$SESSION
export OPENSHELL_DRIVER_CONFIG="$(jq -c . openshell-vllm-storage.json)"
```

Environment names and branch names must be unique for concurrent sessions.

### 2. Create the sandbox, worktree, branch, and agent process

```bash
openshell sandbox create \
  --name "$SESSION" \
  --from "$VLLM_DEV_IMAGE" \
  --gpu 1 \
  --cpu 16 \
  --memory 64Gi \
  --provider "$AGENT_PROVIDER" \
  --provider github-dev \
  --driver-config-json "$OPENSHELL_DRIVER_CONFIG" \
  --env AGENT="$AGENT" \
  --env VLLM_REF="$VLLM_REF" \
  --env GIT_BRANCH="$GIT_BRANCH" \
  --env SESSION="$SESSION" \
  --env VLLM_REPO="$VLLM_REPO" \
  --env VE_CACHE_DIR="$VE_CACHE_DIR" \
  --env VE_ENVS_ROOT="$VE_ENVS_ROOT" \
  --env UV_CACHE_DIR="$UV_CACHE_DIR" \
  --env CCACHE_DIR="$CCACHE_DIR" \
  -- /bin/bash -lc '
    set -euo pipefail
    git -C "$VLLM_REPO" fetch origin
    ve new "$VLLM_REF" --name "$SESSION" --repo "$VLLM_REPO"
    worktree="$VE_ENVS_ROOT/$SESSION"
    git -C "$worktree" switch -c "$GIT_BRANCH"
    cd "$worktree"
    source .venv/bin/activate
    exec "$AGENT"
  '
```

Passing providers explicitly keeps credential selection independent of the
shell wrapper used to create the worktree. The image policy must cover the
selected agent; OpenShell's default Claude policy is not sufficient for Codex.

If worktree creation fails, inspect the sandbox and policy logs:

```bash
openshell sandbox get "$SESSION"
openshell logs "$SESSION" --tail
openshell sandbox connect "$SESSION"
```

### 3. Deploy that worktree with Manifesto

Keep the sandbox image and the model image pinned to the same digest. From the
operator workstation, render and inspect before applying:

```yaml
# In the private model spec used for development:
model:
  image: registry.example.com/team/vllm-dev@sha256:<same-digest>
```

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

Manifesto validates that the path is absolute and covered by a model-pod volume
mount. At pod startup it fails before launching vLLM if the worktree or
`.venv/bin/activate` is missing.

Configure `HF_TOKEN` in the operator's private Manifesto environment before a
deployment that downloads gated models. Manifesto copies it into the namespace
Secret; do not store it in the worktree or shared PVC.

### 4. Resume an existing session

OpenShell keeps the sandbox alive after the agent exits unless configured
otherwise. Resume it with:

```bash
openshell sandbox connect "$SESSION"
```

Then restart the agent from its existing worktree:

```bash
cd "$VLLM_ENV_PATH"
source .venv/bin/activate
ve status
"$AGENT"
```

### 5. Finish and clean up safely

Before deleting anything:

```bash
openshell sandbox exec -n "$SESSION" --workdir "$VLLM_ENV_PATH" -- git status --short
openshell sandbox exec -n "$SESSION" --workdir "$VLLM_ENV_PATH" -- git log -1 --oneline
```

Commit and push wanted work, stop any Manifesto deployment using the worktree,
then remove the environment while the PVC is still mounted:

```bash
manifesto stop "$MODEL_SPEC" --namespace "$DEV_NAMESPACE"

openshell sandbox exec -n "$SESSION" \
  --env SESSION="$SESSION" \
  --env VLLM_REPO="$VLLM_REPO" \
  --env VE_CACHE_DIR="$VE_CACHE_DIR" \
  --env VE_ENVS_ROOT="$VE_ENVS_ROOT" \
  -- /bin/bash -lc 'cd "$VLLM_REPO" && ve rm "$SESSION"'

openshell sandbox delete "$SESSION"
```

Periodically audit shared storage from a maintenance sandbox:

```bash
ve list
ve du
ve gc --dry-run
```

Run `ve gc` without `--dry-run` only after reviewing what it will reclaim.

## Operational invariants

- One PVC is mounted at the same absolute path in OpenShell and Manifesto pods.
- One immutable development image digest is used to build and serve the vLLM
  environment.
- OpenShell owns sandbox lifecycle, policy, and credential injection.
- `vllm-envs` owns repositories, worktrees, virtual environments, builds, and
  caches.
- Manifesto only renders and deploys model servers pointed at an existing
  worktree.
- Never remove a worktree while a model deployment is using its `.venv`.
- Never place long-lived API keys, registry credentials, kubeconfigs, or agent
  authentication files on the shared PVC.

## Upstream references

- [OpenShell Kubernetes setup](https://docs.nvidia.com/openshell/latest/kubernetes/setup)
- [OpenShell sandbox management](https://docs.nvidia.com/openshell/latest/sandboxes/manage-sandboxes)
- [OpenShell Kubernetes PVC mounts](https://docs.nvidia.com/openshell/latest/reference/sandbox-compute-drivers#kubernetes-driver-config-pvc-mounts)
- [OpenShell providers](https://docs.nvidia.com/openshell/latest/sandboxes/manage-providers)
- [OpenShell default policy](https://docs.nvidia.com/openshell/latest/reference/default-policy)
- [`vllm-envs`](https://github.com/neuralmagic/vllm-envs)
- [`vllm-dev-image`](https://github.com/neuralmagic/vllm-dev-image)
