---
name: import-vllm-recipe
description: Import a recipes.vllm.ai deployment or upstream vLLM recipe YAML into a repository-native llm-manifesto model spec. Use when creating or updating Manifesto configs from vLLM Recipes URLs, generated serve commands, recipe strategy overrides, or hardware-specific launch guidance, especially for multi-node, expert-parallel, or prefill/decode deployments.
---

# Import a vLLM recipe

Translate recipe intent into Manifesto's declarative topology. Treat the recipe
as an upstream source, the selected cluster profile as the hardware contract,
and rendered Kubernetes YAML as the result to audit.

## Workflow

1. Fetch the exact recipe URL and its linked upstream YAML. Record the selected
   hardware, strategy, node count, variant, and enabled features; query
   parameters may change the generated command without changing the YAML.
2. Inspect the current Kubernetes context, its effective Manifesto cluster
   profile, the closest model specs, and `config/images.yaml` before editing.
3. Create a model spec under `models/<provider>/`. Reuse inheritance only when
   the parent has the same model/runtime contracts; otherwise keep the import
   self-contained and reviewable.
4. Map recipe settings using the rules below. Do not paste generated shell
   commands into `vllm_raw_args` when structured Manifesto fields exist.
5. Validate, render, and audit the complete PodSpecs and routing objects.
6. Add focused regression coverage when the import exercises a new topology or
   renderer behavior. Preserve unrelated worktree changes.

## Mapping rules

- Resolve GPUs from the selected cluster profile, not the recipe site's default
  hardware. A PD query such as `nodes=4` means four nodes in each pool unless
  the page explicitly says otherwise: eight nodes total.
- Express P/D with `topology: pd` and
  `routing: {kind: pd, target_role: decode}`. Manifesto supplies llm-d routing;
  do not copy `vllm-router` launch commands.
- For Manifesto-standard external load balancing, express each role as
  `parallelism: {tp: <local TP>, dp: <global DP>, ep: true}`. Routing plus DP
  derives external multi-port load balancing. Do not copy
  `--data-parallel-hybrid-lb`, coordinator addresses, ranks, or ports from the
  generated recipe command.
- Calculate `global DP = nodes * GPUs per node / TP`. Require the layout to pack
  every pod exactly; never silently leave GPUs idle.
- A cross-node TP role creates headless LWS followers. Ensure the InferencePool
  can select only API-serving workers. If the standard shared PD pool lacks the
  required role-aware leader filter, use an externally load-balanced DEP shape
  when that matches the user's request rather than emitting unroutable pods.
- Preserve the recipe's connector intent, but follow the repository's existing
  role contract. Current Manifesto PD routing uses `NixlConnector` with
  `kv_role: kv_both` for both roles. Preserve other connector fields from the
  recipe exactly; do not add optional connector keys that are absent upstream.
- Map normal flags into the `vllm:` mapping, connector JSON into
  `kv_transfer_config`, and environment variables into `env`. Keep every EnvVar
  value a YAML string. Use `vllm_raw_args` only for unsupported syntax.
- Translate deployment and API-server operations as needed for Manifesto:
  host/port allocation, device and node placement, ranks and coordinators,
  headless workers, DP load-balancing mode, routing processes, and connector
  roles required by P/D orchestration.
- Preserve every recipe setting below the API-server boundary exactly. This
  includes model and load format, dtype and quantization, scheduler and batching
  limits, cache behavior and dtype, attention/MoE/all-to-all/kernel backends,
  compilation and graph modes, speculative decoding, allocators, and internal
  runtime environment variables. Never add, remove, or change one based on
  topology, hardware inference, or optional guide prose. In particular, do not
  infer an all-to-all backend from EP or from a mega-MoE backend such as
  `deep_gemm_mega_moe`.
- Add a below-boundary recommendation absent from the selected recipe only when
  the user explicitly requests the deviation, then report it in the handoff.
- Let the cluster profile own storage, affinity, resource claims, base fabric
  settings, caches, and shared images. Keep recipe-specific model flags,
  role-specific fabric overrides, and one-off images in the model spec.
- Omit values already supplied by schema, topology, or cluster defaults. Common
  examples are derived model labels, default routing for the selected topology,
  default sidecars, `lws.replicas: 1`, `parallelism.tp: 1`, inferred resources,
  and fabric environment values identical to the selected profile. Keep an
  explicit value only when it communicates a constraint or overrides a default.
- Apply recipe precedence deliberately: base args, strategy/role overrides,
  hardware overrides, then default-on features. Do not enable opt-in features
  unless the URL or user selected them.
- Use a unique `release`. Omit `workload_name` unless a stable external name is
  required; when set, it must remain unique among simultaneous deployments.
  State the total nodes and GPUs in a leading comment for multi-node specs.

## Validation

Use an explicit cluster, namespace, and user:

```bash
uv run manifesto config validate MODEL --cluster CLUSTER
uv run manifesto explain MODEL --cluster CLUSTER --namespace NAMESPACE --user USER
uv run manifesto render manifest MODEL \
  --cluster CLUSTER --namespace NAMESPACE --user USER
```

Audit every rendered role for:

- expected Deployment/LWS count, size, replicas, and full-node GPU requests;
- derived TP/DP sizes, external-load-balancing flags, and headless behavior;
- NIXL side-channel host injection and role-specific collective/fabric settings;
- API-serving ports, Services, InferencePool selectors, and EPP profiles;
- image, pull policy, secrets, storage, affinity, sidecars, and claims;
- string EnvVar values and no zero resource quantities.

Run relevant tests after the render audit. Do not deploy unless the user asks.

## Handoff

Report the source recipe, the effective cluster profile, the final topology,
intentional deviations from the generated command, and validation results.
