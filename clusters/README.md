# Cluster profiles

Only synthetic profiles belong in this directory. The bundled files exist for
documentation and tests; they are not production-ready cluster definitions.

Keep real profiles in the private user catalog:

```text
~/.config/llm-manifesto/clusters/<kube-context>.yaml
```

Before publishing a profile, remove provider and site names, kube contexts,
namespaces, node labels and taints, storage classes and claim names, internal
paths, network interfaces and addresses, resource-claim templates, registry
credentials, and environment-specific tuning.

Accelerator profiles contain an explicit `allocation` block with exactly one
backend: `extended_resource.resource_name` for extended-resource requests or
`dra.device_class_name` for Manifesto-generated DRA `ResourceClaimTemplate`
objects. Keep DeviceClass names and matching Kueue `deviceClassMappings` private
when they are environment-specific.

Slurm profiles set `platform: slurm`, a `slurm` configuration, and
`allocation.slurm.gres` for every accelerator. `example-slurm.yaml` uses
Apptainer. Keep login addresses (`slurm.ssh_host`), accounts, site constraints, and
actual bind paths in private profiles. Slurm profiles are selected explicitly
with `--cluster` or `MANIFESTO_CLUSTER` and do not use kube-context discovery.
