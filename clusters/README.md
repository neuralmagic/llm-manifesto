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
