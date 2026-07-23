# Pulumi patterns across `infra/`

`infra/` hosts several independent Pulumi projects that do genuinely different
jobs. They look similar (Python, GCS state, KMS secrets) but follow four
distinct, intentional patterns. Read this before creating or modifying a Pulumi
project so new work lands in the right pattern instead of copying whichever
neighbor was closest.

Every project is a separate Pulumi program with its own `Pulumi.yaml` `name:`.
The names below are those project names, not directory names.

## The four patterns

### 1. IaC — cluster prerequisites (`infra/pulumi`, project `marin-iac`)

Declarative provisioning of the things a cluster needs before Iris runs on it:
RBAC, NodePools, Kueue, Traefik/cert-manager, DNS, static IPs. One stack per
cluster (`Pulumi.<cluster>.yaml`, e.g. `cw-rno2a`, `marin`). Because these
resources usually already exist when a cluster is first brought under Pulumi,
the program supports a one-shot adoption flow: set `marin-iac:import=true` and
the run imports the live resources into state instead of recreating them
(`infra/pulumi/__main__.py`, `config.get_bool("import")`).

`infra/pulumi` also hosts the shared **`iac` package** (`infra/pulumi/src/iac`)
that the other projects import. The reusable components live here:

- `iac.gcp.cloud_run.CloudRunService`
- `iac.iris.service.IrisService`
- `iac.gcp.cloud_sql.CloudSqlPostgres`
- `iac.gcp.permissions.GcpDeployPermissions`

Use this pattern only for cluster-level prerequisites, or to add a reusable
component to `iac` for others to consume.

### 2. Deploy — ship app code/containers (`infra/loom`, project `marin-loom`)

Builds and pushes a Docker image, writes the image digest and rendered runtime
config into GCE VM metadata, then runs an activation step that polls the
service's readiness endpoint until it is live. Secret *values* never enter
Pulumi state — only a pinned Secret Manager version number does
(`dotenv-secret-version` in the VM metadata), and the VM reads the value at
boot. See `infra/loom/infrastructure.py` for the metadata block and
`infra/loom/activate.sh` for the `/api/ready` poll.

Most deploys do **not** need to hand-declare resources like loom does. The
common sub-pattern is a **thin leaf** on a shared `iac.*` component:

| Directory | Project | Shared component |
| --- | --- | --- |
| `infra/ducky` | `ducky` | `iac.iris.service.IrisService` |
| `infra/grafana` | `marin-grafana` | `iac.gcp.cloud_run.CloudRunService` |
| `infra/evaldash` | `marin-evaldash` | `iac.gcp.cloud_run.CloudRunService` |
| `infra/cloudsql` | `marin-cloudsql` | `iac.gcp.cloud_sql.CloudSqlPostgres` |

Each of these is a short `__main__.py` that constructs one component with its
args. `infra/loom` is the raw-deploy exception, warranted because it manages a
durable single-host VM with a bespoke rollout.

### 3. Shared IAM/permissions grants (`infra/permissions`, project `marin-permissions`)

Narrow, additive-only IAM grants to identities and resources that other
projects own. Built on `iac.gcp.permissions.GcpDeployPermissions`, which
creates non-authoritative (additive) IAM members only — it grants access but
does not own the resource being granted access to. Use this when a deploy needs
a role on a resource declared elsewhere; do not add the grant to the project
that owns the resource.

### 4. SaaS/third-party resource declarations (`infra/pulumi/github`, project `marin-github`)

Records external SaaS state that lives outside GCP — here, GitHub Actions
secrets — as lookup/external resources. It tracks metadata (that a secret
exists, its identifiers) but the secret *values* stay outside Pulumi. A
companion non-Pulumi `infra/pulumi/github/audit.py` audits the live state. Use
this pattern to bring third-party resources under review without Pulumi owning
their contents.

## Shared conventions

All projects share these, and new projects should too:

- **State backend.** GCS, `gs://marin-iac-state`, selected by the operator/CI
  environment (`pulumi login`), not committed in `Pulumi.yaml`.
- **Secrets provider.** GCP KMS
  (`gcpkms://…/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key`), configured
  per-stack in `Pulumi.<stack>.yaml`.
- **Shared virtualenv.** Leaf projects pin `runtime.options.virtualenv` to the
  repo `.venv` (`../../.venv`, or `../../../.venv` for `infra/pulumi/github`) so
  Pulumi reuses the repo's dependency closure instead of building its own.
  `infra/pulumi` itself uses the plain `python` runtime for the same reason —
  it deliberately runs under the repo's uv venv to avoid re-pulling the heavy
  `marin-iris` closure.
- **Cross-project data.** Read another project's outputs with
  `pulumi.StackReference` (e.g. `infra/grafana` references the `cloudsql` and
  `loom` stacks) rather than duplicating values.
- **Secret values never in state.** Store a pinned Secret Manager version
  number and resolve the value at runtime; never put the value in Pulumi config
  or state.

## Picking a pattern for new work

1. Cluster-level prerequisite (RBAC, NodePool, DNS, ingress)? Extend
   `marin-iac`.
2. Shipping a service or container? Prefer a **thin leaf on a shared `iac.*`
   component** (patterns above). Reach for a raw `loom`-style deploy only when
   the shared components genuinely can't express the need — and consider adding
   the missing capability to `iac` instead.
3. Only granting access to a resource another project owns? Add an additive
   grant in `marin-permissions`.
4. Recording third-party/SaaS resources for review, values staying outside
   Pulumi? Follow the `marin-github` external-resource pattern.

When in doubt, prefer extending or adding to the shared `iac.*` components over
a bespoke program.
