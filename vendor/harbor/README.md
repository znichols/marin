# Harbor evaluation runtime

Marin vendors the [Harbor fork](https://github.com/marin-community/harbor) at the
revision recorded in [UPSTREAM.toml](UPSTREAM.toml). The source is licensed under
[Apache 2.0](LICENSE). Adapter-specific license files remain beside their sources.

This package contains the evaluation runtime, agents, environment providers,
verification, trajectories, registry downloads, FineStore archives, telemetry,
and every adapter and task-generation template from that revision. The `harbor`
distribution provides both the `harbor` and `harbor_config` Python modules.

The website, results viewer, Hub result sharing, benchmark publishing commands,
standalone config release tooling, and upstream development automation are
omitted. Task collection and hashing remain part of evaluation job locking.
The full upstream `registry.json` snapshot is also omitted. Named registry datasets
use Harbor's hosted service; explicit registry URLs and paths remain supported.
Registry subset generation requires an explicit `--registry` input file.

Run from the Marin repository root in Harbor's isolated project:

```bash
uv run --isolated --project vendor/harbor --frozen --extra runtime harbor run --help
uv run --isolated --project vendor/harbor --frozen --extra runtime harbor run --dataset terminal-bench@2.0 --agent oracle
```

Harbor is excluded from Marin's uv workspace. Its own lock selects local editable
Marin FineStore and Rigging packages. The normal Marin environment and Iris
`--all-packages` environments do not install Harbor. The `runtime` extra preserves
the deployed Daytona SDK version, 0.200.2.

Adapters retain their own Python requirements and dependency files. Run each
adapter through its documented entry point; their Python ranges differ, so they
are separate projects. The Tau3 package points to this local Harbor source.

Edit Harbor and its local Marin dependencies in the same pull request. Update
Harbor's lock when a dependency manifest changes, and run
`uv run --no-project infra/ci/run_harbor_tests.py` for the offline runtime checks.
For an upstream import, export the selected Git tree, preserve adapters and
templates, reapply the deliberate removals, and update `UPSTREAM.toml`.
Historical evaluation environments remain in `config/external/harbor/pins/`.
