# Repository instructions

LuxRecurrentLayers.jl is a Julia 1.10+ package that implements recurrent cells for Lux.jl.
Treat Lux's recurrent-cell interface and the exported cell constructors as the public API.

## Commands

Run commands from the repository root.

- Full test suite: `julia --project=. -e 'using Pkg; Pkg.test()'`
- Prepare the standalone test environment:
  `julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'`
- CPU-only tests (after preparation): `BACKEND_GROUP=cpu julia --project=test test/runtests.jl`
- Package QA (after preparation): `julia --project=test test/qa.jl`
- Build docs: `julia --project=docs docs/make.jl`
- Format all Julia files: `julia -e 'using JuliaFormatter; format(".")'`
- Check formatting without edits:
  `julia -e 'using JuliaFormatter; @assert format(".", overwrite=false)'`

`Pkg.test()` may probe CUDA and AMDGPU but skips unavailable functional devices. Prefer the
CPU-only command for quick local iteration. Documentation builds perform external link checks
and therefore require network access. JuliaFormatter is a developer tool and is not a package
dependency; install it in a global or dedicated tooling environment.

## Project map

- `src/LuxRecurrentLayers.jl`: module imports, exports, and source includes.
- `src/generics.jl`: shared recurrent-cell behavior.
- `src/cells/`: one implementation file per recurrent-cell family.
- `test/cells.jl`: behavioral, differentiation, and device tests.
- `test/setups.jl`: shared test fixtures and the canonical list of tested cells.
- `test/qa.jl`: Aqua and JET package-quality checks.
- `docs/src/api/cells/`: one public API page per documented cell.
- `.github/workflows/CI.yml`: supported Julia/OS matrix, coverage, QA, and docs deployment.

## Implementation conventions

- Follow `.JuliaFormatter.toml` and the SciML Julia style.
- Preserve the Lux call contract: layers return `(output, state)`; recurrent cells generally
  return `((output, carry), state)`.
- Use explicit Lux parameter/state initialization methods and preserve `use_*` and `train_*`
  constructor options.
- Keep public constructors documented with their keyword defaults and expected input/output
  shapes. Add new exports and `include` entries in `src/LuxRecurrentLayers.jl`.
- Use `Float32`-compatible initializers and avoid unnecessary allocations or scalar indexing
  that would break GPU execution.
- Match nearby code. For example, constructors use pair dimensions and keyword arguments:

```julia
cell = AntisymmetricRNNCell(3 => 5; use_bias=true, train_state=false)
ps, st = Lux.setup(rng, cell)
(y, carry), st = cell(x, ps, st)
```

## Testing expectations

- Every code change needs a focused regression or behavior test.
- For a new cell, add it to `RECURRENT_CELLS` in `test/setups.jl` and test constructor options,
  output/carry shapes, parameter and state behavior, automatic differentiation, and supported
  devices using existing helpers.
- Public API changes require corresponding pages under `docs/src/api/cells/` and navigation
  updates where needed.
- Run the narrowest relevant test while iterating, then the full CPU suite. Run package QA and
  docs for public API, dependency, export, or docstring changes.
- Do not weaken, delete, or skip a failing test merely to make the suite pass.

## Git and collaboration

- Follow [SciML ColPrac](https://github.com/SciML/ColPrac).
- Keep pull requests focused. Explain the problem, implementation, and verification.
- Match the repository's conventional commit subjects, such as `fix: ...`, `test: ...`,
  `docs: ...`, and `chore: ...`.
- Do not bump the package version for an ordinary pull request; maintainers handle releases.
- Preserve unrelated working-tree changes and never rewrite history unless explicitly asked.

## Boundaries

Always:

- Inspect adjacent implementations and tests before changing a cell.
- Update tests and public documentation with behavior or API changes.
- Report which verification commands ran and any environment-related skips.

Ask first:

- Add or remove dependencies, change compatibility bounds, or alter CI/release workflows.
- Introduce breaking API changes, deprecations, or broad mechanical refactors.
- Regenerate committed manifests or change supported Julia/device versions.

Never:

- Commit secrets, credentials, local settings, generated documentation builds, or coverage data.
- Edit dependency source code or generated artifacts to work around a package defect.
- Claim GPU or cross-platform verification that was not actually run.
