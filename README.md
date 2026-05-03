# InfinitePE.jl

## EDMC Comparison Baseline

For a lightweight qualitative comparison with PRL 113, 197205, run the EDMC
baseline temperature scan:

```sh
julia --project=. scripts/edmc_prl113197205_baseline.jl
```

Full ED and Fig. 5-style plotting baselines are also available:

```sh
julia --project=. scripts/fulled_prl113197205_baseline.jl
julia --project=. scripts/plot_prl113197205_fig5.jl
```

The defaults use a small honeycomb lattice (`Lx=2`, `Ly=2`) with short warmup
and sampling runs, so they are meant only to check qualitative trends such as
the energy curve and specific-heat peak shape. A custom scan can be started as:

```sh
julia --project=. scripts/edmc_prl113197205_baseline.jl \
  --Lx 3 --Ly 3 --warmup 20 --sampling 50 --seed 2024 \
  --tmin 0.05 --tmax 3.0 --ntemps 40 \
  --output edmc_baseline_L3.csv
```

The output is CSV with one row per temperature. Key columns are
`temperature`, `energy_per_site`, `energy2_per_site2`,
`energy_beta_derivative_per_site`, `specific_heat_per_site`, and acceptance
rates, which are the intended comparison fields for plotting against an
Infinite PE result or the PRL baseline trend.

### Comparison API

EDMC scan results can be converted to method-agnostic comparison rows:

```julia
rows = InfinitePE.EDMC.comparison_table(scan)
```

Each row is a `NamedTuple` with stable fields:
`method`, `temperature`, `beta`, `energy_per_site`,
`energy2_per_site2`, `energy_beta_derivative_per_site`,
`specific_heat_per_site`, `nsamples`, `nsites`, and `metadata`.
Infinite PE comparison outputs should use the same field names.

The intended public EDMC API is the exported surface under `InfinitePE.EDMC`,
including lattice conversion, Hamiltonian construction, diagonalization,
Metropolis updates, temperature scans, measurements, and comparison rows.
Helper names beginning with `_` are internal implementation details and should
not be used by external scripts.
