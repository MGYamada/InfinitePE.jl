using Random
using SparseArrays

@testset "Sparse infinite product expansion" begin
    lat = generate_honeycomb(2, 2, TypeI())
    input = EDMC.lattice_to_edmc(lat)
    beta = 1.7
    cutoff = 4

    dense_a = build_majorana_matrix(input)
    sparse_a = build_sparse_majorana_matrix(input)
    @test sparse_a isa SparseMatrixCSC
    @test Matrix(sparse_a) ≈ dense_a

    for n in 0:(cutoff - 1)
        @test Matrix(sparse_matsubara_matrix(sparse_a, beta, n)) ≈ matsubara_matrix(dense_a, beta, n)
        vector = collect(1.0:lat.nsites)
        @test sparse_matsubara_mul(sparse_a, beta, n, vector) ≈ matsubara_matrix(dense_a, beta, n) * vector
        @test sparse_matsubara_transpose_mul(sparse_a, beta, n, vector) ≈ transpose(matsubara_matrix(dense_a, beta, n)) * vector
    end

    rng1 = MersenneTwister(123)
    rng2 = MersenneTwister(123)
    dense_fields = refresh_real_pseudofermions(dense_a, beta; cutoff=cutoff, rng=rng1)
    sparse_fields = refresh_sparse_real_pseudofermions(sparse_a, beta; cutoff=cutoff, rng=rng2)
    @test sparse_fields ≈ dense_fields

    @test sparse_pseudofermion_action(sparse_a, beta, sparse_fields; solver=:direct) ≈
          pseudofermion_action(dense_a, beta, sparse_fields)
    @test sparse_pseudofermion_action(sparse_a, beta, sparse_fields; solver=:cg, operator=:matrix) ≈
          sparse_pseudofermion_action(sparse_a, beta, sparse_fields; solver=:direct) atol = 1e-9
    @test sparse_pseudofermion_action(sparse_a, beta, sparse_fields; solver=:cg, operator=:matrix_free) ≈
          sparse_pseudofermion_action(sparse_a, beta, sparse_fields; solver=:direct) atol = 1e-9
    @test sparse_pseudofermion_action(input, beta, sparse_fields) ≈
          pseudofermion_action(input, beta, sparse_fields)

    estimators = pure_pseudofermion_estimators(
        sparse_a,
        beta,
        sparse_fields;
        solver=:cg,
        operator=:matrix_free,
        tol=1e-11,
    )
    delta_beta = 1e-5
    action_minus = sparse_pseudofermion_action(
        sparse_a,
        beta - delta_beta,
        sparse_fields;
        solver=:cg,
        operator=:matrix_free,
        tol=1e-11,
    )
    action_center = sparse_pseudofermion_action(
        sparse_a,
        beta,
        sparse_fields;
        solver=:cg,
        operator=:matrix_free,
        tol=1e-11,
    )
    action_plus = sparse_pseudofermion_action(
        sparse_a,
        beta + delta_beta,
        sparse_fields;
        solver=:cg,
        operator=:matrix_free,
        tol=1e-11,
    )
    @test estimators.energy ≈ (action_plus - action_minus) / (2 * delta_beta) rtol = 1e-5
    @test estimators.energy_beta_derivative ≈
          (action_plus - 2 * action_center + action_minus) / delta_beta^2 rtol = 5e-3 atol = 1e-5
    @test estimators.cutoff == cutoff
    @test estimators.solver == :cg
    @test estimators.operator == :matrix_free

    flipped = EDMC.flip_gauge(input, 1)
    dense_after = build_majorana_matrix(flipped)
    sparse_after = build_sparse_majorana_matrix(flipped)
    @test delta_sparse_pseudofermion_action(sparse_a, sparse_after, beta, sparse_fields) ≈
          delta_pseudofermion_action(dense_a, dense_after, beta, sparse_fields)
    @test delta_sparse_pseudofermion_action(input, flipped, beta, sparse_fields) ≈
          delta_pseudofermion_action(input, flipped, beta, sparse_fields)

    run1 = run_sparse_pseudofermion_mc(
        input,
        beta;
        cutoff=2,
        warmup_sweeps=1,
        sampling_sweeps=2,
        seed=456,
        solver=:cg,
        operator=:matrix_free,
    )
    run2 = run_sparse_pseudofermion_mc(
        input,
        beta;
        cutoff=2,
        warmup_sweeps=1,
        sampling_sweeps=2,
        seed=456,
        solver=:cg,
        operator=:matrix_free,
    )
    @test run1.samples == run2.samples
    @test run1.final_input.gauge.u == run2.final_input.gauge.u
    @test length(run1.samples) == 2
    @test run1.attempted == 3 * length(input.bondset.bonds)
    @test run1.accepted == run1.warmup_accepted + run1.sampling_accepted
    @test 0.0 <= run1.acceptance_rate <= 1.0
    @test run1.cutoff == 2
    @test run1.solver == :cg
    @test run1.operator == :matrix_free

    pure_run_obs = measure_sparse_pseudofermion_mc(
        input,
        beta,
        run1;
        observable=:pure,
        seed=321,
        solver=:cg,
        operator=:matrix_free,
    )
    auto_large_obs = measure_sparse_pseudofermion_mc(
        input,
        beta,
        run1;
        observable=:auto,
        seed=321,
        solver=:cg,
        operator=:matrix_free,
        large_lattice_threshold=1,
    )
    @test pure_run_obs.energy == auto_large_obs.energy
    @test pure_run_obs.specific_heat == auto_large_obs.specific_heat

    scan = scan_sparse_pseudofermion_temperatures(
        input,
        [0.8, 1.6];
        cutoff=2,
        warmup_sweeps=1,
        sampling_sweeps=2,
        seed=789,
        solver=:cg,
        operator=:matrix_free,
    )
    rows = sparse_pseudofermion_comparison_table(scan; metadata=(observable="EDMC-compatible",))
    @test scan.temperatures == [0.8, 1.6]
    @test length(scan.observables) == 2
    @test length(scan.runs) == 2
    @test rows[1].method == :SparsePseudofermionIPE
    @test rows[1].metadata.cutoff == 2
    @test rows[1].metadata.solver == :cg
    @test rows[1].metadata.operator == :matrix_free
    @test all(row -> isfinite(row.energy_per_site) && isfinite(row.specific_heat_per_site), rows)

    pure_scan = scan_sparse_pseudofermion_temperatures(
        input,
        [0.8];
        cutoff=2,
        warmup_sweeps=1,
        sampling_sweeps=2,
        seed=246,
        solver=:cg,
        operator=:matrix_free,
        observable=:pure,
        measurement_seed=135,
    )
    @test pure_scan.observable == :pure
    @test length(pure_scan.observables) == 1
    @test isfinite(pure_scan.observables[1].energy)

    edmc_run = EDMC.run_edmc(input, beta; warmup_sweeps=1, sampling_sweeps=3, seed=654)
    edmc_obs = EDMC.measure(input, beta; samples=edmc_run.samples)
    pure_obs1 = measure_pure_pseudofermion_observables(
        input,
        beta;
        samples=edmc_run.samples,
        cutoff=2,
        seed=987,
        solver=:cg,
        operator=:matrix_free,
    )
    pure_obs2 = measure_pure_pseudofermion_observables(
        input,
        beta;
        samples=edmc_run.samples,
        cutoff=2,
        seed=987,
        solver=:cg,
        operator=:matrix_free,
    )
    @test pure_obs1.energy == pure_obs2.energy
    @test pure_obs1.specific_heat == pure_obs2.specific_heat
    @test pure_obs1.nsamples == edmc_obs.nsamples
    @test pure_obs1.nsites == edmc_obs.nsites
    @test isfinite(pure_obs1.energy)
    @test isfinite(pure_obs1.energy_beta_derivative)
    @test pure_obs1.specific_heat >= 0

    diagnostics1 = pure_pseudofermion_cutoff_diagnostics(
        input,
        beta;
        samples=edmc_run.samples,
        cutoffs=[1, 2],
        seed=111,
        solver=:cg,
        operator=:matrix_free,
    )
    diagnostics2 = pure_pseudofermion_cutoff_diagnostics(
        input,
        beta;
        samples=edmc_run.samples,
        cutoffs=[1, 2],
        seed=111,
        solver=:cg,
        operator=:matrix_free,
    )
    @test diagnostics1 == diagnostics2
    @test getproperty.(diagnostics1, :cutoff) == [1, 2]
    @test all(row -> row.reference_energy_per_site == edmc_obs.energy, diagnostics1)
    @test all(row -> isfinite(row.energy_bias_per_site) && isfinite(row.specific_heat_bias_per_site), diagnostics1)
    @test all(row -> row.energy_variance_per_site >= -1e-12, diagnostics1)

    @test_throws ArgumentError refresh_sparse_real_pseudofermions(sparse_a, beta; cutoff=0)
    @test_throws ArgumentError sparse_pseudofermion_action(sparse_a, beta, Vector{Float64}[])
    @test_throws ArgumentError sparse_pseudofermion_action(sparse_a, beta, [[1.0]])
    @test_throws ArgumentError sparse_pseudofermion_action(sparse_a, beta, sparse_fields; solver=:unknown)
    @test_throws ArgumentError sparse_pseudofermion_action(sparse_a, beta, sparse_fields; operator=:unknown)
    @test_throws ArgumentError run_sparse_pseudofermion_mc(input, beta; cutoff=0, sampling_sweeps=1)
    @test_throws ArgumentError run_sparse_pseudofermion_mc(input, beta; cutoff=1, sampling_sweeps=1, seed=1, rng=MersenneTwister(1))
    @test_throws ArgumentError measure_sparse_pseudofermion_mc(input, beta, run1; observable=:unknown)
    @test_throws ArgumentError scan_sparse_pseudofermion_temperatures(input, Float64[]; cutoff=1, sampling_sweeps=1)
    @test_throws ArgumentError measure_pure_pseudofermion_observables(input, beta; samples=EDMC.Z2GaugeField[], cutoff=1)
    @test_throws ArgumentError pure_pseudofermion_cutoff_diagnostics(input, beta; samples=edmc_run.samples, cutoffs=Int[])
end
