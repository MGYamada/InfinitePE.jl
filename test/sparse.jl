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
    @test InfinitePE._is_acceptable_cg_residual(1.1e-10, 1e-10)
    @test !InfinitePE._is_acceptable_cg_residual(1.1e-9, 1e-10)
    factorizations = InfinitePE._sparse_normal_matsubara_factorizations(sparse_a, beta, cutoff)
    @test pure_pseudofermion_estimators(
        sparse_a,
        beta,
        sparse_fields;
        solver=:direct,
        factorizations=factorizations,
    ).energy ≈ pure_pseudofermion_estimators(sparse_a, beta, sparse_fields; solver=:direct).energy

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

    determinant_estimators = determinant_pseudofermion_estimators(sparse_a, beta; cutoff=cutoff)
    dense_determinant_estimators = determinant_pseudofermion_estimators(dense_a, beta; cutoff=cutoff)
    @test determinant_estimators.energy ≈ dense_determinant_estimators.energy
    @test determinant_estimators.energy_beta_derivative ≈ dense_determinant_estimators.energy_beta_derivative
    log_weight_minus = log_weight(dense_a, beta - delta_beta; cutoff=cutoff)
    log_weight_center = log_weight(dense_a, beta; cutoff=cutoff)
    log_weight_plus = log_weight(dense_a, beta + delta_beta; cutoff=cutoff)
    @test determinant_estimators.energy ≈ -(log_weight_plus - log_weight_minus) / (2 * delta_beta) rtol = 1e-5
    @test determinant_estimators.energy_beta_derivative ≈
          -(log_weight_plus - 2 * log_weight_center + log_weight_minus) / delta_beta^2 rtol = 5e-3 atol = 1e-5
    @test determinant_estimators.cutoff == cutoff

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
    @test run1.field_refresh == :per_attempt
    @test run1.solver == :cg
    @test run1.operator == :matrix_free

    per_sweep_run1 = run_sparse_pseudofermion_mc(
        input,
        beta;
        cutoff=2,
        warmup_sweeps=1,
        sampling_sweeps=2,
        field_refresh=:per_sweep,
        seed=457,
        solver=:cg,
        operator=:matrix_free,
    )
    per_sweep_run2 = run_sparse_pseudofermion_mc(
        input,
        beta;
        cutoff=2,
        warmup_sweeps=1,
        sampling_sweeps=2,
        field_refresh=:per_sweep,
        seed=457,
        solver=:cg,
        operator=:matrix_free,
    )
    @test per_sweep_run1.samples == per_sweep_run2.samples
    @test per_sweep_run1.final_input.gauge.u == per_sweep_run2.final_input.gauge.u
    @test per_sweep_run1.attempted == 3 * length(input.bondset.bonds)
    @test per_sweep_run1.field_refresh == :per_sweep

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

    determinant_run_obs = measure_sparse_pseudofermion_mc(
        input,
        beta,
        run1;
        observable=:determinant,
        solver=:cg,
        operator=:matrix_free,
    )
    @test determinant_run_obs.nsamples == length(run1.samples)
    @test determinant_run_obs.nsites == input.bondset.nsites
    @test isfinite(determinant_run_obs.energy)
    @test isfinite(determinant_run_obs.energy_beta_derivative)

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
    pure_obs_replicated = measure_pure_pseudofermion_observables(
        input,
        beta;
        samples=edmc_run.samples,
        cutoff=2,
        measurement_replicas=2,
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
    @test pure_obs_replicated.nsamples == 2 * edmc_obs.nsamples
    @test pure_obs1.nsites == edmc_obs.nsites
    @test pure_obs_replicated.nsites == edmc_obs.nsites
    @test isfinite(pure_obs1.energy)
    @test isfinite(pure_obs1.energy_beta_derivative)
    @test isfinite(pure_obs_replicated.energy)
    @test isfinite(pure_obs_replicated.energy_beta_derivative)
    @test pure_obs1.specific_heat >= 0

    block_diagnostics1 = pure_pseudofermion_block_diagnostics(
        input,
        beta;
        samples=edmc_run.samples,
        cutoff=2,
        measurement_replicas=3,
        seed=222,
        solver=:cg,
        operator=:matrix_free,
    )
    block_diagnostics2 = pure_pseudofermion_block_diagnostics(
        input,
        beta;
        samples=edmc_run.samples,
        cutoff=2,
        measurement_replicas=3,
        seed=222,
        solver=:cg,
        operator=:matrix_free,
    )
    @test block_diagnostics1 == block_diagnostics2
    @test length(block_diagnostics1) == edmc_obs.nsamples
    @test getproperty.(block_diagnostics1, :gauge_index) == collect(1:edmc_obs.nsamples)
    @test getproperty.(block_diagnostics1, :measurement_replicas) == fill(3, edmc_obs.nsamples)
    @test all(row -> isfinite(row.mean_energy_per_site), block_diagnostics1)
    @test all(row -> isfinite(row.mean_energy_beta_derivative_per_site), block_diagnostics1)
    @test all(row -> row.field_variance_per_site >= -1e-12, block_diagnostics1)
    @test all(row -> row.raw_field_specific_heat_per_site ≈ beta^2 * row.field_variance_minus_derivative_per_site, block_diagnostics1)
    @test all(row -> row.field_specific_heat_per_site == max(0.0, row.raw_field_specific_heat_per_site), block_diagnostics1)

    diagnostics1 = pure_pseudofermion_cutoff_diagnostics(
        input,
        beta;
        samples=edmc_run.samples,
        cutoffs=[1, 2],
        measurement_replicas=2,
        seed=111,
        solver=:cg,
        operator=:matrix_free,
    )
    diagnostics2 = pure_pseudofermion_cutoff_diagnostics(
        input,
        beta;
        samples=edmc_run.samples,
        cutoffs=[1, 2],
        measurement_replicas=2,
        seed=111,
        solver=:cg,
        operator=:matrix_free,
    )
    @test diagnostics1 == diagnostics2
    @test getproperty.(diagnostics1, :cutoff) == [1, 2]
    @test getproperty.(diagnostics1, :measurement_replicas) == [2, 2]
    @test getproperty.(diagnostics1, :ngauge_samples) == [edmc_obs.nsamples, edmc_obs.nsamples]
    @test all(row -> row.reference_energy_per_site == edmc_obs.energy, diagnostics1)
    @test all(row -> isfinite(row.energy_bias_per_site) && isfinite(row.specific_heat_bias_per_site), diagnostics1)
    @test all(row -> isfinite(row.variance_minus_derivative_per_site), diagnostics1)
    @test all(row -> row.raw_specific_heat_per_site ≈ beta^2 * row.variance_minus_derivative_per_site, diagnostics1)
    @test all(row -> row.specific_heat_per_site == max(0.0, row.raw_specific_heat_per_site), diagnostics1)
    @test all(row -> row.energy_variance_per_site >= -1e-12, diagnostics1)

    @test_throws ArgumentError refresh_sparse_real_pseudofermions(sparse_a, beta; cutoff=0)
    @test_throws ArgumentError sparse_pseudofermion_action(sparse_a, beta, Vector{Float64}[])
    @test_throws ArgumentError sparse_pseudofermion_action(sparse_a, beta, [[1.0]])
    @test_throws ArgumentError sparse_pseudofermion_action(sparse_a, beta, sparse_fields; solver=:unknown)
    @test_throws ArgumentError sparse_pseudofermion_action(sparse_a, beta, sparse_fields; operator=:unknown)
    @test_throws ArgumentError run_sparse_pseudofermion_mc(input, beta; cutoff=0, sampling_sweeps=1)
    @test_throws ArgumentError run_sparse_pseudofermion_mc(input, beta; cutoff=1, sampling_sweeps=1, seed=1, rng=MersenneTwister(1))
    @test_throws ArgumentError run_sparse_pseudofermion_mc(input, beta; cutoff=1, sampling_sweeps=1, field_refresh=:never)
    @test_throws ArgumentError measure_sparse_pseudofermion_mc(input, beta, run1; observable=:unknown)
    @test_throws ArgumentError scan_sparse_pseudofermion_temperatures(input, Float64[]; cutoff=1, sampling_sweeps=1)
    @test_throws ArgumentError measure_pure_pseudofermion_observables(input, beta; samples=EDMC.Z2GaugeField[], cutoff=1)
    @test_throws ArgumentError measure_pure_pseudofermion_observables(input, beta; samples=edmc_run.samples, cutoff=1, measurement_replicas=0)
    @test_throws ArgumentError pure_pseudofermion_block_diagnostics(input, beta; samples=edmc_run.samples, cutoff=1, measurement_replicas=0)
    @test_throws ArgumentError pure_pseudofermion_cutoff_diagnostics(input, beta; samples=edmc_run.samples, cutoffs=Int[])
end
