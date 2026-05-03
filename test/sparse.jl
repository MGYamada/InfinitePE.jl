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
    @test sparse_pseudofermion_action(sparse_a, beta, sparse_fields; solver=:gmres, operator=:matrix) ≈
          sparse_pseudofermion_action(sparse_a, beta, sparse_fields; solver=:direct) atol = 1e-9
    @test sparse_pseudofermion_action(sparse_a, beta, sparse_fields; solver=:gmres, operator=:matrix_free) ≈
          sparse_pseudofermion_action(sparse_a, beta, sparse_fields; solver=:direct) atol = 1e-9
    @test sparse_pseudofermion_action(input, beta, sparse_fields) ≈
          pseudofermion_action(input, beta, sparse_fields)

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
        solver=:gmres,
        operator=:matrix_free,
    )
    run2 = run_sparse_pseudofermion_mc(
        input,
        beta;
        cutoff=2,
        warmup_sweeps=1,
        sampling_sweeps=2,
        seed=456,
        solver=:gmres,
        operator=:matrix_free,
    )
    @test run1.samples == run2.samples
    @test run1.final_input.gauge.u == run2.final_input.gauge.u
    @test length(run1.samples) == 2
    @test run1.attempted == 3 * length(input.bondset.bonds)
    @test run1.accepted == run1.warmup_accepted + run1.sampling_accepted
    @test 0.0 <= run1.acceptance_rate <= 1.0

    @test_throws ArgumentError refresh_sparse_real_pseudofermions(sparse_a, beta; cutoff=0)
    @test_throws ArgumentError sparse_pseudofermion_action(sparse_a, beta, Vector{Float64}[])
    @test_throws ArgumentError sparse_pseudofermion_action(sparse_a, beta, [[1.0]])
    @test_throws ArgumentError sparse_pseudofermion_action(sparse_a, beta, sparse_fields; solver=:unknown)
    @test_throws ArgumentError sparse_pseudofermion_action(sparse_a, beta, sparse_fields; operator=:unknown)
    @test_throws ArgumentError run_sparse_pseudofermion_mc(input, beta; cutoff=0, sampling_sweeps=1)
    @test_throws ArgumentError run_sparse_pseudofermion_mc(input, beta; cutoff=1, sampling_sweeps=1, seed=1, rng=MersenneTwister(1))
end
