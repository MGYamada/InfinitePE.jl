using LinearAlgebra
using Random

function _with_gauge(input, gauge)
    return EDMC.KitaevHamiltonianInput(input.bondset, EDMC.Z2GaugeField(Int8.(gauge)))
end

function _exact_log_weight_mean(input, beta, cutoff)
    nbonds = length(input.bondset.bonds)
    weights = Float64[]
    values = Float64[]
    for mask in 0:(2^nbonds - 1)
        gauge = [((mask >> (i - 1)) & 1) == 1 ? Int8(1) : Int8(-1) for i in 1:nbonds]
        state = _with_gauge(input, gauge)
        value = log_weight_infinite_product(state, beta; cutoff=cutoff)
        push!(values, value)
        push!(weights, exp(value))
    end
    return sum(weights .* values) / sum(weights)
end

function _dense_determinant_mc_log_weight_mean(input, beta, cutoff; seed, warmup=1_000, samples=10_000)
    rng = MersenneTwister(seed)
    current = input
    values = Float64[]
    nbonds = length(input.bondset.bonds)
    current_logw = log_weight_infinite_product(current, beta; cutoff=cutoff)
    for step in 1:(warmup + samples)
        bond = rand(rng, 1:nbonds)
        proposed = EDMC.flip_gauge(current, bond)
        proposed_logw = log_weight_infinite_product(proposed, beta; cutoff=cutoff)
        if rand(rng) < acceptance_probability_logweight(proposed_logw - current_logw)
            current = proposed
            current_logw = proposed_logw
        end
        step > warmup && push!(values, current_logw)
    end
    return sum(values) / length(values)
end

function _dense_pseudofermion_mc_log_weight_mean(input, beta, cutoff; seed, warmup=1_000, samples=10_000)
    rng = MersenneTwister(seed)
    current = input
    values = Float64[]
    nbonds = length(input.bondset.bonds)
    for step in 1:(warmup + samples)
        fields = refresh_real_pseudofermions(current, beta; cutoff=cutoff, rng=rng)
        bond = rand(rng, 1:nbonds)
        proposed = EDMC.flip_gauge(current, bond)
        delta_action = delta_pseudofermion_action(current, proposed, beta, fields)
        if rand(rng) < acceptance_probability_pseudofermion(delta_action)
            current = proposed
        end
        step > warmup && push!(values, log_weight_infinite_product(current, beta; cutoff=cutoff))
    end
    return sum(values) / length(values)
end

@testset "Infinite product expansion" begin
    beta = 1.7
    cutoff = 12
    lambda = 0.8
    a = [0.0 -lambda; lambda 0.0]
    h = im .* a

    @test matsubara_frequency(0, beta) ≈ pi / beta
    @test matsubara_frequency(2, beta) ≈ 5pi / beta
    @test majorana_matrix(h) ≈ a
    @test logdet_matsubara_dense(a, beta, 0) ≈ log(1 + lambda^2 / matsubara_frequency(0, beta)^2)

    finite_product = 0.0
    for n in 0:(cutoff - 1)
        omega = matsubara_frequency(n, beta)
        finite_product += log(1 + lambda^2 / omega^2)
    end

    @test log_weight_infinite_product(a, beta; cutoff=cutoff) ≈ finite_product
    @test log_weight_infinite_product(h, beta; cutoff=cutoff) ≈ finite_product
    @test log_weight(a, beta; cutoff=cutoff) ≈ finite_product

    lambda_after = 1.1
    after = [0.0 -lambda_after; lambda_after 0.0]
    delta = delta_log_weight_infinite_product(a, after, beta; cutoff=cutoff)
    @test delta ≈ log_weight_infinite_product(after, beta; cutoff=cutoff) - log_weight_infinite_product(a, beta; cutoff=cutoff)
    @test delta_log_weight(a, after, beta; cutoff=cutoff) ≈ delta
    @test acceptance_probability_logweight(delta) == 1.0
    @test acceptance_probability(delta) == 1.0
    @test 0.0 < acceptance_probability_logweight(-abs(delta)) < 1.0

    xi = [0.2, -1.3]
    field = transpose(matsubara_matrix(a, beta, 0)) * xi
    @test pseudofermion_action(a, beta, [field]) ≈ 0.5 * dot(xi, xi)
    @test acceptance_probability_pseudofermion(-0.3) == 1.0
    @test acceptance_probability_pseudofermion(0.3) ≈ exp(-0.3)

    lat = generate_honeycomb(2, 2, TypeI())
    input = EDMC.lattice_to_edmc(lat)
    flipped = EDMC.flip_gauge(input, 1)
    beta_edmc = 2.0
    target_delta_logw = -beta_edmc * EDMC.delta_free_energy(input, flipped, beta_edmc)
    product_delta_10 = delta_log_weight_infinite_product(input, flipped, beta_edmc; cutoff=10)
    product_delta_100 = delta_log_weight_infinite_product(input, flipped, beta_edmc; cutoff=100)

    @test build_majorana_matrix(input) ≈ 2 .* majorana_matrix(EDMC.build_hamiltonian(input))
    @test abs(product_delta_100 - target_delta_logw) < abs(product_delta_10 - target_delta_logw)
    @test product_delta_100 ≈ target_delta_logw atol = 1e-9
    @test log_weight_infinite_product(input, beta_edmc; cutoff=10) ≈
          log_weight_infinite_product(build_majorana_matrix(input), beta_edmc; cutoff=10)

    rng1 = MersenneTwister(2024)
    rng2 = MersenneTwister(2024)
    fields_from_input = refresh_real_pseudofermions(input, beta_edmc; cutoff=3, rng=rng1)
    fields_from_matrix = refresh_real_pseudofermions(build_majorana_matrix(input), beta_edmc; cutoff=3, rng=rng2)
    @test fields_from_input == fields_from_matrix
    @test length(fields_from_input) == 3
    @test all(field -> length(field) == lat.nsites && all(isfinite, field), fields_from_input)

    action_input = pseudofermion_action(input, beta_edmc, fields_from_input)
    action_matrix = pseudofermion_action(build_majorana_matrix(input), beta_edmc, fields_from_input)
    @test action_input ≈ action_matrix
    @test delta_pseudofermion_action(input, flipped, beta_edmc, fields_from_input) ≈
          pseudofermion_action(flipped, beta_edmc, fields_from_input) - action_input

    mc_lat = generate_honeycomb(2, 1, TypeI())
    mc_input = EDMC.lattice_to_edmc(mc_lat)
    mc_beta = 1.3
    mc_cutoff = 4
    exact_mean_logw = _exact_log_weight_mean(mc_input, mc_beta, mc_cutoff)
    det_mean_logw = _dense_determinant_mc_log_weight_mean(mc_input, mc_beta, mc_cutoff; seed=1)
    pf_mean_logw = _dense_pseudofermion_mc_log_weight_mean(mc_input, mc_beta, mc_cutoff; seed=11)

    @test det_mean_logw ≈ exact_mean_logw atol = 0.06
    @test pf_mean_logw ≈ exact_mean_logw atol = 0.06
    @test det_mean_logw ≈ pf_mean_logw atol = 0.06

    temperatures = [0.8, 1.6]
    edmc_scan = EDMC.scan_temperatures(
        mc_input,
        temperatures;
        warmup_sweeps=5,
        sampling_sweeps=25,
        seed=21,
    )
    pf_scan = scan_pseudofermion_temperatures(
        mc_input,
        temperatures;
        cutoff=mc_cutoff,
        warmup_sweeps=5,
        sampling_sweeps=25,
        seed=21,
    )
    pf_rows = pseudofermion_comparison_table(pf_scan; metadata=(observable="EDMC-compatible",))
    edmc_rows = EDMC.comparison_table(edmc_scan)

    @test pf_scan.temperatures == edmc_scan.temperatures
    @test length(pf_scan.observables) == length(edmc_scan.observables)
    @test all(run -> 0.0 <= run.acceptance_rate <= 1.0, pf_scan.runs)
    @test pf_rows[1].method == :PseudofermionIPE
    @test pf_rows[1].metadata.cutoff == mc_cutoff
    @test pf_rows[1].metadata.observable == "EDMC-compatible"
    @test keys(pf_rows[1]) == keys(edmc_rows[1])
    @test all(row -> isfinite(row.energy_per_site) && isfinite(row.specific_heat_per_site), pf_rows)

    @test_throws ArgumentError matsubara_frequency(-1, beta)
    @test_throws ArgumentError log_weight(a, beta; cutoff=0)
    @test_throws ArgumentError log_weight([0.0 1.0; 1.0 0.0], beta; cutoff=cutoff)
    @test_throws ArgumentError refresh_real_pseudofermions(a, beta; cutoff=0)
    @test_throws ArgumentError pseudofermion_action(a, beta, Vector{Float64}[])
    @test_throws ArgumentError pseudofermion_action(a, beta, [[1.0]])
    @test_throws ArgumentError run_pseudofermion_mc(mc_input, mc_beta; cutoff=0, sampling_sweeps=1)
    @test_throws ArgumentError run_pseudofermion_mc(mc_input, mc_beta; cutoff=1, sampling_sweeps=1, seed=1, rng=MersenneTwister(1))
end
