using Random

@testset "EDMC bond mapping" begin
    lat = generate_honeycomb(2, 2, TypeI())
    bondset = EDMC.extract_kitaev_bonds(lat)

    @test bondset.lattice_kind == :honeycomb
    @test bondset.dims == lat.dims
    @test bondset.nsites == lat.nsites
    @test !isempty(bondset.bonds)
    @test all(b -> b.kind in (:x, :y, :z), bondset.bonds)
    @test getfield.(bondset.bonds, :index) == collect(1:length(bondset.bonds))

    reversed_lat = Lattice(
        kind=lat.kind,
        dims=lat.dims,
        boundary=lat.boundary,
        nsites=lat.nsites,
        edges=reverse(lat.edges),
    )
    reversed_bondset = EDMC.extract_kitaev_bonds(reversed_lat)
    @test [(b.src, b.dst, b.kind, b.wrapped) for b in reversed_bondset.bonds] ==
          [(b.src, b.dst, b.kind, b.wrapped) for b in bondset.bonds]

    gauge = EDMC.initialize_z2_gauge(bondset)
    @test gauge.u == fill(Int8(1), length(bondset.bonds))
    @test_throws ArgumentError EDMC.initialize_z2_gauge(bondset; value=0)
end

@testset "EDMC delta free energy" begin
    lat = generate_honeycomb(2, 2, TypeI())
    input = EDMC.lattice_to_edmc(lat)
    beta = 2.0

    hamiltonian = EDMC.build_hamiltonian(input)
    @test size(hamiltonian) == (lat.nsites, lat.nsites)
    @test hamiltonian == hamiltonian'

    result = EDMC.diagonalize(hamiltonian)
    @test length(result.eigenvalues) == lat.nsites
    @test all(isfinite, result.eigenvalues)
    @test length(EDMC.majorana_energies(result)) == div(lat.nsites, 2)

    free_energy = EDMC.free_energy(input, beta)
    @test isfinite(free_energy)
    @test EDMC.free_energy(hamiltonian, beta) ≈ free_energy
    @test EDMC.free_energy(result, beta) ≈ free_energy

    delta_f_1 = EDMC.delta_free_energy(input, 1, beta)
    delta_f_2 = EDMC.delta_free_energy(input, 1, beta)
    @test isfinite(delta_f_1)
    @test delta_f_1 == delta_f_2

    probability = EDMC.acceptance_probability(delta_f_1, beta)
    @test 0.0 <= probability <= 1.0
    @test isfinite(EDMC.free_energy(input, 1.0e-8))
    @test isfinite(EDMC.free_energy(input, 1.0e8))
    @test_throws ArgumentError EDMC.free_energy(input, 0.0)
    @test_throws ArgumentError EDMC.diagonalize([0.0 1.0; NaN 0.0])
end

@testset "EDMC acceptance range" begin
    lat = generate_honeycomb(2, 2, TypeI())
    input = EDMC.lattice_to_edmc(lat)
    beta = 2.0

    run1 = EDMC.run_edmc(input, beta; warmup_sweeps=2, sampling_sweeps=3, seed=1234)
    run2 = EDMC.run_edmc(input, beta; warmup_sweeps=2, sampling_sweeps=3, seed=1234)

    @test run1.final_input.gauge.u == run2.final_input.gauge.u
    @test [sample.u for sample in run1.samples] == [sample.u for sample in run2.samples]
    @test length(run1.samples) == 3
    @test run1.attempted == 5 * length(input.bondset.bonds)
    @test run1.accepted == run1.warmup_accepted + run1.sampling_accepted
    @test 0.0 <= EDMC.acceptance_rate(run1) <= 1.0
    @test 0.0 <= EDMC.warmup_acceptance_rate(run1) <= 1.0
    @test 0.0 <= EDMC.sampling_acceptance_rate(run1) <= 1.0

    proposal_result = EDMC.attempt_bond_flip(input, MersenneTwister(99), beta)
    @test 1 <= proposal_result.proposal.bond_index <= length(input.bondset.bonds)
    @test 0.0 <= proposal_result.proposal.probability <= 1.0
    @test proposal_result.input isa EDMC.KitaevHamiltonianInput

    @test_throws ArgumentError EDMC.run_edmc(input, beta; warmup_sweeps=-1, sampling_sweeps=1, seed=1)
    @test_throws ArgumentError EDMC.run_edmc(input, beta; warmup_sweeps=1, sampling_sweeps=1, seed=1, rng=MersenneTwister(1))
end

@testset "EDMC finite observables and comparison rows" begin
    lat = generate_honeycomb(2, 2, TypeI())
    input = EDMC.lattice_to_edmc(lat)
    temperatures = [0.5, 1.0, 2.0]

    run = EDMC.run_edmc(input, 1.0; warmup_sweeps=1, sampling_sweeps=3, seed=42)
    obs = EDMC.measure(input, 1.0; samples=run.samples)
    @test obs.nsamples == 3
    @test obs.nsites == lat.nsites
    @test isfinite(obs.energy)
    @test isfinite(obs.energy2)
    @test isfinite(obs.energy_beta_derivative)
    @test obs.specific_heat >= 0

    scan1 = EDMC.scan_temperatures(input, temperatures; warmup_sweeps=1, sampling_sweeps=2, seed=2024)
    scan2 = EDMC.scan_temperatures(input, temperatures; warmup_sweeps=1, sampling_sweeps=2, seed=2024)

    @test scan1.temperatures == temperatures
    @test length(scan1.observables) == length(temperatures)
    @test length(scan1.runs) == length(temperatures)
    @test [o.energy for o in scan1.observables] == [o.energy for o in scan2.observables]
    @test [o.specific_heat for o in scan1.observables] == [o.specific_heat for o in scan2.observables]
    @test all(o -> isfinite(o.energy) && isfinite(o.energy2) && isfinite(o.energy_beta_derivative), scan1.observables)
    @test all(o -> o.specific_heat >= 0, scan1.observables)
    @test all(run -> 0.0 <= EDMC.acceptance_rate(run) <= 1.0, scan1.runs)

    rows = EDMC.comparison_table(scan1)
    @test length(rows) == length(temperatures)
    @test keys(rows[1]) == (
        :method,
        :temperature,
        :beta,
        :energy_per_site,
        :energy2_per_site2,
        :energy_beta_derivative_per_site,
        :specific_heat_per_site,
        :nsamples,
        :nsites,
        :metadata,
    )
    @test rows[1].method == :EDMC
    @test rows[1].temperature == scan1.observables[1].temperature
    @test rows[1].energy_per_site == scan1.observables[1].energy
    @test isfinite(rows[1].specific_heat_per_site)

    @test_throws ArgumentError EDMC.measure(input, 1.0; samples=EDMC.Z2GaugeField[])
    @test_throws ArgumentError EDMC.scan_temperatures(input, Float64[]; sampling_sweeps=1, seed=1)
    @test_throws ArgumentError EDMC.scan_temperatures(input, [1.0, 0.0]; sampling_sweeps=1, seed=1)
end

@testset "EDMC Majorana energy convention" begin
    lat = generate_honeycomb(1, 1, TypeI())
    input = EDMC.lattice_to_edmc(lat)
    couplings = (x=0.0, y=0.0, z=1.0)

    result = EDMC.diagonalize(EDMC.build_hamiltonian(input; couplings=couplings))
    @test result.eigenvalues == [-1.0, 1.0]
    @test EDMC.majorana_energies(result) == [2.0]

    beta = 1.7
    edmc_obs = EDMC.measure(input, beta; couplings=couplings)
    fulled_obs = FullED.thermal_observables(FullED.lattice_to_fulled(lat), beta; couplings=couplings)
    @test edmc_obs.energy ≈ fulled_obs.energy
    @test edmc_obs.specific_heat ≈ fulled_obs.specific_heat
end
