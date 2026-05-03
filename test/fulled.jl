@testset "FullED Hamiltonian and observables" begin
    lat = generate_honeycomb(1, 1, TypeI())
    input = FullED.lattice_to_fulled(lat)

    hamiltonian = FullED.build_hamiltonian(input)
    @test size(hamiltonian) == (4, 4)
    @test hamiltonian == hamiltonian'

    result = FullED.diagonalize(hamiltonian)
    @test length(result.eigenvalues) == 4
    @test all(isfinite, result.eigenvalues)
    @test minimum(result.eigenvalues) < maximum(result.eigenvalues)

    obs = FullED.thermal_observables(result, 2.0, lat.nsites)
    @test obs.nsites == lat.nsites
    @test obs.nstates == 4
    @test isfinite(obs.energy)
    @test isfinite(obs.energy2)
    @test obs.specific_heat >= 0
    @test obs.entropy >= 0

    high_t = FullED.thermal_observables(result, 1.0e-8, lat.nsites)
    @test high_t.entropy ≈ log(4) / lat.nsites atol = 1e-7

    scan = FullED.scan_temperatures(input, [0.5, 1.0, 2.0])
    @test scan.temperatures == [0.5, 1.0, 2.0]
    @test length(scan.observables) == 3
    @test scan.diag_result.eigenvalues ≈ result.eigenvalues
    @test size(scan.diag_result.eigenvectors, 2) == 0

    rows = FullED.comparison_table(scan; metadata=(baseline="PRL113197205",))
    @test length(rows) == 3
    @test rows[1].method == :FullED
    @test rows[1].nsamples == 4
    @test haskey(rows[1].metadata, :baseline)
    @test isfinite(rows[1].free_energy_per_site)
    @test isfinite(rows[1].entropy_per_site)

    @test_throws ArgumentError FullED.scan_temperatures(input, Float64[])
    @test_throws ArgumentError FullED.build_hamiltonian(input; max_states=2)
end

@testset "FullED coupling conventions" begin
    lat = generate_honeycomb(1, 1, TypeI())
    input = FullED.lattice_to_fulled(lat)

    ferro = FullED.diagonalize(input; couplings=(x=0.0, y=0.0, z=1.0), sign=-1.0)
    antiferro = FullED.diagonalize(input; couplings=(x=0.0, y=0.0, z=1.0), sign=1.0)

    @test sort(ferro.eigenvalues) == [-1.0, -1.0, 1.0, 1.0]
    @test sort(antiferro.eigenvalues) == [-1.0, -1.0, 1.0, 1.0]
    @test_throws ArgumentError FullED.build_hamiltonian(input; couplings=(x=1.0, y=1.0))
end
