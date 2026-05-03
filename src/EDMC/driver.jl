"""
    run_edmc(input, beta; warmup_sweeps=0, sampling_sweeps, seed=nothing)

Run the minimal Z2-bond-flip Metropolis driver.

One sweep performs `length(input.bondset.bonds)` single-bond flip attempts.
Samples are stored after each sampling sweep.
"""
function run_edmc(
    input::KitaevHamiltonianInput,
    beta::Real;
    warmup_sweeps::Integer=0,
    sampling_sweeps::Integer,
    seed=nothing,
    rng=nothing,
    couplings=(x=1.0, y=1.0, z=1.0),
    atol::Real=1e-10,
)
    _validate_driver_counts(warmup_sweeps, sampling_sweeps)
    local_rng = _edmc_rng(seed, rng)
    attempts_per_sweep = length(input.bondset.bonds)
    attempts_per_sweep > 0 || throw(ArgumentError("EDMC driver requires at least one bond"))

    current = input
    warmup_accepted = 0
    warmup_attempted = 0
    sampling_accepted = 0
    sampling_attempted = 0
    samples = Z2GaugeField[]

    for _ in 1:warmup_sweeps
        current, accepted = _run_bond_flip_sweep(current, local_rng, beta, attempts_per_sweep; couplings=couplings, atol=atol)
        warmup_accepted += accepted
        warmup_attempted += attempts_per_sweep
    end

    for _ in 1:sampling_sweeps
        current, accepted = _run_bond_flip_sweep(current, local_rng, beta, attempts_per_sweep; couplings=couplings, atol=atol)
        sampling_accepted += accepted
        sampling_attempted += attempts_per_sweep
        push!(samples, Z2GaugeField(copy(current.gauge.u)))
    end

    accepted = warmup_accepted + sampling_accepted
    attempted = warmup_attempted + sampling_attempted
    return EDMCDriverResult(
        current,
        samples,
        accepted,
        attempted,
        warmup_accepted,
        warmup_attempted,
        sampling_accepted,
        sampling_attempted,
    )
end

"""
    scan_temperatures(input, temperatures; warmup_sweeps=0, sampling_sweeps, seed=nothing)

Run EDMC sequentially over `temperatures` and measure energy and specific heat.

Temperatures are in units where `k_B = 1`; internally `β = 1/T`. Each
temperature starts from the previous temperature's final Z2 gauge field, which
is convenient for scans. Results are returned as an
[`EDMCTemperatureScanResult`](@ref) containing per-temperature run metadata and
[`EDMCObservables`](@ref).
"""
function scan_temperatures(
    input::KitaevHamiltonianInput,
    temperatures::AbstractVector{<:Real};
    warmup_sweeps::Integer=0,
    sampling_sweeps::Integer,
    seed=nothing,
    rng=nothing,
    couplings=(x=1.0, y=1.0, z=1.0),
    atol::Real=1e-10,
)
    isempty(temperatures) && throw(ArgumentError("temperature scan requires at least one temperature"))
    all(t -> isfinite(t) && t > 0, temperatures) ||
        throw(ArgumentError("all scan temperatures must be positive and finite"))
    local_rng = _edmc_rng(seed, rng)

    current = input
    runs = EDMCDriverResult[]
    observables = EDMCObservables[]

    for temperature in temperatures
        beta = inv(Float64(temperature))
        run = run_edmc(
            current,
            beta;
            warmup_sweeps=warmup_sweeps,
            sampling_sweeps=sampling_sweeps,
            rng=local_rng,
            couplings=couplings,
            atol=atol,
        )
        obs = measure(current, beta; samples=run.samples, couplings=couplings, atol=atol)
        push!(runs, run)
        push!(observables, obs)
        current = run.final_input
    end

    return EDMCTemperatureScanResult(temperatures, observables, runs)
end

function _run_bond_flip_sweep(
    input::KitaevHamiltonianInput,
    rng::AbstractRNG,
    beta::Real,
    attempts::Integer;
    couplings=(x=1.0, y=1.0, z=1.0),
    atol::Real=1e-10,
)
    current = input
    accepted = 0
    for _ in 1:attempts
        result = attempt_bond_flip(current, rng, beta; couplings=couplings, atol=atol)
        current = result.input
        accepted += result.accepted ? 1 : 0
    end
    return current, accepted
end

function _edmc_rng(seed, rng)
    if rng !== nothing && seed !== nothing
        throw(ArgumentError("pass either rng or seed, not both"))
    elseif rng !== nothing
        rng isa AbstractRNG || throw(ArgumentError("rng must be an AbstractRNG; got $(typeof(rng))"))
        return rng
    elseif seed !== nothing
        return MersenneTwister(seed)
    else
        return Random.default_rng()
    end
end

function _validate_driver_counts(warmup_sweeps::Integer, sampling_sweeps::Integer)
    warmup_sweeps >= 0 || throw(ArgumentError("warmup_sweeps must be non-negative; got $warmup_sweeps"))
    sampling_sweeps >= 0 || throw(ArgumentError("sampling_sweeps must be non-negative; got $sampling_sweeps"))
    return nothing
end

"""
    run_edmc(model, initial_state, params, config; rng=nothing)

Run an EDMC simulation.

This driver is a placeholder for the future orchestration of Hamiltonian
construction, diagonalization, Monte Carlo updates, and measurements.
"""
function run_edmc(
    model::AbstractEDMCModel,
    initial_state::EDMCState,
    params::EDMCParameters,
    config::EDMCRunConfig;
    rng=nothing,
)
    throw(ErrorException("EDMC driver is not implemented yet"))
end
