using LinearAlgebra
using Random

export build_majorana_matrix, matsubara_frequency, majorana_matrix, matsubara_matrix
export logdet_matsubara_dense, log_weight_infinite_product, delta_log_weight_infinite_product
export acceptance_probability_logweight
export refresh_real_pseudofermions, pseudofermion_action, delta_pseudofermion_action
export acceptance_probability_pseudofermion
export log_weight, delta_log_weight, acceptance_probability
export run_pseudofermion_mc, measure_pseudofermion_mc, scan_pseudofermion_temperatures
export pseudofermion_comparison_row, pseudofermion_comparison_table

"""
Finite-cutoff Matsubara infinite-product expansion utilities for Majorana
free-fermion Boltzmann weights.

The core convention is `H = iA`, where `H` is the purely imaginary Hermitian
single-particle Majorana Hamiltonian and `A` is a real antisymmetric matrix.
"""

"""
    build_majorana_matrix(input; couplings=(x=1.0, y=1.0, z=1.0))

Build the real antisymmetric Majorana matrix `A` from a Kitaev bond/gauge
input. The returned matrix follows the excitation-energy scale used in the PRL
Boltzmann factor, so each bond contributes `2Jγuij`.
"""
function build_majorana_matrix(input; couplings=(x=1.0, y=1.0, z=1.0))
    bondset = getproperty(input, :bondset)
    gauge = getproperty(input, :gauge)
    nsites = getproperty(bondset, :nsites)
    bonds = getproperty(bondset, :bonds)
    u = getproperty(gauge, :u)
    length(u) == length(bonds) ||
        throw(ArgumentError("gauge length ($(length(u))) must match number of bonds ($(length(bonds)))"))

    matrix = zeros(Float64, nsites, nsites)
    for bond in bonds
        index = getproperty(bond, :index)
        src = getproperty(bond, :src)
        dst = getproperty(bond, :dst)
        kind = getproperty(bond, :kind)
        amplitude = 2.0 * _kitaev_coupling(couplings, kind) * u[index]
        matrix[src, dst] += amplitude
        matrix[dst, src] -= amplitude
    end
    _validate_majorana_matrix(matrix)
    return matrix
end

"""
    matsubara_frequency(n, beta)

Return the positive fermionic Matsubara frequency
`ω_n = (2n + 1)π / β` for zero-based index `n`.
"""
function matsubara_frequency(n::Integer, beta::Real)
    n >= 0 || throw(ArgumentError("Matsubara index must be non-negative; got $n"))
    _validate_beta(beta)
    return (2 * n + 1) * pi / Float64(beta)
end

"""
    majorana_matrix(hamiltonian; atol=1e-10)

Convert a purely imaginary Hermitian Majorana Hamiltonian `H = iA` into the
real antisymmetric matrix `A`.
"""
function majorana_matrix(hamiltonian::AbstractMatrix; atol::Real=1e-10)
    _validate_square_even_finite(hamiltonian, "Hamiltonian")
    _validate_atol(atol)
    maximum(abs, real.(hamiltonian)) <= atol ||
        throw(ArgumentError("Majorana Hamiltonian must be purely imaginary within tolerance"))
    ishermitian(hamiltonian) ||
        throw(ArgumentError("Majorana Hamiltonian must be Hermitian"))

    matrix = real.(-im .* Matrix(hamiltonian))
    _validate_majorana_matrix(matrix; atol=atol)
    return matrix
end

"""
    matsubara_matrix(A, beta, n; atol=1e-10)

Build `M_n = I - A / ω_n` for a real antisymmetric Majorana matrix `A`.
"""
function matsubara_matrix(matrix::AbstractMatrix{<:Real}, beta::Real, n::Integer; atol::Real=1e-10)
    _validate_majorana_matrix(matrix; atol=atol)
    omega = matsubara_frequency(n, beta)
    return Matrix{Float64}(I, size(matrix, 1), size(matrix, 2)) .- Matrix(matrix) ./ omega
end

"""
    refresh_real_pseudofermions(A, beta; cutoff, rng=Random.default_rng(), atol=1e-10)
    refresh_real_pseudofermions(input, beta; cutoff, rng=Random.default_rng(), couplings=(x=1.0, y=1.0, z=1.0), atol=1e-10)

Sample real pseudofermion fields for the finite Matsubara product. For each
frequency, draw a standard normal vector `ξ_n` and return
`φ_n = M_n' * ξ_n`, where `M_n = I - A / ω_n`.
"""
function refresh_real_pseudofermions(
    matrix::AbstractMatrix,
    beta::Real;
    cutoff::Integer,
    rng::AbstractRNG=Random.default_rng(),
    atol::Real=1e-10,
)
    cutoff > 0 || throw(ArgumentError("cutoff must be positive; got $cutoff"))
    majorana = _as_majorana_matrix(matrix; atol=atol)
    fields = Vector{Vector{Float64}}(undef, cutoff)
    for n in 0:(cutoff - 1)
        xi = randn(rng, size(majorana, 1))
        fields[n + 1] = transpose(matsubara_matrix(majorana, beta, n; atol=atol)) * xi
    end
    return fields
end

function refresh_real_pseudofermions(
    input,
    beta::Real;
    cutoff::Integer,
    rng::AbstractRNG=Random.default_rng(),
    couplings=(x=1.0, y=1.0, z=1.0),
    atol::Real=1e-10,
)
    return refresh_real_pseudofermions(
        build_majorana_matrix(input; couplings=couplings),
        beta;
        cutoff=cutoff,
        rng=rng,
        atol=atol,
    )
end

"""
    logdet_matsubara_dense(A, beta, n; atol=1e-10)

Compute `logdet(I - A / ω_n)` with a dense matrix factorization.
"""
function logdet_matsubara_dense(matrix::AbstractMatrix, beta::Real, n::Integer; atol::Real=1e-10)
    return _real_logdet_matsubara(_as_majorana_matrix(matrix; atol=atol), beta, n; atol=atol)
end

"""
    log_weight(A, beta; cutoff, atol=1e-10)
    log_weight(H, beta; cutoff, atol=1e-10)

Approximate the Majorana Boltzmann log-weight using the finite Matsubara
product

`log W = sum(logdet(I - A / ω_n), n=0:cutoff-1)`.

The overall constant prefactor is omitted.
"""
function log_weight_infinite_product(matrix::AbstractMatrix, beta::Real; cutoff::Integer, atol::Real=1e-10)
    cutoff > 0 || throw(ArgumentError("cutoff must be positive; got $cutoff"))
    _validate_beta(beta)

    majorana = _as_majorana_matrix(matrix; atol=atol)
    logw = 0.0
    for n in 0:(cutoff - 1)
        logw += _real_logdet_matsubara(majorana, beta, n; atol=atol)
    end
    isfinite(logw) || throw(ArgumentError("log weight is not finite"))
    return logw
end

log_weight(matrix::AbstractMatrix, beta::Real; cutoff::Integer, atol::Real=1e-10) =
    log_weight_infinite_product(matrix, beta; cutoff=cutoff, atol=atol)

log_weight_infinite_product(input, beta::Real; cutoff::Integer, couplings=(x=1.0, y=1.0, z=1.0), atol::Real=1e-10) =
    log_weight_infinite_product(build_majorana_matrix(input; couplings=couplings), beta; cutoff=cutoff, atol=atol)

"""
    delta_log_weight_infinite_product(before, after, beta; cutoff, atol=1e-10)

Return `log_weight(after) - log_weight(before)` for two Majorana matrices.
Any omitted constant prefactor cancels for equal matrix sizes.
"""
function delta_log_weight_infinite_product(
    before::AbstractMatrix,
    after::AbstractMatrix,
    beta::Real;
    cutoff::Integer,
    atol::Real=1e-10,
)
    size(before) == size(after) ||
        throw(ArgumentError("matrix sizes must match; got $(size(before)) and $(size(after))"))
    return log_weight_infinite_product(after, beta; cutoff=cutoff, atol=atol) -
           log_weight_infinite_product(before, beta; cutoff=cutoff, atol=atol)
end

delta_log_weight(before::AbstractMatrix, after::AbstractMatrix, beta::Real; cutoff::Integer, atol::Real=1e-10) =
    delta_log_weight_infinite_product(before, after, beta; cutoff=cutoff, atol=atol)

function delta_log_weight_infinite_product(
    before,
    after,
    beta::Real;
    cutoff::Integer,
    couplings=(x=1.0, y=1.0, z=1.0),
    atol::Real=1e-10,
)
    return delta_log_weight_infinite_product(
        build_majorana_matrix(before; couplings=couplings),
        build_majorana_matrix(after; couplings=couplings),
        beta;
        cutoff=cutoff,
        atol=atol,
    )
end

"""
    acceptance_probability_logweight(delta_logw)

Return the Metropolis acceptance probability `min(1, exp(delta_logw))`.
"""
function acceptance_probability_logweight(delta_logw::Real)
    isfinite(delta_logw) || throw(ArgumentError("delta_logw must be finite; got $delta_logw"))
    delta_logw >= 0 && return 1.0
    return exp(delta_logw)
end

acceptance_probability(delta_logw::Real) = acceptance_probability_logweight(delta_logw)

"""
    pseudofermion_action(A, beta, fields; atol=1e-10)
    pseudofermion_action(input, beta, fields; couplings=(x=1.0, y=1.0, z=1.0), atol=1e-10)

Evaluate the finite-cutoff real pseudofermion action
`S_pf = 1/2 * sum(φ_n' * (M_n(A)'M_n(A))^(-1) * φ_n)`.
"""
function pseudofermion_action(matrix::AbstractMatrix, beta::Real, fields; atol::Real=1e-10)
    _validate_beta(beta)
    majorana = _as_majorana_matrix(matrix; atol=atol)
    isempty(fields) && throw(ArgumentError("pseudofermion field collection must be non-empty"))

    action = 0.0
    for (offset, field) in enumerate(fields)
        phi = _validate_pseudofermion_field(field, size(majorana, 1), offset)
        m = matsubara_matrix(majorana, beta, offset - 1; atol=atol)
        y = transpose(m) \ phi
        x = m \ y
        action += 0.5 * dot(phi, x)
    end
    isfinite(action) || throw(ArgumentError("pseudofermion action is not finite"))
    return action
end

function pseudofermion_action(input, beta::Real, fields; couplings=(x=1.0, y=1.0, z=1.0), atol::Real=1e-10)
    return pseudofermion_action(
        build_majorana_matrix(input; couplings=couplings),
        beta,
        fields;
        atol=atol,
    )
end

"""
    delta_pseudofermion_action(before, after, beta, fields; atol=1e-10)

Return `S_pf(after, fields) - S_pf(before, fields)` for fixed real
pseudofermion fields.
"""
function delta_pseudofermion_action(
    before::AbstractMatrix,
    after::AbstractMatrix,
    beta::Real,
    fields;
    atol::Real=1e-10,
)
    size(before) == size(after) ||
        throw(ArgumentError("matrix sizes must match; got $(size(before)) and $(size(after))"))
    return pseudofermion_action(after, beta, fields; atol=atol) -
           pseudofermion_action(before, beta, fields; atol=atol)
end

function delta_pseudofermion_action(
    before,
    after,
    beta::Real,
    fields;
    couplings=(x=1.0, y=1.0, z=1.0),
    atol::Real=1e-10,
)
    return delta_pseudofermion_action(
        build_majorana_matrix(before; couplings=couplings),
        build_majorana_matrix(after; couplings=couplings),
        beta,
        fields;
        atol=atol,
    )
end

"""
    acceptance_probability_pseudofermion(delta_action)

Return the Metropolis acceptance probability `min(1, exp(-delta_action))`
for a fixed-pseudofermion action difference.
"""
function acceptance_probability_pseudofermion(delta_action::Real)
    isfinite(delta_action) || throw(ArgumentError("delta_action must be finite; got $delta_action"))
    delta_action <= 0 && return 1.0
    return exp(-delta_action)
end

"""
    run_pseudofermion_mc(input, beta; cutoff, warmup_sweeps=0, sampling_sweeps, seed=nothing, rng=nothing, couplings=(x=1.0, y=1.0, z=1.0), atol=1e-10)

Run a dense real-pseudofermion Infinite Product Expansion Z2 bond-flip Monte
Carlo simulation. One sweep performs one attempted flip per bond. Each attempt
refreshes pseudofermions from the current gauge configuration, proposes one
random bond flip, and accepts with `min(1, exp(-ΔS_pf))`.

The returned object is a `NamedTuple` with EDMC-like run metadata. Gauge
samples are stored as raw `Vector{Int8}` values so this function remains
lightly coupled to the EDMC submodule.
"""
function run_pseudofermion_mc(
    input,
    beta::Real;
    cutoff::Integer,
    warmup_sweeps::Integer=0,
    sampling_sweeps::Integer,
    seed=nothing,
    rng=nothing,
    couplings=(x=1.0, y=1.0, z=1.0),
    atol::Real=1e-10,
)
    _validate_beta(beta)
    cutoff > 0 || throw(ArgumentError("cutoff must be positive; got $cutoff"))
    warmup_sweeps >= 0 || throw(ArgumentError("warmup_sweeps must be non-negative; got $warmup_sweeps"))
    sampling_sweeps >= 0 || throw(ArgumentError("sampling_sweeps must be non-negative; got $sampling_sweeps"))
    local_rng = _pseudofermion_rng(seed, rng)

    nbonds = length(getproperty(getproperty(input, :bondset), :bonds))
    nbonds > 0 || throw(ArgumentError("pseudofermion MC requires at least one bond"))

    current = input
    warmup_accepted = 0
    sampling_accepted = 0
    samples = Vector{Int8}[]

    for _ in 1:warmup_sweeps
        current, accepted = _run_pseudofermion_sweep(
            current,
            local_rng,
            beta,
            cutoff,
            nbonds;
            couplings=couplings,
            atol=atol,
        )
        warmup_accepted += accepted
    end

    for _ in 1:sampling_sweeps
        current, accepted = _run_pseudofermion_sweep(
            current,
            local_rng,
            beta,
            cutoff,
            nbonds;
            couplings=couplings,
            atol=atol,
        )
        sampling_accepted += accepted
        push!(samples, copy(getproperty(getproperty(current, :gauge), :u)))
    end

    warmup_attempted = warmup_sweeps * nbonds
    sampling_attempted = sampling_sweeps * nbonds
    accepted = warmup_accepted + sampling_accepted
    attempted = warmup_attempted + sampling_attempted
    return (
        method=:PseudofermionIPE,
        final_input=current,
        samples=samples,
        accepted=accepted,
        attempted=attempted,
        warmup_accepted=warmup_accepted,
        warmup_attempted=warmup_attempted,
        sampling_accepted=sampling_accepted,
        sampling_attempted=sampling_attempted,
        acceptance_rate=attempted == 0 ? 0.0 : accepted / attempted,
        sampling_acceptance_rate=sampling_attempted == 0 ? 0.0 : sampling_accepted / sampling_attempted,
        cutoff=cutoff,
    )
end

"""
    measure_pseudofermion_mc(input, beta, run; couplings=(x=1.0, y=1.0, z=1.0), atol=1e-10)

Measure EDMC-compatible observables on gauge samples generated by
[`run_pseudofermion_mc`](@ref). This deliberately reuses the EDMC energy and
specific-heat estimators so dense pseudofermion IPE runs can be compared
against EDMC with the same observable convention.
"""
function measure_pseudofermion_mc(input, beta::Real, run; couplings=(x=1.0, y=1.0, z=1.0), atol::Real=1e-10)
    samples = _z2_gauge_samples(getproperty(run, :samples))
    return EDMC.measure(input, beta; samples=samples, couplings=couplings, atol=atol)
end

"""
    scan_pseudofermion_temperatures(input, temperatures; cutoff, warmup_sweeps=0, sampling_sweeps, seed=nothing, rng=nothing, couplings=(x=1.0, y=1.0, z=1.0), atol=1e-10)

Run dense pseudofermion IPE sequentially over temperatures. Each temperature
starts from the previous temperature's final gauge configuration, matching the
EDMC scan workflow.
"""
function scan_pseudofermion_temperatures(
    input,
    temperatures::AbstractVector{<:Real};
    cutoff::Integer,
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
    local_rng = _pseudofermion_rng(seed, rng)

    current = input
    runs = []
    observables = []

    for temperature in temperatures
        beta = inv(Float64(temperature))
        run = run_pseudofermion_mc(
            current,
            beta;
            cutoff=cutoff,
            warmup_sweeps=warmup_sweeps,
            sampling_sweeps=sampling_sweeps,
            rng=local_rng,
            couplings=couplings,
            atol=atol,
        )
        obs = measure_pseudofermion_mc(current, beta, run; couplings=couplings, atol=atol)
        push!(runs, run)
        push!(observables, obs)
        current = getproperty(run, :final_input)
    end

    return (
        method=:PseudofermionIPE,
        temperatures=Float64.(temperatures),
        observables=observables,
        runs=runs,
        cutoff=cutoff,
    )
end

"""
    pseudofermion_comparison_row(observables; metadata=NamedTuple(), cutoff=nothing)

Return a stable comparison row for pseudofermion IPE observables. The field
layout matches `EDMC.comparison_row`.
"""
function pseudofermion_comparison_row(observables; metadata=NamedTuple(), cutoff=nothing)
    row_metadata = cutoff === nothing ? metadata : merge((cutoff=cutoff,), metadata)
    return EDMC.comparison_row(observables; method=:PseudofermionIPE, metadata=row_metadata)
end

"""
    pseudofermion_comparison_table(scan; metadata=NamedTuple())

Convert a pseudofermion temperature scan to EDMC-compatible comparison rows.
"""
function pseudofermion_comparison_table(scan; metadata=NamedTuple())
    cutoff = getproperty(scan, :cutoff)
    return [pseudofermion_comparison_row(obs; metadata=metadata, cutoff=cutoff) for obs in getproperty(scan, :observables)]
end

function _run_pseudofermion_sweep(
    input,
    rng::AbstractRNG,
    beta::Real,
    cutoff::Integer,
    attempts::Integer;
    couplings,
    atol::Real,
)
    current = input
    accepted = 0
    for _ in 1:attempts
        fields = refresh_real_pseudofermions(current, beta; cutoff=cutoff, rng=rng, couplings=couplings, atol=atol)
        bond = rand(rng, 1:attempts)
        proposed = EDMC.flip_gauge(current, bond)
        delta_action = delta_pseudofermion_action(current, proposed, beta, fields; couplings=couplings, atol=atol)
        if rand(rng) < acceptance_probability_pseudofermion(delta_action)
            current = proposed
            accepted += 1
        end
    end
    return current, accepted
end

function _z2_gauge_samples(samples)
    isempty(samples) && throw(ArgumentError("cannot measure pseudofermion observables without samples"))
    return [EDMC.Z2GaugeField(sample) for sample in samples]
end

function _pseudofermion_rng(seed, rng)
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

function _real_logdet_matsubara(matrix::AbstractMatrix{<:Real}, beta::Real, n::Integer; atol::Real)
    m = matsubara_matrix(matrix, beta, n; atol=atol)
    value, sign = logabsdet(m)
    sign > 0 || throw(ArgumentError("Matsubara determinant must be positive for a real antisymmetric Majorana matrix"))
    return value
end

function _as_majorana_matrix(matrix::AbstractMatrix; atol::Real)
    if eltype(matrix) <: Real
        majorana = Matrix{Float64}(matrix)
        _validate_majorana_matrix(majorana; atol=atol)
        return majorana
    end
    return majorana_matrix(matrix; atol=atol)
end

function _validate_majorana_matrix(matrix::AbstractMatrix{<:Real}; atol::Real=1e-10)
    _validate_square_even_finite(matrix, "Majorana matrix")
    _validate_atol(atol)
    maximum(abs, Matrix(matrix) + transpose(Matrix(matrix))) <= atol ||
        throw(ArgumentError("Majorana matrix must be real antisymmetric within tolerance"))
    return nothing
end

function _validate_square_even_finite(matrix::AbstractMatrix, name::AbstractString)
    size(matrix, 1) == size(matrix, 2) ||
        throw(ArgumentError("$name must be square; got size $(size(matrix))"))
    iseven(size(matrix, 1)) ||
        throw(ArgumentError("$name dimension must be even; got $(size(matrix, 1))"))
    all(isfinite, matrix) ||
        throw(ArgumentError("$name contains NaN or Inf entries"))
    return nothing
end

function _validate_beta(beta::Real)
    isfinite(beta) && beta > 0 ||
        throw(ArgumentError("beta must be a positive finite number; got $beta"))
    return nothing
end

function _validate_atol(atol::Real)
    isfinite(atol) && atol >= 0 ||
        throw(ArgumentError("atol must be a non-negative finite number; got $atol"))
    return nothing
end

function _validate_pseudofermion_field(field, expected_length::Integer, offset::Integer)
    length(field) == expected_length ||
        throw(ArgumentError("pseudofermion field $offset has length $(length(field)); expected $expected_length"))
    all(isfinite, field) ||
        throw(ArgumentError("pseudofermion field $offset contains NaN or Inf entries"))
    return Float64.(field)
end

function _kitaev_coupling(couplings::Real, kind::Symbol)
    isfinite(couplings) || throw(ArgumentError("Kitaev coupling for :$kind must be finite; got $couplings"))
    return Float64(couplings)
end

function _kitaev_coupling(couplings::NamedTuple, kind::Symbol)
    hasproperty(couplings, kind) ||
        throw(ArgumentError("missing Kitaev coupling for bond kind :$kind"))
    return _kitaev_coupling(getproperty(couplings, kind), kind)
end

function _kitaev_coupling(couplings::AbstractDict, kind::Symbol)
    haskey(couplings, kind) ||
        throw(ArgumentError("missing Kitaev coupling for bond kind :$kind"))
    return _kitaev_coupling(couplings[kind], kind)
end
