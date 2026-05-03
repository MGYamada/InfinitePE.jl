"""
    partition_function(result, beta)

Compute the exact partition function from a full spectrum.
"""
function partition_function(result::FullEDDiagonalizationResult, beta::Real)
    _validate_beta(beta)
    shifted_weights, shift = _boltzmann_weights(result.eigenvalues, beta)
    return exp(-beta * shift) * sum(shifted_weights)
end

"""
    free_energy(result, beta)

Exact Helmholtz free energy from a full spin spectrum.
"""
function free_energy(result::FullEDDiagonalizationResult, beta::Real)
    _validate_beta(beta)
    weights, shift = _boltzmann_weights(result.eigenvalues, beta)
    logz = -beta * shift + log(sum(weights))
    return -logz / beta
end

"""
    thermal_observables(result, beta, nsites)

Compute exact thermal observables per site from a full spin spectrum.
"""
function thermal_observables(result::FullEDDiagonalizationResult, beta::Real, nsites::Integer)
    _validate_beta(beta)
    nsites > 0 || throw(ArgumentError("nsites must be positive; got $nsites"))

    weights, shift = _boltzmann_weights(result.eigenvalues, beta)
    weight_sum = sum(weights)
    weight_sum > 0 && isfinite(weight_sum) ||
        throw(ArgumentError("Boltzmann weight sum is not finite and positive"))

    probabilities = weights ./ weight_sum
    total_energy = sum(probabilities .* result.eigenvalues)
    total_energy2 = sum(probabilities .* abs2.(result.eigenvalues))
    variance = max(0.0, total_energy2 - total_energy^2)
    logz = -beta * shift + log(weight_sum)
    total_free_energy = -logz / beta
    total_entropy = logz + beta * total_energy

    return FullEDObservables(
        beta,
        inv(beta),
        total_energy / nsites,
        total_energy2 / nsites^2,
        beta^2 * variance / nsites,
        total_free_energy / nsites,
        total_entropy / nsites,
        length(result.eigenvalues),
        nsites,
    )
end

function thermal_observables(input::FullEDHamiltonianInput, beta::Real; kwargs...)
    result = diagonalize(input; kwargs...)
    return thermal_observables(result, beta, input.bondset.nsites)
end

"""
    comparison_row(observables; method=:FullED, metadata=NamedTuple())

Return a stable row for comparing Full ED with EDMC or Infinite PE outputs.
"""
function comparison_row(observables::FullEDObservables; method::Symbol=:FullED, metadata=NamedTuple())
    return (
        method=method,
        temperature=observables.temperature,
        beta=observables.beta,
        energy_per_site=observables.energy,
        energy2_per_site2=observables.energy2,
        energy_beta_derivative_per_site=0.0,
        specific_heat_per_site=observables.specific_heat,
        free_energy_per_site=observables.free_energy,
        entropy_per_site=observables.entropy,
        nsamples=observables.nstates,
        nsites=observables.nsites,
        metadata=metadata,
    )
end

"""
    comparison_table(scan; method=:FullED, metadata=NamedTuple())

Convert a Full ED temperature scan to comparison rows.
"""
function comparison_table(scan::FullEDTemperatureScanResult; method::Symbol=:FullED, metadata=NamedTuple())
    return [comparison_row(obs; method=method, metadata=metadata) for obs in scan.observables]
end

function _boltzmann_weights(eigenvalues::AbstractVector{<:Real}, beta::Real)
    isempty(eigenvalues) && throw(ArgumentError("cannot compute thermodynamics for an empty spectrum"))
    shift = minimum(eigenvalues)
    weights = exp.(-beta .* (eigenvalues .- shift))
    all(isfinite, weights) || throw(ArgumentError("Boltzmann weights contain NaN or Inf"))
    return weights, shift
end

function _validate_beta(beta::Real)
    isfinite(beta) && beta > 0 || throw(ArgumentError("beta must be a positive finite number; got $beta"))
    return nothing
end
