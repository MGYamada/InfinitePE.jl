"""
    internal_energy(x, beta; couplings=(x=1.0, y=1.0, z=1.0))

Compute the Majorana internal-energy estimator for one gauge configuration.

The convention is total energy `E = -1/2 * sum(ε * tanh(β ε / 2))` over the
non-negative Majorana single-particle energies `ε`.
"""
function internal_energy(input::KitaevHamiltonianInput, beta::Real; couplings=(x=1.0, y=1.0, z=1.0), atol::Real=1e-10)
    return internal_energy(diagonalize(build_hamiltonian(input; couplings=couplings)), beta; atol=atol)
end

function internal_energy(hamiltonian::AbstractMatrix, beta::Real; atol::Real=1e-10)
    return internal_energy(diagonalize(hamiltonian), beta; atol=atol)
end

function internal_energy(result::DiagonalizationResult, beta::Real; atol::Real=1e-10)
    _validate_beta(beta)
    value = sum(_internal_energy_term(energy, beta) for energy in majorana_energies(result; atol=atol))
    isfinite(value) || throw(ArgumentError("internal energy is not finite"))
    return value
end

"""
    internal_energy_beta_derivative(x, beta; couplings=(x=1.0, y=1.0, z=1.0))

Compute `∂E/∂β` for the Majorana internal-energy estimator.

For `E = -1/2 * sum(ε * tanh(β ε / 2))`, this is
`∂E/∂β = -1/4 * sum(ε^2 * sech(β ε / 2)^2)`.
"""
function internal_energy_beta_derivative(input::KitaevHamiltonianInput, beta::Real; couplings=(x=1.0, y=1.0, z=1.0), atol::Real=1e-10)
    return internal_energy_beta_derivative(diagonalize(build_hamiltonian(input; couplings=couplings)), beta; atol=atol)
end

function internal_energy_beta_derivative(hamiltonian::AbstractMatrix, beta::Real; atol::Real=1e-10)
    return internal_energy_beta_derivative(diagonalize(hamiltonian), beta; atol=atol)
end

function internal_energy_beta_derivative(result::DiagonalizationResult, beta::Real; atol::Real=1e-10)
    _validate_beta(beta)
    value = sum(_internal_energy_beta_derivative_term(energy, beta) for energy in majorana_energies(result; atol=atol))
    isfinite(value) || throw(ArgumentError("internal energy beta derivative is not finite"))
    return value
end

"""
    measure(input, beta; samples=[input.gauge], couplings=(x=1.0, y=1.0, z=1.0))

Aggregate EDMC observables for gauge samples.

Units/convention: `energy` is the sample average of the Majorana internal
energy per site. `energy2` is the sample average of `(E_total / N)^2`.
`energy_beta_derivative` is `⟨∂E_total/∂β⟩ / N`. `specific_heat` is reported
per site as `(β^2 / N) * (⟨E_total^2⟩ - ⟨E_total⟩^2 - ⟨∂E_total/∂β⟩)`.
"""
function measure(
    input::KitaevHamiltonianInput,
    beta::Real;
    samples::AbstractVector{Z2GaugeField}=[input.gauge],
    couplings=(x=1.0, y=1.0, z=1.0),
    atol::Real=1e-10,
)
    _validate_beta(beta)
    isempty(samples) && throw(ArgumentError("cannot measure EDMC observables without samples"))

    nsites = input.bondset.nsites
    energies = Float64[]
    derivatives = Float64[]
    sizehint!(energies, length(samples))
    sizehint!(derivatives, length(samples))

    for sample in samples
        sample_input = KitaevHamiltonianInput(input.bondset, sample)
        total_energy = internal_energy(sample_input, beta; couplings=couplings, atol=atol)
        total_derivative = internal_energy_beta_derivative(sample_input, beta; couplings=couplings, atol=atol)
        energy_per_site = total_energy / nsites
        derivative_per_site = total_derivative / nsites
        isfinite(energy_per_site) || throw(ArgumentError("sample energy is not finite"))
        isfinite(derivative_per_site) || throw(ArgumentError("sample energy beta derivative is not finite"))
        push!(energies, energy_per_site)
        push!(derivatives, derivative_per_site)
    end

    energy = sum(energies) / length(energies)
    energy2 = sum(abs2, energies) / length(energies)
    energy_beta_derivative = sum(derivatives) / length(derivatives)
    specific_heat = beta^2 * (nsites * (energy2 - energy^2) - energy_beta_derivative)
    specific_heat = max(0.0, specific_heat)
    return EDMCObservables(beta, inv(beta), energy, energy2, energy_beta_derivative, specific_heat, length(energies), nsites)
end

"""
    comparison_row(observables; method=:EDMC, metadata=NamedTuple())

Return a stable `NamedTuple` row for comparing EDMC with Infinite PE outputs.

Field names are intentionally method-agnostic:
`method`, `temperature`, `beta`, `energy_per_site`, `energy2_per_site2`,
`energy_beta_derivative_per_site`, `specific_heat_per_site`, `nsamples`,
`nsites`, and `metadata`.
"""
function comparison_row(observables::EDMCObservables; method::Symbol=:EDMC, metadata=NamedTuple())
    return (
        method=method,
        temperature=observables.temperature,
        beta=observables.beta,
        energy_per_site=observables.energy,
        energy2_per_site2=observables.energy2,
        energy_beta_derivative_per_site=observables.energy_beta_derivative,
        specific_heat_per_site=observables.specific_heat,
        nsamples=observables.nsamples,
        nsites=observables.nsites,
        metadata=metadata,
    )
end

"""
    comparison_table(scan; method=:EDMC, metadata=NamedTuple())

Convert a temperature scan to comparison rows with stable field names.
"""
function comparison_table(scan::EDMCTemperatureScanResult; method::Symbol=:EDMC, metadata=NamedTuple())
    return [comparison_row(obs; method=method, metadata=metadata) for obs in scan.observables]
end

function _internal_energy_term(energy::Real, beta::Real)
    energy >= 0 || throw(ArgumentError("Majorana energy must be non-negative; got $energy"))
    x = 0.5 * beta * energy
    return -0.5 * energy * tanh(x)
end

function _internal_energy_beta_derivative_term(energy::Real, beta::Real)
    energy >= 0 || throw(ArgumentError("Majorana energy must be non-negative; got $energy"))
    x = 0.5 * beta * energy
    sech2 = if x > 350
        0.0
    else
        inv(cosh(x))^2
    end
    return -0.25 * energy^2 * sech2
end

"""
    measure(state, diag_result)

Measure observables for an EDMC state and diagonalization result.

The observable set and return type will be defined with the physical model.
"""
function measure(state::EDMCState, diag_result)
    throw(ErrorException("EDMC measurements are not implemented yet"))
end
