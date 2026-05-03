"""
    build_hamiltonian(input; couplings=(x=1.0, y=1.0, z=1.0))

Build the single-particle Hamiltonian from a [`KitaevHamiltonianInput`](@ref).

The returned matrix is Hermitian and follows the minimal Majorana hopping
convention `H[i,j] = im * Jγ * u_ij` and `H[j,i] = -im * Jγ * u_ij`.
`couplings` may be a scalar, a `NamedTuple`, or a dictionary keyed by `:x`,
`:y`, and `:z`.
"""
function build_hamiltonian(input::KitaevHamiltonianInput; couplings=(x=1.0, y=1.0, z=1.0))
    nsites = input.bondset.nsites
    matrix = zeros(ComplexF64, nsites, nsites)

    for bond in input.bondset.bonds
        uij = input.gauge.u[bond.index]
        coupling = _kitaev_coupling(couplings, bond.kind)
        amplitude = coupling * uij
        matrix[bond.src, bond.dst] += im * amplitude
        matrix[bond.dst, bond.src] -= im * amplitude
    end

    return matrix
end

"""
    free_energy(x, beta; couplings=(x=1.0, y=1.0, z=1.0), atol=1e-10)

Compute the fermionic free-energy contribution used by EDMC acceptance tests.

For a Majorana spectrum this evaluates
`-(1/beta) * sum(log(2cosh(beta * epsilon / 2)))` over the non-negative
single-particle energies. The `log(2cosh)` expression is evaluated in a stable
form to avoid overflow at low temperature.
"""
function free_energy(input::KitaevHamiltonianInput, beta::Real; couplings=(x=1.0, y=1.0, z=1.0), atol::Real=1e-10)
    return free_energy(build_hamiltonian(input; couplings=couplings), beta; atol=atol)
end

function free_energy(hamiltonian::AbstractMatrix, beta::Real; atol::Real=1e-10)
    return free_energy(diagonalize(hamiltonian), beta; atol=atol)
end

function free_energy(result::DiagonalizationResult, beta::Real; atol::Real=1e-10)
    _validate_beta(beta)
    energies = majorana_energies(result; atol=atol)
    value = sum(_free_energy_term(energy, beta) for energy in energies)
    isfinite(value) ||
        throw(ArgumentError("free energy is not finite; beta=$beta may be too close to zero for this spectrum"))
    return value
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

function _validate_beta(beta::Real)
    isfinite(beta) && beta > 0 ||
        throw(ArgumentError("beta must be a positive finite number; got $beta"))
    return nothing
end

function _log2cosh(x::Real)
    ax = abs(x)
    return ax + log1p(exp(-2 * ax))
end

function _free_energy_term(energy::Real, beta::Real)
    energy >= 0 || throw(ArgumentError("Majorana energy must be non-negative; got $energy"))
    x = 0.5 * beta * energy
    if isfinite(x)
        return -_log2cosh(x) / beta
    end
    return -0.5 * energy
end

"""
    build_hamiltonian(model, state, params)

Build the single-particle Hamiltonian for an EDMC configuration.

No Hamiltonian construction is implemented yet. Future implementations should
return an object accepted by [`diagonalize`](@ref).
"""
function build_hamiltonian(model::AbstractEDMCModel, state::EDMCState, params::EDMCParameters)
    throw(ErrorException("EDMC Hamiltonian construction is not implemented yet"))
end
