"""
    flip_gauge(input, bond_index)

Return a new [`KitaevHamiltonianInput`](@ref) with one `u_ij` variable flipped.
"""
function flip_gauge(input::KitaevHamiltonianInput, bond_index::Integer)
    1 <= bond_index <= length(input.gauge.u) ||
        throw(ArgumentError("bond_index must be in 1:$(length(input.gauge.u)); got $bond_index"))

    flipped = copy(input.gauge.u)
    flipped[bond_index] = -flipped[bond_index]
    return KitaevHamiltonianInput(input.bondset, Z2GaugeField(flipped))
end

"""
    delta_free_energy(before, after, beta; couplings=(x=1.0, y=1.0, z=1.0))
    delta_free_energy(input, bond_index, beta; couplings=(x=1.0, y=1.0, z=1.0))

Compute `F_after - F_before` for an EDMC gauge update.
"""
function delta_free_energy(
    before::KitaevHamiltonianInput,
    after::KitaevHamiltonianInput,
    beta::Real;
    couplings=(x=1.0, y=1.0, z=1.0),
    atol::Real=1e-10,
)
    _same_bondset(before.bondset, after.bondset)
    value = free_energy(after, beta; couplings=couplings, atol=atol) -
            free_energy(before, beta; couplings=couplings, atol=atol)
    isfinite(value) || throw(ArgumentError("delta free energy is not finite"))
    return value
end

function delta_free_energy(input::KitaevHamiltonianInput, bond_index::Integer, beta::Real; kwargs...)
    return delta_free_energy(input, flip_gauge(input, bond_index), beta; kwargs...)
end

"""
    acceptance_probability(delta_f, beta)

Return the Metropolis acceptance probability `min(1, exp(-beta * delta_f))`.
"""
function acceptance_probability(delta_f::Real, beta::Real)
    isfinite(delta_f) || throw(ArgumentError("delta_f must be finite; got $delta_f"))
    _validate_beta(beta)
    exponent = -beta * delta_f
    if exponent >= 0
        return 1.0
    elseif exponent == -Inf
        return 0.0
    elseif !isfinite(exponent)
        throw(ArgumentError("acceptance exponent is not finite; beta=$beta, delta_f=$delta_f"))
    else
        return exp(exponent)
    end
end

"""
    propose_bond_flip(input, rng, beta; couplings=(x=1.0, y=1.0, z=1.0))

Choose one Z2 gauge bond uniformly and compute its Metropolis proposal data.
"""
function propose_bond_flip(
    input::KitaevHamiltonianInput,
    rng::AbstractRNG,
    beta::Real;
    couplings=(x=1.0, y=1.0, z=1.0),
    atol::Real=1e-10,
)
    nbonds = length(input.gauge.u)
    nbonds > 0 || throw(ArgumentError("cannot propose a Z2 bond flip with zero bonds"))
    bond_index = rand(rng, 1:nbonds)
    delta_f = delta_free_energy(input, bond_index, beta; couplings=couplings, atol=atol)
    probability = acceptance_probability(delta_f, beta)
    return Z2BondFlipProposal(bond_index, delta_f, probability)
end

"""
    attempt_bond_flip(input, rng, beta; couplings=(x=1.0, y=1.0, z=1.0))

Attempt a single detailed-balance-preserving Z2 bond flip using the Metropolis
acceptance rule. Returns a [`Z2BondFlipResult`](@ref).
"""
function attempt_bond_flip(
    input::KitaevHamiltonianInput,
    rng::AbstractRNG,
    beta::Real;
    couplings=(x=1.0, y=1.0, z=1.0),
    atol::Real=1e-10,
)
    proposal = propose_bond_flip(input, rng, beta; couplings=couplings, atol=atol)
    accepted = rand(rng) < proposal.probability
    next_input = accepted ? flip_gauge(input, proposal.bond_index) : input
    return Z2BondFlipResult(next_input, proposal, accepted)
end

function _same_bondset(a::KitaevBondSet, b::KitaevBondSet)
    a.lattice_kind == b.lattice_kind ||
        throw(ArgumentError("bondsets have different lattice kinds: $(a.lattice_kind) and $(b.lattice_kind)"))
    a.dims == b.dims ||
        throw(ArgumentError("bondsets have different dimensions: $(a.dims) and $(b.dims)"))
    a.nsites == b.nsites ||
        throw(ArgumentError("bondsets have different site counts: $(a.nsites) and $(b.nsites)"))
    length(a.bonds) == length(b.bonds) ||
        throw(ArgumentError("bondsets have different bond counts: $(length(a.bonds)) and $(length(b.bonds))"))

    for (i, (ba, bb)) in enumerate(zip(a.bonds, b.bonds))
        (ba.src, ba.dst, ba.kind, ba.wrapped) == (bb.src, bb.dst, bb.kind, bb.wrapped) ||
            throw(ArgumentError("bondsets differ at bond index $i"))
    end
    return nothing
end

"""
    propose_update(state, rng)

Propose a Monte Carlo update for an EDMC state.

The update rule and proposal object are intentionally unspecified at this
stage.
"""
function propose_update(state::EDMCState, rng)
    throw(ErrorException("EDMC update proposals are not implemented yet"))
end

"""
    accept_update(state, proposal)

Apply an accepted EDMC proposal and return the updated state.
"""
function accept_update(state::EDMCState, proposal)
    throw(ErrorException("EDMC update acceptance is not implemented yet"))
end
