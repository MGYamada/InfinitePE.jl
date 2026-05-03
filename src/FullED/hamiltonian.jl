"""
    lattice_to_fulled(lattice)

Convert an `InfinitePE` lattice to a full spin exact-diagonalization input.
"""
function lattice_to_fulled(lattice)
    return FullEDHamiltonianInput(extract_kitaev_bonds(lattice))
end

"""
    build_hamiltonian(input; couplings=(x=1.0, y=1.0, z=1.0), sign=-1.0, max_states=65536)

Build the dense spin-1/2 Kitaev Hamiltonian in the Pauli convention
`H = sign * sum(Jγ * σᵢγ * σⱼγ)`.

The default `sign=-1` corresponds to ferromagnetic Kitaev couplings for
positive `Jγ`. `max_states` guards against accidental dense matrices that are
too large for a comparison baseline.
"""
function build_hamiltonian(
    input::FullEDHamiltonianInput;
    couplings=(x=1.0, y=1.0, z=1.0),
    sign::Real=-1.0,
    max_states::Integer=65536,
)
    isfinite(sign) || throw(ArgumentError("sign must be finite; got $sign"))
    nsites = input.bondset.nsites
    nstates = _hilbert_size(nsites)
    nstates <= max_states ||
        throw(ArgumentError("Full ED requires a $nstates x $nstates dense Hamiltonian; increase max_states explicitly"))

    matrix = zeros(ComplexF64, nstates, nstates)
    for bond in input.bondset.bonds
        coefficient = Float64(sign) * _kitaev_coupling(couplings, bond.kind)
        _add_two_spin_term!(matrix, nstates, bond.src, bond.dst, bond.kind, coefficient)
    end

    return Hermitian(matrix)
end

function _add_two_spin_term!(
    matrix::AbstractMatrix{ComplexF64},
    nstates::Int,
    site_i::Int,
    site_j::Int,
    kind::Symbol,
    coefficient::Float64,
)
    mask_i = UInt(1) << (site_i - 1)
    mask_j = UInt(1) << (site_j - 1)
    flip_mask = mask_i | mask_j

    for state0 in UInt(0):(UInt(nstates) - UInt(1))
        target, phase = _two_spin_action(state0, mask_i, mask_j, flip_mask, kind)
        row = Int(target) + 1
        col = Int(state0) + 1
        matrix[row, col] += coefficient * phase
    end

    return matrix
end

function _two_spin_action(state::UInt, mask_i::UInt, mask_j::UInt, flip_mask::UInt, kind::Symbol)
    if kind === :x
        return state ⊻ flip_mask, 1.0 + 0.0im
    elseif kind === :y
        bit_i = (state & mask_i) == 0 ? 0 : 1
        bit_j = (state & mask_j) == 0 ? 0 : 1
        phase_i = bit_i == 0 ? 1.0im : -1.0im
        phase_j = bit_j == 0 ? 1.0im : -1.0im
        return state ⊻ flip_mask, phase_i * phase_j
    elseif kind === :z
        spin_i = (state & mask_i) == 0 ? 1.0 : -1.0
        spin_j = (state & mask_j) == 0 ? 1.0 : -1.0
        return state, complex(spin_i * spin_j)
    end
    throw(ArgumentError("unsupported Kitaev bond kind :$kind"))
end

function _hilbert_size(nsites::Integer)
    nsites >= 0 || throw(ArgumentError("number of sites must be non-negative; got $nsites"))
    nsites < Sys.WORD_SIZE ||
        throw(ArgumentError("cannot form 2^$nsites basis states on a $(Sys.WORD_SIZE)-bit system"))
    return 1 << nsites
end

function _kitaev_coupling(couplings::Real, kind::Symbol)
    isfinite(couplings) || throw(ArgumentError("Kitaev coupling for :$kind must be finite; got $couplings"))
    return Float64(couplings)
end

function _kitaev_coupling(couplings::NamedTuple, kind::Symbol)
    hasproperty(couplings, kind) || throw(ArgumentError("missing Kitaev coupling for bond kind :$kind"))
    return _kitaev_coupling(getproperty(couplings, kind), kind)
end

function _kitaev_coupling(couplings::AbstractDict, kind::Symbol)
    haskey(couplings, kind) || throw(ArgumentError("missing Kitaev coupling for bond kind :$kind"))
    return _kitaev_coupling(couplings[kind], kind)
end
