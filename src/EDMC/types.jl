"""
    AbstractEDMCModel

Abstract supertype for EDMC model descriptions.

Concrete subtypes should carry the model-specific data needed to build an
EDMC Hamiltonian from a lattice and field configuration.
"""
abstract type AbstractEDMCModel end

"""
    EDMCParameters(; temperature, coupling, chemical_potential=0.0)

Minimal parameter container for future EDMC calculations.

The fields are deliberately generic while the EDMC implementation is only a
scaffold.
"""
Base.@kwdef struct EDMCParameters
    temperature::Float64
    coupling::Float64
    chemical_potential::Float64 = 0.0
end

"""
    EDMCState{L,F}

Container for the lattice and classical field configuration used by an EDMC
run.
"""
Base.@kwdef struct EDMCState{L,F}
    lattice::L
    field::F
end

"""
    EDMCRunConfig(; sweeps, thermalization=0, measure_interval=1)

Execution settings for an EDMC driver.
"""
Base.@kwdef struct EDMCRunConfig
    sweeps::Int
    thermalization::Int = 0
    measure_interval::Int = 1
end

const KITAEV_BOND_KINDS = (:x, :y, :z)

"""
    KitaevBond

Stable EDMC representation of one Kitaev `x`, `y`, or `z` bond.

`index` is the position of this bond in the associated [`KitaevBondSet`](@ref)
and is therefore also the index used by the corresponding `u_ij` gauge
variable.
"""
struct KitaevBond
    index::Int
    src::Int
    dst::Int
    kind::Symbol
    wrapped::Bool
    src_coord::Any
    dst_coord::Any

    function KitaevBond(index::Int, src::Int, dst::Int, kind::Symbol, wrapped::Bool, src_coord, dst_coord)
        index > 0 || throw(ArgumentError("KitaevBond index must be positive; got $index"))
        src > 0 || throw(ArgumentError("KitaevBond src site index must be positive; got $src"))
        dst > 0 || throw(ArgumentError("KitaevBond dst site index must be positive; got $dst"))
        kind in KITAEV_BOND_KINDS ||
            throw(ArgumentError("KitaevBond kind must be one of $(KITAEV_BOND_KINDS); got :$kind"))
        return new(index, src, dst, kind, wrapped, src_coord, dst_coord)
    end
end

"""
    KitaevBondSet

Deterministically ordered collection of Kitaev bonds extracted from a lattice.

The order of `bonds` defines the stable mapping from `bond.index` to the
corresponding entry in a [`Z2GaugeField`](@ref).
"""
struct KitaevBondSet
    lattice_kind::Symbol
    dims::NTuple{3,Int}
    nsites::Int
    bonds::Vector{KitaevBond}
end

"""
    Z2GaugeField(values)

Gauge variables `u_ij = ±1` aligned with a [`KitaevBondSet`](@ref).
"""
struct Z2GaugeField
    u::Vector{Int8}

    function Z2GaugeField(values::AbstractVector{<:Integer})
        all(v -> v == 1 || v == -1, values) ||
            throw(ArgumentError("Z2 gauge values must all be ±1"))
        return new(Int8.(values))
    end
end

"""
    KitaevHamiltonianInput(bondset, gauge)

Unified, model-independent input format for future Kitaev EDMC Hamiltonian
construction.
"""
struct KitaevHamiltonianInput
    bondset::KitaevBondSet
    gauge::Z2GaugeField

    function KitaevHamiltonianInput(bondset::KitaevBondSet, gauge::Z2GaugeField)
        length(gauge.u) == length(bondset.bonds) ||
            throw(ArgumentError(
                "Z2 gauge length ($(length(gauge.u))) must match number of Kitaev bonds ($(length(bondset.bonds)))",
            ))
        return new(bondset, gauge)
    end
end

"""
    DiagonalizationResult(eigenvalues, eigenvectors)

Result of the EDMC exact diagonalization backend.
"""
struct DiagonalizationResult{T<:Real,V<:AbstractMatrix}
    eigenvalues::Vector{T}
    eigenvectors::V

    function DiagonalizationResult(eigenvalues::AbstractVector{<:Real}, eigenvectors::AbstractMatrix)
        length(eigenvalues) == size(eigenvectors, 1) ||
            throw(ArgumentError(
                "number of eigenvalues ($(length(eigenvalues))) must match eigenvector row count ($(size(eigenvectors, 1)))",
            ))
        all(isfinite, eigenvalues) ||
            throw(ArgumentError("diagonalization produced non-finite eigenvalues"))
        return new{eltype(collect(eigenvalues)),typeof(eigenvectors)}(collect(eigenvalues), eigenvectors)
    end
end

"""
    Z2BondFlipProposal

Proposal data for a single Z2 gauge-bond flip.
"""
struct Z2BondFlipProposal
    bond_index::Int
    delta_f::Float64
    probability::Float64

    function Z2BondFlipProposal(bond_index::Integer, delta_f::Real, probability::Real)
        bond_index > 0 || throw(ArgumentError("bond_index must be positive; got $bond_index"))
        isfinite(delta_f) || throw(ArgumentError("delta_f must be finite; got $delta_f"))
        isfinite(probability) && 0 <= probability <= 1 ||
            throw(ArgumentError("probability must be finite and in [0, 1]; got $probability"))
        return new(Int(bond_index), Float64(delta_f), Float64(probability))
    end
end

"""
    Z2BondFlipResult

Result of a single Metropolis Z2 bond-flip attempt.
"""
struct Z2BondFlipResult
    input::KitaevHamiltonianInput
    proposal::Z2BondFlipProposal
    accepted::Bool
end

"""
    EDMCDriverResult

Result returned by the minimal EDMC Metropolis driver.
"""
struct EDMCDriverResult
    final_input::KitaevHamiltonianInput
    samples::Vector{Z2GaugeField}
    accepted::Int
    attempted::Int
    warmup_accepted::Int
    warmup_attempted::Int
    sampling_accepted::Int
    sampling_attempted::Int

    function EDMCDriverResult(
        final_input::KitaevHamiltonianInput,
        samples::Vector{Z2GaugeField},
        accepted::Integer,
        attempted::Integer,
        warmup_accepted::Integer,
        warmup_attempted::Integer,
        sampling_accepted::Integer,
        sampling_attempted::Integer,
    )
        0 <= accepted <= attempted ||
            throw(ArgumentError("accepted count must be in 0:attempted; got $accepted / $attempted"))
        0 <= warmup_accepted <= warmup_attempted ||
            throw(ArgumentError("warmup accepted count must be in 0:warmup_attempted"))
        0 <= sampling_accepted <= sampling_attempted ||
            throw(ArgumentError("sampling accepted count must be in 0:sampling_attempted"))
        accepted == warmup_accepted + sampling_accepted ||
            throw(ArgumentError("accepted count must equal warmup plus sampling accepted counts"))
        attempted == warmup_attempted + sampling_attempted ||
            throw(ArgumentError("attempted count must equal warmup plus sampling attempted counts"))
        return new(
            final_input,
            samples,
            Int(accepted),
            Int(attempted),
            Int(warmup_accepted),
            Int(warmup_attempted),
            Int(sampling_accepted),
            Int(sampling_attempted),
        )
    end
end

acceptance_rate(result::EDMCDriverResult) =
    result.attempted == 0 ? 0.0 : result.accepted / result.attempted

warmup_acceptance_rate(result::EDMCDriverResult) =
    result.warmup_attempted == 0 ? 0.0 : result.warmup_accepted / result.warmup_attempted

sampling_acceptance_rate(result::EDMCDriverResult) =
    result.sampling_attempted == 0 ? 0.0 : result.sampling_accepted / result.sampling_attempted

"""
    EDMCObservables

Thermal observables estimated from EDMC samples.

`energy` and `energy2` are sample averages of the Majorana internal-energy
estimator per site and its square per site squared. `energy_beta_derivative`
is `⟨∂E_total/∂β⟩ / N`. `specific_heat` follows the EDMC estimator used in the
Kitaev Monte Carlo literature, `(β^2 / N) * (⟨E_total^2⟩ - ⟨E_total⟩^2 -
⟨∂E_total/∂β⟩)`.
"""
struct EDMCObservables
    beta::Float64
    temperature::Float64
    energy::Float64
    energy2::Float64
    energy_beta_derivative::Float64
    specific_heat::Float64
    nsamples::Int
    nsites::Int

    function EDMCObservables(
        beta::Real,
        temperature::Real,
        energy::Real,
        energy2::Real,
        energy_beta_derivative::Real,
        specific_heat::Real,
        nsamples::Integer,
        nsites::Integer,
    )
        isfinite(beta) && beta > 0 || throw(ArgumentError("beta must be positive and finite; got $beta"))
        isfinite(temperature) && temperature > 0 ||
            throw(ArgumentError("temperature must be positive and finite; got $temperature"))
        all(isfinite, (energy, energy2, energy_beta_derivative, specific_heat)) ||
            throw(ArgumentError("EDMC observables must be finite"))
        nsamples >= 0 || throw(ArgumentError("nsamples must be non-negative; got $nsamples"))
        nsites > 0 || throw(ArgumentError("nsites must be positive; got $nsites"))
        return new(
            Float64(beta),
            Float64(temperature),
            Float64(energy),
            Float64(energy2),
            Float64(energy_beta_derivative),
            Float64(specific_heat),
            Int(nsamples),
            Int(nsites),
        )
    end
end

"""
    EDMCTemperatureScanResult

Structured result for a sequence of EDMC runs over temperatures.
"""
struct EDMCTemperatureScanResult
    temperatures::Vector{Float64}
    observables::Vector{EDMCObservables}
    runs::Vector{EDMCDriverResult}

    function EDMCTemperatureScanResult(
        temperatures::AbstractVector{<:Real},
        observables::Vector{EDMCObservables},
        runs::Vector{EDMCDriverResult},
    )
        all(t -> isfinite(t) && t > 0, temperatures) ||
            throw(ArgumentError("all scan temperatures must be positive and finite"))
        length(temperatures) == length(observables) == length(runs) ||
            throw(ArgumentError("temperatures, observables, and runs must have the same length"))
        return new(Float64.(temperatures), observables, runs)
    end
end
