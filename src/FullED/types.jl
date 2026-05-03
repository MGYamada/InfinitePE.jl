"""
    FullEDHamiltonianInput(bondset)

Input for full spin exact diagonalization. `bondset` is obtained from an
`InfinitePE` lattice and determines the Kitaev x/y/z spin interactions.
"""
struct FullEDHamiltonianInput
    bondset::KitaevBondSet

    function FullEDHamiltonianInput(bondset::KitaevBondSet)
        bondset.nsites > 0 || throw(ArgumentError("bondset must contain at least one site"))
        bondset.nsites <= 62 ||
            throw(ArgumentError("Full ED bit-basis indexing supports at most 62 sites; got $(bondset.nsites)"))
        return new(bondset)
    end
end

"""
    FullEDDiagonalizationResult(eigenvalues, eigenvectors)

Full spin Hilbert-space diagonalization result.
"""
struct FullEDDiagonalizationResult{T<:Real,V<:AbstractMatrix}
    eigenvalues::Vector{T}
    eigenvectors::V

    function FullEDDiagonalizationResult(eigenvalues::AbstractVector{<:Real}, eigenvectors::AbstractMatrix)
        length(eigenvalues) == size(eigenvectors, 1) ||
            throw(ArgumentError(
                "number of eigenvalues ($(length(eigenvalues))) must match eigenvector row count ($(size(eigenvectors, 1)))",
            ))
        all(isfinite, eigenvalues) || throw(ArgumentError("diagonalization produced non-finite eigenvalues"))
        return new{eltype(collect(eigenvalues)),typeof(eigenvectors)}(collect(eigenvalues), eigenvectors)
    end
end

"""
    FullEDObservables

Exact thermal observables from the full spin spectrum.
"""
struct FullEDObservables
    beta::Float64
    temperature::Float64
    energy::Float64
    energy2::Float64
    specific_heat::Float64
    free_energy::Float64
    entropy::Float64
    nstates::Int
    nsites::Int

    function FullEDObservables(
        beta::Real,
        temperature::Real,
        energy::Real,
        energy2::Real,
        specific_heat::Real,
        free_energy::Real,
        entropy::Real,
        nstates::Integer,
        nsites::Integer,
    )
        isfinite(beta) && beta > 0 || throw(ArgumentError("beta must be positive and finite; got $beta"))
        isfinite(temperature) && temperature > 0 ||
            throw(ArgumentError("temperature must be positive and finite; got $temperature"))
        nstates > 0 || throw(ArgumentError("nstates must be positive; got $nstates"))
        nsites > 0 || throw(ArgumentError("nsites must be positive; got $nsites"))
        all(isfinite, (energy, energy2, specific_heat, free_energy, entropy)) ||
            throw(ArgumentError("FullED observables must be finite"))
        specific_heat >= -1e-12 ||
            throw(ArgumentError("specific_heat must be non-negative within tolerance; got $specific_heat"))
        return new(
            Float64(beta),
            Float64(temperature),
            Float64(energy),
            Float64(energy2),
            max(0.0, Float64(specific_heat)),
            Float64(free_energy),
            Float64(entropy),
            Int(nstates),
            Int(nsites),
        )
    end
end

"""
    FullEDTemperatureScanResult(temperatures, observables, diag_result)

Temperature scan result sharing one exact spectrum across all temperatures.
"""
struct FullEDTemperatureScanResult
    temperatures::Vector{Float64}
    observables::Vector{FullEDObservables}
    diag_result::FullEDDiagonalizationResult

    function FullEDTemperatureScanResult(
        temperatures::AbstractVector{<:Real},
        observables::Vector{FullEDObservables},
        diag_result::FullEDDiagonalizationResult,
    )
        length(temperatures) == length(observables) ||
            throw(ArgumentError("temperatures and observables must have the same length"))
        return new(Float64.(temperatures), observables, diag_result)
    end
end
