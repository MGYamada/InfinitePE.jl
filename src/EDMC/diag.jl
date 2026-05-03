"""
    diagonalize(hamiltonian)

Diagonalize an EDMC Hamiltonian using `LinearAlgebra.eigen`.

The input must be a square finite matrix. Hermitian matrices are routed through
`Hermitian` so the returned eigenvalues are real and reproducible for identical
input.
"""
function diagonalize(hamiltonian::AbstractMatrix)
    _validate_hamiltonian_matrix(hamiltonian)

    eig = if ishermitian(hamiltonian)
        eigen(Hermitian(Matrix(hamiltonian)))
    else
        eigen(Matrix(hamiltonian))
    end

    maximum(abs, imag.(eig.values)) <= 1e-10 ||
        throw(ArgumentError("EDMC Hamiltonian eigenvalues must be real within tolerance"))

    return DiagonalizationResult(real.(eig.values), eig.vectors)
end

"""
    diagonalize(hamiltonian)

Diagonalize an EDMC Hamiltonian and return the solver result.

This signature is reserved for the exact diagonalization backend.
"""
function diagonalize(hamiltonian)
    throw(ArgumentError("EDMC diagonalization expects a matrix; got $(typeof(hamiltonian))"))
end

"""
    majorana_energies(result; atol=1e-10)

Return the non-negative single-particle energies from a particle-hole paired
Majorana spectrum.

The diagonalized Hermitian Majorana hopping matrix has paired eigenvalues
`±λ`. Thermodynamic formulas in this module follow the convention
`H = sum(ε * (f†f - 1/2))`, so the returned excitation energies are `ε = 2λ`.
"""
function majorana_energies(result::DiagonalizationResult; atol::Real=1e-10)
    atol >= 0 || throw(ArgumentError("atol must be non-negative; got $atol"))
    values = sort(result.eigenvalues)
    positives = [v for v in values if v > atol]
    nzero = count(v -> abs(v) <= atol, values)

    iseven(nzero) ||
        throw(ArgumentError("Majorana spectrum has an odd number of zero modes ($nzero); cannot form paired energies"))

    energies = vcat(fill(0.0, div(nzero, 2)), 2 .* positives)
    expected = div(length(values), 2)
    length(energies) == expected ||
        throw(ArgumentError(
            "Majorana spectrum is not paired: expected $expected non-negative energies, got $(length(energies))",
        ))

    return energies
end

function _validate_hamiltonian_matrix(hamiltonian::AbstractMatrix)
    size(hamiltonian, 1) == size(hamiltonian, 2) ||
        throw(ArgumentError("EDMC Hamiltonian must be square; got size $(size(hamiltonian))"))
    iseven(size(hamiltonian, 1)) ||
        throw(ArgumentError("EDMC Majorana Hamiltonian dimension must be even; got $(size(hamiltonian, 1))"))
    all(isfinite, hamiltonian) ||
        throw(ArgumentError("EDMC Hamiltonian contains NaN or Inf entries"))
    return nothing
end
