"""
    diagonalize(hamiltonian)

Diagonalize a full spin Hamiltonian and return all eigenpairs.
"""
function diagonalize(hamiltonian::AbstractMatrix)
    _validate_hamiltonian_matrix(hamiltonian)
    eig = if ishermitian(hamiltonian)
        eigen(Hermitian(Matrix(hamiltonian)))
    else
        eigen(Matrix(hamiltonian))
    end
    maximum(abs, imag.(eig.values)) <= 1e-10 ||
        throw(ArgumentError("FullED Hamiltonian eigenvalues must be real within tolerance"))
    return FullEDDiagonalizationResult(real.(eig.values), eig.vectors)
end

function diagonalize(input::FullEDHamiltonianInput; kwargs...)
    return diagonalize(build_hamiltonian(input; kwargs...))
end

"""
    spectrum(hamiltonian)
    spectrum(input; kwargs...)

Compute only the full spin Hamiltonian eigenvalues. This is the preferred
path for thermal observables because no eigenvectors are needed.
"""
function spectrum(hamiltonian::AbstractMatrix)
    _validate_hamiltonian_matrix(hamiltonian)
    values = if ishermitian(hamiltonian)
        eigvals(Hermitian(Matrix(hamiltonian)))
    else
        eigvals(Matrix(hamiltonian))
    end
    maximum(abs, imag.(values)) <= 1e-10 ||
        throw(ArgumentError("FullED Hamiltonian eigenvalues must be real within tolerance"))
    eigenvectors = Matrix{Float64}(undef, length(values), 0)
    return FullEDDiagonalizationResult(real.(values), eigenvectors)
end

function spectrum(input::FullEDHamiltonianInput; kwargs...)
    return spectrum(build_hamiltonian(input; kwargs...))
end

function _validate_hamiltonian_matrix(hamiltonian::AbstractMatrix)
    size(hamiltonian, 1) == size(hamiltonian, 2) ||
        throw(ArgumentError("FullED Hamiltonian must be square; got size $(size(hamiltonian))"))
    all(isfinite, hamiltonian) || throw(ArgumentError("FullED Hamiltonian contains NaN or Inf entries"))
    return nothing
end
