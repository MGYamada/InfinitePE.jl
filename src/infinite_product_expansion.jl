using LinearAlgebra
using Random

export build_majorana_matrix, matsubara_frequency, majorana_matrix, matsubara_matrix
export logdet_matsubara_dense, log_weight_infinite_product, delta_log_weight_infinite_product
export acceptance_probability_logweight
export refresh_real_pseudofermions, pseudofermion_action, delta_pseudofermion_action
export acceptance_probability_pseudofermion
export log_weight, delta_log_weight, acceptance_probability

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
`S_pf = 1/2 * sum(φ_n' * M_n(A)^(-1) * φ_n)`.
"""
function pseudofermion_action(matrix::AbstractMatrix, beta::Real, fields; atol::Real=1e-10)
    _validate_beta(beta)
    majorana = _as_majorana_matrix(matrix; atol=atol)
    isempty(fields) && throw(ArgumentError("pseudofermion field collection must be non-empty"))

    action = 0.0
    for (offset, field) in enumerate(fields)
        phi = _validate_pseudofermion_field(field, size(majorana, 1), offset)
        m = matsubara_matrix(majorana, beta, offset - 1; atol=atol)
        x = m \ phi
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
