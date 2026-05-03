using LinearAlgebra
using KrylovKit
using Random
using SparseArrays

export build_sparse_majorana_matrix, sparse_matsubara_matrix
export sparse_matsubara_mul, sparse_matsubara_transpose_mul
export refresh_sparse_real_pseudofermions, sparse_pseudofermion_action
export delta_sparse_pseudofermion_action
export run_sparse_pseudofermion_mc

"""
    build_sparse_majorana_matrix(input; couplings=(x=1.0, y=1.0, z=1.0))

Build the sparse real antisymmetric Majorana matrix `A` from a Kitaev
bond/gauge input. The convention matches [`build_majorana_matrix`](@ref):
each bond contributes `2Jγuij`.
"""
function build_sparse_majorana_matrix(input; couplings=(x=1.0, y=1.0, z=1.0))
    bondset = getproperty(input, :bondset)
    gauge = getproperty(input, :gauge)
    nsites = getproperty(bondset, :nsites)
    bonds = getproperty(bondset, :bonds)
    u = getproperty(gauge, :u)
    length(u) == length(bonds) ||
        throw(ArgumentError("gauge length ($(length(u))) must match number of bonds ($(length(bonds)))"))

    rows = Int[]
    cols = Int[]
    values = Float64[]
    sizehint!(rows, 2 * length(bonds))
    sizehint!(cols, 2 * length(bonds))
    sizehint!(values, 2 * length(bonds))

    for bond in bonds
        index = getproperty(bond, :index)
        src = getproperty(bond, :src)
        dst = getproperty(bond, :dst)
        kind = getproperty(bond, :kind)
        amplitude = 2.0 * _kitaev_coupling(couplings, kind) * u[index]
        push!(rows, src)
        push!(cols, dst)
        push!(values, amplitude)
        push!(rows, dst)
        push!(cols, src)
        push!(values, -amplitude)
    end

    matrix = sparse(rows, cols, values, nsites, nsites)
    _validate_sparse_majorana_matrix(matrix)
    return matrix
end

"""
    sparse_matsubara_matrix(A, beta, n)

Build the sparse Matsubara matrix `M_n = I - A / ω_n` from a sparse real
antisymmetric Majorana matrix `A`.
"""
function sparse_matsubara_matrix(matrix::SparseMatrixCSC{<:Real}, beta::Real, n::Integer)
    _validate_sparse_majorana_matrix(matrix)
    omega = matsubara_frequency(n, beta)
    return sparse(I, size(matrix, 1), size(matrix, 2)) - matrix / omega
end

"""
    sparse_matsubara_mul(A, beta, n, v)

Apply the matrix-free Matsubara operator `M_n v = v - A*v/ω_n`.
"""
function sparse_matsubara_mul(matrix::SparseMatrixCSC{<:Real}, beta::Real, n::Integer, vector::AbstractVector{<:Real})
    _validate_sparse_majorana_matrix(matrix)
    _validate_vector_length(vector, size(matrix, 1), "vector")
    omega = matsubara_frequency(n, beta)
    return Vector{Float64}(vector) - matrix * Vector{Float64}(vector) / omega
end

"""
    sparse_matsubara_transpose_mul(A, beta, n, v)

Apply the transpose matrix-free Matsubara operator
`M_n'v = v + A*v/ω_n`, using `A' = -A`.
"""
function sparse_matsubara_transpose_mul(matrix::SparseMatrixCSC{<:Real}, beta::Real, n::Integer, vector::AbstractVector{<:Real})
    _validate_sparse_majorana_matrix(matrix)
    _validate_vector_length(vector, size(matrix, 1), "vector")
    omega = matsubara_frequency(n, beta)
    return Vector{Float64}(vector) + matrix * Vector{Float64}(vector) / omega
end

"""
    refresh_sparse_real_pseudofermions(A, beta; cutoff, rng=Random.default_rng())
    refresh_sparse_real_pseudofermions(input, beta; cutoff, rng=Random.default_rng(), couplings=(x=1.0, y=1.0, z=1.0))

Sample real pseudofermion fields using sparse Matsubara matrix-vector
multiplication: `φ_n = M_n' * ξ_n`.
"""
function refresh_sparse_real_pseudofermions(
    matrix::SparseMatrixCSC{<:Real},
    beta::Real;
    cutoff::Integer,
    rng::AbstractRNG=Random.default_rng(),
)
    cutoff > 0 || throw(ArgumentError("cutoff must be positive; got $cutoff"))
    _validate_sparse_majorana_matrix(matrix)
    fields = Vector{Vector{Float64}}(undef, cutoff)
    for n in 0:(cutoff - 1)
        xi = randn(rng, size(matrix, 1))
        fields[n + 1] = sparse_matsubara_transpose_mul(matrix, beta, n, xi)
    end
    return fields
end

function refresh_sparse_real_pseudofermions(
    input,
    beta::Real;
    cutoff::Integer,
    rng::AbstractRNG=Random.default_rng(),
    couplings=(x=1.0, y=1.0, z=1.0),
)
    return refresh_sparse_real_pseudofermions(
        build_sparse_majorana_matrix(input; couplings=couplings),
        beta;
        cutoff=cutoff,
        rng=rng,
    )
end

"""
    sparse_pseudofermion_action(A, beta, fields; solver=:gmres, operator=:matrix_free, tol=1e-10, maxiter=1000, krylovdim=30)
    sparse_pseudofermion_action(input, beta, fields; couplings=(x=1.0, y=1.0, z=1.0), ...)

Evaluate the finite-cutoff real pseudofermion action using sparse solves.
`solver=:gmres` uses `KrylovKit.linsolve`; `solver=:direct` uses Julia's
sparse direct solver. With GMRES, `operator=:matrix_free` applies `M_n` by
sparse matrix-vector products without constructing `M_n`, while
`operator=:matrix` constructs the sparse Matsubara matrix explicitly.
"""
function sparse_pseudofermion_action(
    matrix::SparseMatrixCSC{<:Real},
    beta::Real,
    fields;
    solver::Symbol=:gmres,
    operator::Symbol=:matrix_free,
    tol::Real=1e-10,
    maxiter::Integer=1000,
    krylovdim::Integer=30,
)
    _validate_beta(beta)
    _validate_sparse_majorana_matrix(matrix)
    _validate_solver_options(solver, operator, tol, maxiter, krylovdim)
    isempty(fields) && throw(ArgumentError("pseudofermion field collection must be non-empty"))

    action = 0.0
    for (offset, field) in enumerate(fields)
        phi = _validate_pseudofermion_field(field, size(matrix, 1), offset)
        x = _solve_sparse_matsubara_system(
            matrix,
            beta,
            offset - 1,
            phi;
            solver=solver,
            operator=operator,
            tol=tol,
            maxiter=maxiter,
            krylovdim=krylovdim,
        )
        action += 0.5 * dot(phi, x)
    end
    isfinite(action) || throw(ArgumentError("pseudofermion action is not finite"))
    return action
end

function sparse_pseudofermion_action(
    input,
    beta::Real,
    fields;
    couplings=(x=1.0, y=1.0, z=1.0),
    solver::Symbol=:gmres,
    operator::Symbol=:matrix_free,
    tol::Real=1e-10,
    maxiter::Integer=1000,
    krylovdim::Integer=30,
)
    return sparse_pseudofermion_action(
        build_sparse_majorana_matrix(input; couplings=couplings),
        beta,
        fields;
        solver=solver,
        operator=operator,
        tol=tol,
        maxiter=maxiter,
        krylovdim=krylovdim,
    )
end

"""
    delta_sparse_pseudofermion_action(before, after, beta, fields)

Return the sparse pseudofermion action difference for fixed real
pseudofermion fields.
"""
function delta_sparse_pseudofermion_action(
    before::SparseMatrixCSC{<:Real},
    after::SparseMatrixCSC{<:Real},
    beta::Real,
    fields;
    solver::Symbol=:gmres,
    operator::Symbol=:matrix_free,
    tol::Real=1e-10,
    maxiter::Integer=1000,
    krylovdim::Integer=30,
)
    size(before) == size(after) ||
        throw(ArgumentError("matrix sizes must match; got $(size(before)) and $(size(after))"))
    return sparse_pseudofermion_action(after, beta, fields; solver=solver, operator=operator, tol=tol, maxiter=maxiter, krylovdim=krylovdim) -
           sparse_pseudofermion_action(before, beta, fields; solver=solver, operator=operator, tol=tol, maxiter=maxiter, krylovdim=krylovdim)
end

function delta_sparse_pseudofermion_action(
    before,
    after,
    beta::Real,
    fields;
    couplings=(x=1.0, y=1.0, z=1.0),
    solver::Symbol=:gmres,
    operator::Symbol=:matrix_free,
    tol::Real=1e-10,
    maxiter::Integer=1000,
    krylovdim::Integer=30,
)
    return delta_sparse_pseudofermion_action(
        build_sparse_majorana_matrix(before; couplings=couplings),
        build_sparse_majorana_matrix(after; couplings=couplings),
        beta,
        fields;
        solver=solver,
        operator=operator,
        tol=tol,
        maxiter=maxiter,
        krylovdim=krylovdim,
    )
end

"""
    run_sparse_pseudofermion_mc(input, beta; cutoff, warmup_sweeps=0, sampling_sweeps, seed=nothing, rng=nothing, couplings=(x=1.0, y=1.0, z=1.0), solver=:gmres, operator=:matrix_free)

Run a sparse real-pseudofermion Z2 bond-flip Monte Carlo simulation. One sweep
performs one attempted flip per bond. Each attempt refreshes pseudofermions
from the current gauge configuration, proposes one random bond flip, and
accepts with `min(1, exp(-ΔS_pf))`.
"""
function run_sparse_pseudofermion_mc(
    input,
    beta::Real;
    cutoff::Integer,
    warmup_sweeps::Integer=0,
    sampling_sweeps::Integer,
    seed=nothing,
    rng=nothing,
    couplings=(x=1.0, y=1.0, z=1.0),
    solver::Symbol=:gmres,
    operator::Symbol=:matrix_free,
    tol::Real=1e-10,
    maxiter::Integer=1000,
    krylovdim::Integer=30,
)
    _validate_beta(beta)
    cutoff > 0 || throw(ArgumentError("cutoff must be positive; got $cutoff"))
    warmup_sweeps >= 0 || throw(ArgumentError("warmup_sweeps must be non-negative; got $warmup_sweeps"))
    sampling_sweeps >= 0 || throw(ArgumentError("sampling_sweeps must be non-negative; got $sampling_sweeps"))
    _validate_solver_options(solver, operator, tol, maxiter, krylovdim)
    local_rng = _sparse_pf_rng(seed, rng)

    nbonds = length(getproperty(getproperty(input, :bondset), :bonds))
    nbonds > 0 || throw(ArgumentError("pseudofermion MC requires at least one bond"))

    current = input
    warmup_accepted = 0
    sampling_accepted = 0
    samples = Vector{Int8}[]

    for _ in 1:warmup_sweeps
        current, accepted = _run_sparse_pseudofermion_sweep(
            current,
            local_rng,
            beta,
            cutoff,
            nbonds;
            couplings=couplings,
            solver=solver,
            operator=operator,
            tol=tol,
            maxiter=maxiter,
            krylovdim=krylovdim,
        )
        warmup_accepted += accepted
    end

    for _ in 1:sampling_sweeps
        current, accepted = _run_sparse_pseudofermion_sweep(
            current,
            local_rng,
            beta,
            cutoff,
            nbonds;
            couplings=couplings,
            solver=solver,
            operator=operator,
            tol=tol,
            maxiter=maxiter,
            krylovdim=krylovdim,
        )
        sampling_accepted += accepted
        push!(samples, copy(getproperty(getproperty(current, :gauge), :u)))
    end

    warmup_attempted = warmup_sweeps * nbonds
    sampling_attempted = sampling_sweeps * nbonds
    accepted = warmup_accepted + sampling_accepted
    attempted = warmup_attempted + sampling_attempted
    return (
        final_input=current,
        samples=samples,
        accepted=accepted,
        attempted=attempted,
        warmup_accepted=warmup_accepted,
        warmup_attempted=warmup_attempted,
        sampling_accepted=sampling_accepted,
        sampling_attempted=sampling_attempted,
        acceptance_rate=attempted == 0 ? 0.0 : accepted / attempted,
        sampling_acceptance_rate=sampling_attempted == 0 ? 0.0 : sampling_accepted / sampling_attempted,
    )
end

function _run_sparse_pseudofermion_sweep(
    input,
    rng::AbstractRNG,
    beta::Real,
    cutoff::Integer,
    attempts::Integer;
    couplings,
    solver::Symbol,
    operator::Symbol,
    tol::Real,
    maxiter::Integer,
    krylovdim::Integer,
)
    current = input
    accepted = 0
    for _ in 1:attempts
        fields = refresh_sparse_real_pseudofermions(current, beta; cutoff=cutoff, rng=rng, couplings=couplings)
        bond = rand(rng, 1:attempts)
        proposed = EDMC.flip_gauge(current, bond)
        delta_action = delta_sparse_pseudofermion_action(
            current,
            proposed,
            beta,
            fields;
            couplings=couplings,
            solver=solver,
            operator=operator,
            tol=tol,
            maxiter=maxiter,
            krylovdim=krylovdim,
        )
        if rand(rng) < acceptance_probability_pseudofermion(delta_action)
            current = proposed
            accepted += 1
        end
    end
    return current, accepted
end

function _solve_sparse_matsubara_system(
    matrix::SparseMatrixCSC{<:Real},
    beta::Real,
    n::Integer,
    rhs::AbstractVector{<:Real};
    solver::Symbol,
    operator::Symbol,
    tol::Real,
    maxiter::Integer,
    krylovdim::Integer,
)
    solver === :direct && return sparse_matsubara_matrix(matrix, beta, n) \ rhs
    if solver === :gmres
        linear_map = if operator === :matrix_free
            vector -> sparse_matsubara_mul(matrix, beta, n, vector)
        elseif operator === :matrix
            sparse_matsubara_matrix(matrix, beta, n)
        else
            throw(ArgumentError("unsupported operator :$operator; expected :matrix_free or :matrix"))
        end
        x, info = linsolve(
            linear_map,
            Vector{Float64}(rhs),
            zeros(Float64, length(rhs)),
            GMRES(maxiter=maxiter, krylovdim=krylovdim, tol=Float64(tol), verbosity=0),
        )
        info.converged > 0 ||
            throw(ArgumentError("GMRES did not converge; residual=$(info.normres), iterations=$(info.numiter)"))
        return x
    end
    throw(ArgumentError("unsupported sparse solver :$solver; expected :gmres or :direct"))
end

function _validate_solver_options(solver::Symbol, operator::Symbol, tol::Real, maxiter::Integer, krylovdim::Integer)
    (solver === :gmres || solver === :direct) ||
        throw(ArgumentError("unsupported sparse solver :$solver; expected :gmres or :direct"))
    (operator === :matrix_free || operator === :matrix) ||
        throw(ArgumentError("unsupported operator :$operator; expected :matrix_free or :matrix"))
    isfinite(tol) && tol > 0 ||
        throw(ArgumentError("tol must be a positive finite number; got $tol"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive; got $maxiter"))
    krylovdim > 0 || throw(ArgumentError("krylovdim must be positive; got $krylovdim"))
    return nothing
end

function _validate_vector_length(vector::AbstractVector, expected_length::Integer, name::AbstractString)
    length(vector) == expected_length ||
        throw(ArgumentError("$name has length $(length(vector)); expected $expected_length"))
    all(isfinite, vector) ||
        throw(ArgumentError("$name contains NaN or Inf entries"))
    return nothing
end

function _sparse_pf_rng(seed, rng)
    if rng !== nothing && seed !== nothing
        throw(ArgumentError("pass either rng or seed, not both"))
    elseif rng !== nothing
        rng isa AbstractRNG || throw(ArgumentError("rng must be an AbstractRNG; got $(typeof(rng))"))
        return rng
    elseif seed !== nothing
        return MersenneTwister(seed)
    else
        return Random.default_rng()
    end
end

function _validate_sparse_majorana_matrix(matrix::SparseMatrixCSC{<:Real})
    _validate_square_even_finite(matrix, "Sparse Majorana matrix")
    maximum(abs, Matrix(matrix + transpose(matrix))) <= 1e-10 ||
        throw(ArgumentError("Sparse Majorana matrix must be real antisymmetric within tolerance"))
    return nothing
end
