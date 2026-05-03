abstract type BoundaryType end

"Type-I boundary condition: open along a and periodic along b."
struct TypeI <: BoundaryType end

"Type-II boundary condition: shifted periodic along a, periodic along b, with one omitted x bond."
struct TypeII <: BoundaryType end

Base.@kwdef struct SiteCoord
    x::Int
    y::Int
    z::Int
    sublattice::Int
end

Base.@kwdef struct BondEdge
    src::SiteCoord
    dst::SiteCoord
    bond::Symbol
    wrapped::Bool = false
end

Base.@kwdef struct Lattice
    kind::Symbol
    dims::NTuple{3,Int}
    boundary::DataType
    nsites::Int
    edges::Vector{BondEdge}
end

@inline function _neighbor_index(i::Int, L::Int, periodic::Bool)
    if 1 <= i <= L
        return i, false
    elseif periodic
        return mod1(i, L), true
    else
        return nothing, false
    end
end

function _cell_index(x::Int, y::Int, z::Int, sublattice::Int, dims::NTuple{3,Int}, nsublat::Int)
    Lx, Ly, Lz = dims
    return ((((z - 1) * Ly + (y - 1)) * Lx + (x - 1)) * nsublat) + sublattice
end

function _honeycomb_periodicity(::TypeI)
    return (false, true, false)
end

function _honeycomb_periodicity(::TypeII)
    return (false, true, false)
end

function _hyperhoneycomb_periodicity(::TypeI)
    return (false, false, true)
end

function _hyperhoneycomb_periodicity(::TypeII)
    throw(ArgumentError("Type-II boundary condition is only defined for honeycomb lattices"))
end

function generate_honeycomb(Lx::Int, Ly::Int, bc::BoundaryType)
    Lx > 0 || throw(ArgumentError("Lx must be positive"))
    Ly > 0 || throw(ArgumentError("Ly must be positive"))

    px, py, _ = _honeycomb_periodicity(bc)
    dims = (Lx, Ly, 1)
    edges = BondEdge[]

    for y in 1:Ly, x in 1:Lx
        a = SiteCoord(x=x, y=y, z=1, sublattice=1)
        b = SiteCoord(x=x, y=y, z=1, sublattice=2)
        push!(edges, BondEdge(src=a, dst=b, bond=:z, wrapped=false))

        _push_honeycomb_x_bond!(edges, a, x, y, Lx, Ly, bc, px)

        ny, wy = _neighbor_index(y - 1, Ly, py)
        if ny !== nothing
            yb = SiteCoord(x=x, y=ny, z=1, sublattice=2)
            push!(edges, BondEdge(src=a, dst=yb, bond=:y, wrapped=wy))
        end
    end

    return Lattice(kind=:honeycomb, dims=dims, boundary=typeof(bc), nsites=2 * Lx * Ly, edges=edges)
end

function _push_honeycomb_x_bond!(edges::Vector{BondEdge}, a::SiteCoord, x::Int, y::Int, Lx::Int, Ly::Int, ::TypeI, px::Bool)
    nx, wx = _neighbor_index(x - 1, Lx, px)
    if nx !== nothing
        xb = SiteCoord(x=nx, y=y, z=1, sublattice=2)
        push!(edges, BondEdge(src=a, dst=xb, bond=:x, wrapped=wx))
    end
    return edges
end

function _push_honeycomb_x_bond!(edges::Vector{BondEdge}, a::SiteCoord, x::Int, y::Int, Lx::Int, Ly::Int, ::TypeII, px::Bool)
    if x > 1
        xb = SiteCoord(x=x - 1, y=y, z=1, sublattice=2)
        push!(edges, BondEdge(src=a, dst=xb, bond=:x, wrapped=false))
    elseif y != 1
        xb = SiteCoord(x=Lx, y=mod1(y + 1, Ly), z=1, sublattice=2)
        push!(edges, BondEdge(src=a, dst=xb, bond=:x, wrapped=true))
    end
    return edges
end

function generate_hyperhoneycomb(Lx::Int, Ly::Int, Lz::Int, bc::BoundaryType)
    Lx > 0 || throw(ArgumentError("Lx must be positive"))
    Ly > 0 || throw(ArgumentError("Ly must be positive"))
    Lz > 0 || throw(ArgumentError("Lz must be positive"))

    px, py, pz = _hyperhoneycomb_periodicity(bc)
    dims = (Lx, Ly, Lz)
    edges = BondEdge[]

    for z in 1:Lz, y in 1:Ly, x in 1:Lx
        s1 = SiteCoord(x=x, y=y, z=z, sublattice=1)
        s2 = SiteCoord(x=x, y=y, z=z, sublattice=2)
        s3 = SiteCoord(x=x, y=y, z=z, sublattice=3)
        s4 = SiteCoord(x=x, y=y, z=z, sublattice=4)

        push!(edges, BondEdge(src=s1, dst=s2, bond=:z, wrapped=false))
        push!(edges, BondEdge(src=s2, dst=s3, bond=:x, wrapped=false))
        push!(edges, BondEdge(src=s3, dst=s4, bond=:z, wrapped=false))

        nx, wx = _neighbor_index(x + 1, Lx, px)
        if nx !== nothing
            t1 = SiteCoord(x=nx, y=y, z=z, sublattice=1)
            push!(edges, BondEdge(src=s4, dst=t1, bond=:y, wrapped=wx))
        end

        ny, wy = _neighbor_index(y + 1, Ly, py)
        if ny !== nothing
            t3 = SiteCoord(x=x, y=ny, z=z, sublattice=3)
            push!(edges, BondEdge(src=s2, dst=t3, bond=:y, wrapped=wy))
        end

        nz, wz = _neighbor_index(z + 1, Lz, pz)
        if nz !== nothing
            t4 = SiteCoord(x=x, y=y, z=nz, sublattice=4)
            push!(edges, BondEdge(src=s1, dst=t4, bond=:x, wrapped=wz))
        end
    end

    return Lattice(kind=:hyperhoneycomb, dims=dims, boundary=typeof(bc), nsites=4 * Lx * Ly * Lz, edges=edges)
end
