"""
    lattice_to_edmc(lattice)

Convert an `InfinitePE` lattice object into the internal representation used
by EDMC routines.

This extracts a deterministic [`KitaevBondSet`](@ref) and initializes all
`u_ij` gauge variables to `+1`.
"""
function lattice_to_edmc(lattice; gauge_value::Integer=1)
    bondset = extract_kitaev_bonds(lattice)
    gauge = initialize_z2_gauge(bondset; value=gauge_value)
    return KitaevHamiltonianInput(bondset, gauge)
end

"""
    extract_kitaev_bonds(lattice)

Extract `x`, `y`, and `z` bonds from an `InfinitePE` lattice into a
deterministically ordered [`KitaevBondSet`](@ref).

The stable order is by bond kind (`x`, then `y`, then `z`), source coordinate,
destination coordinate, and wrap flag. This keeps the bond index to `u_ij`
mapping reproducible even if the source lattice edge order changes.
"""
function extract_kitaev_bonds(lattice)
    _require_lattice_field(lattice, :kind)
    _require_lattice_field(lattice, :dims)
    _require_lattice_field(lattice, :nsites)
    _require_lattice_field(lattice, :edges)

    dims = _normalize_dims(lattice.dims)
    nsites = lattice.nsites
    nsites isa Integer && nsites > 0 ||
        throw(ArgumentError("lattice.nsites must be a positive integer; got $(repr(nsites))"))

    nsublattices = _infer_nsublattices(dims, nsites)

    raw_bonds = Tuple{Int,Int,Symbol,Bool,Any,Any,Tuple}[]
    for edge in lattice.edges
        _require_lattice_field(edge, :src)
        _require_lattice_field(edge, :dst)
        _require_lattice_field(edge, :bond)
        _require_lattice_field(edge, :wrapped)

        kind = edge.bond
        kind in KITAEV_BOND_KINDS || continue

        src_coord = edge.src
        dst_coord = edge.dst
        src = _site_index(src_coord, dims, nsublattices)
        dst = _site_index(dst_coord, dims, nsublattices)
        key = (_bond_order(kind), _coord_key(src_coord), _coord_key(dst_coord), edge.wrapped ? 1 : 0)
        push!(raw_bonds, (src, dst, kind, Bool(edge.wrapped), src_coord, dst_coord, key))
    end

    isempty(raw_bonds) &&
        throw(ArgumentError("lattice contains no Kitaev x/y/z bonds in lattice.edges"))

    sort!(raw_bonds; by=last)
    bonds = KitaevBond[
        KitaevBond(i, src, dst, kind, wrapped, src_coord, dst_coord)
        for (i, (src, dst, kind, wrapped, src_coord, dst_coord, _)) in enumerate(raw_bonds)
    ]

    return KitaevBondSet(Symbol(lattice.kind), dims, Int(nsites), bonds)
end

"""
    initialize_z2_gauge(bondset; value=1)

Create a `Z2GaugeField` aligned with `bondset`. The default initializes every
`u_ij` to `+1`.
"""
function initialize_z2_gauge(bondset::KitaevBondSet; value::Integer=1)
    value == 1 || value == -1 ||
        throw(ArgumentError("initial Z2 gauge value must be +1 or -1; got $value"))
    return Z2GaugeField(fill(Int8(value), length(bondset.bonds)))
end

function _require_lattice_field(object, name::Symbol)
    hasproperty(object, name) ||
        throw(ArgumentError("expected $(typeof(object)) to have field/property :$name"))
    return nothing
end

function _normalize_dims(dims)
    dims isa NTuple{3,<:Integer} ||
        throw(ArgumentError("lattice.dims must be a 3-tuple of integers; got $(repr(dims))"))
    all(>(0), dims) ||
        throw(ArgumentError("lattice.dims entries must be positive; got $(repr(dims))"))
    return Int.(dims)
end

function _infer_nsublattices(dims::NTuple{3,Int}, nsites::Integer)
    ncells = prod(dims)
    nsites % ncells == 0 ||
        throw(ArgumentError("lattice.nsites ($nsites) must be divisible by number of cells ($ncells)"))
    nsublattices = div(nsites, ncells)
    nsublattices > 0 ||
        throw(ArgumentError("inferred number of sublattices must be positive; got $nsublattices"))
    return nsublattices
end

function _site_index(coord, dims::NTuple{3,Int}, nsublattices::Int)
    for name in (:x, :y, :z, :sublattice)
        _require_lattice_field(coord, name)
    end

    x, y, z, sublattice = coord.x, coord.y, coord.z, coord.sublattice
    1 <= x <= dims[1] ||
        throw(ArgumentError("site coordinate x=$x is outside lattice dimension 1:$(dims[1])"))
    1 <= y <= dims[2] ||
        throw(ArgumentError("site coordinate y=$y is outside lattice dimension 1:$(dims[2])"))
    1 <= z <= dims[3] ||
        throw(ArgumentError("site coordinate z=$z is outside lattice dimension 1:$(dims[3])"))
    1 <= sublattice <= nsublattices ||
        throw(ArgumentError("site sublattice=$sublattice is outside inferred range 1:$nsublattices"))

    return ((((z - 1) * dims[2] + (y - 1)) * dims[1] + (x - 1)) * nsublattices) + sublattice
end

function _coord_key(coord)
    for name in (:x, :y, :z, :sublattice)
        _require_lattice_field(coord, name)
    end
    return (coord.x, coord.y, coord.z, coord.sublattice)
end

function _bond_order(kind::Symbol)
    kind === :x && return 1
    kind === :y && return 2
    kind === :z && return 3
    throw(ArgumentError("unsupported Kitaev bond kind :$kind"))
end
