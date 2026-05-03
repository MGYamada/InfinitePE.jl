function _is_connected_xy_chain(lat)
    adjacency = [Int[] for _ in 1:lat.nsites]
    for edge in lat.edges
        edge.bond in (:x, :y) || continue
        src = _test_site_index(edge.src, lat.dims)
        dst = _test_site_index(edge.dst, lat.dims)
        push!(adjacency[src], dst)
        push!(adjacency[dst], src)
    end

    @test count(==(1), length.(adjacency)) == 2
    @test all(degree -> degree <= 2, length.(adjacency))

    seen = falses(lat.nsites)
    stack = [findfirst(!isempty, adjacency)]
    while !isempty(stack)
        site = pop!(stack)
        seen[site] && continue
        seen[site] = true
        append!(stack, adjacency[site])
    end
    return all(seen)
end

function _test_site_index(coord, dims)
    return (((coord.y - 1) * dims[1] + (coord.x - 1)) * 2) + coord.sublattice
end

@testset "Honeycomb lattice generation" begin
    lat1 = generate_honeycomb(4, 3, TypeI())
    @test lat1.kind == :honeycomb
    @test lat1.nsites == 24
    @test lat1.boundary == TypeI
    @test !any(e -> e.bond == :x && e.wrapped, lat1.edges)
    @test any(e -> e.bond == :y && e.wrapped, lat1.edges)

    lat2 = generate_honeycomb(4, 3, TypeII())
    @test lat2.kind == :honeycomb
    @test lat2.nsites == 24
    @test lat2.boundary == TypeII
    @test any(e -> e.bond == :x && e.wrapped, lat2.edges)
    @test any(e -> e.bond == :y && e.wrapped, lat2.edges)
    @test count(e -> e.bond in (:x, :y), lat2.edges) == lat2.nsites - 1
    @test !any(e -> e.bond == :x && e.src.x == 1 && e.src.y == 1 && e.wrapped, lat2.edges)
    @test _is_connected_xy_chain(lat2)
end

@testset "Hyperhoneycomb lattice generation" begin
    lat1 = generate_hyperhoneycomb(3, 2, 2, TypeI())
    @test lat1.kind == :hyperhoneycomb
    @test lat1.nsites == 48
    @test lat1.boundary == TypeI
    @test !any(e -> e.wrapped && (e.src.x != e.dst.x), lat1.edges)
    @test !any(e -> e.wrapped && (e.src.y != e.dst.y), lat1.edges)
    @test any(e -> e.wrapped && (e.src.z != e.dst.z), lat1.edges)

    @test_throws ArgumentError generate_hyperhoneycomb(3, 2, 2, TypeII())
end
