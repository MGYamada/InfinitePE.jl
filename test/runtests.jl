using Test
using InfinitePE

@testset "Honeycomb lattice generation" begin
    lat1 = generate_honeycomb(4, 3, TypeI())
    @test lat1.kind == :honeycomb
    @test lat1.nsites == 24
    @test lat1.boundary == TypeI
    @test !any(e -> e.bond == :y && e.wrapped, lat1.edges)

    lat2 = generate_honeycomb(4, 3, TypeII())
    @test lat2.kind == :honeycomb
    @test lat2.nsites == 24
    @test lat2.boundary == TypeII
    @test any(e -> e.bond == :y && e.wrapped, lat2.edges)
    @test length(lat2.edges) > length(lat1.edges)
end

@testset "Hyperhoneycomb lattice generation" begin
    lat1 = generate_hyperhoneycomb(3, 2, 2, TypeI())
    @test lat1.kind == :hyperhoneycomb
    @test lat1.nsites == 48
    @test lat1.boundary == TypeI
    @test !any(e -> e.wrapped && (e.src.y != e.dst.y), lat1.edges)
    @test !any(e -> e.wrapped && (e.src.z != e.dst.z), lat1.edges)

    lat2 = generate_hyperhoneycomb(3, 2, 2, TypeII())
    @test lat2.kind == :hyperhoneycomb
    @test lat2.nsites == 48
    @test lat2.boundary == TypeII
    @test any(e -> e.wrapped && (e.src.y != e.dst.y), lat2.edges)
    @test any(e -> e.wrapped && (e.src.z != e.dst.z), lat2.edges)
    @test length(lat2.edges) > length(lat1.edges)
end
