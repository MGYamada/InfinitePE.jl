module InfinitePE

include("lattices.jl")
include("infinite_product_expansion.jl")
include("sparse.jl")
include("EDMC/EDMC.jl")
include("FullED/FullED.jl")

export BoundaryType, TypeI, TypeII, SiteCoord, BondEdge, Lattice
export generate_honeycomb, generate_hyperhoneycomb
export EDMC, FullED

end
