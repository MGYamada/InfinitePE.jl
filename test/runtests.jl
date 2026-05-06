using Test
using InfinitePE

function _with_ci_group(body, name::AbstractString)
    is_github_actions = get(ENV, "GITHUB_ACTIONS", "false") == "true"
    is_github_actions && println("::group::$name")
    try
        return body()
    finally
        is_github_actions && println("::endgroup::")
    end
end

_with_ci_group("InfinitePE test suite") do
    @testset "InfinitePE test suite" begin
        include("lattices.jl")
        include("infinite_product_expansion.jl")
        include("sparse.jl")
        include("edmc.jl")
        include("fulled.jl")
    end
end
