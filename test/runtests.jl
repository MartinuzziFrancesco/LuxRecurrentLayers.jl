using LuxRecurrentLayers
using Test
using Aqua
using JET

@testset "LuxRecurrentLayers.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(LuxRecurrentLayers; ambiguities = false, deps_compat = false)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(LuxRecurrentLayers; target_defined_modules = true)
    end
    # Write your tests here.
end
