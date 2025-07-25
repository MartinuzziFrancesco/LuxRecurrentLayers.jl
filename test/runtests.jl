using Test, LuxTestUtils, ReTestItems, LuxRecurrentLayers

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "all"))

# if you support CUDA / AMDGPU, load them here
if BACKEND_GROUP in ("all", "cuda")
    try
        using LuxCUDA
        @info "LuxCUDA: $(sprint(CUDA.versioninfo))"
    catch e
        @warn "Could not load LuxCUDA, CUDA tests will be skipped" exception = (e, catch_backtrace())
    end
end
if BACKEND_GROUP in ("all", "amdgpu")
    try
        using AMDGPU
        @info "AMDGPU loaded"
    catch e
        @warn "Could not load AMDGPU, GPU tests will be skipped" exception = (e, catch_backtrace())
    end
end

using Lux, MLDataDevices, StableRNGs

#include(joinpath(@__DIR__, "qa.jl"))
include(joinpath(@__DIR__, "setups.jl"))
include(joinpath(@__DIR__, "cells.jl"))
