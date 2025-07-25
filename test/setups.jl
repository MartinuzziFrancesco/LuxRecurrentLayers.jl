@testsetup module RecurrentLayersSetup

import Reexport: @reexport
@reexport using LuxRecurrentLayers

const RECURRENT_CELLS = [
    (:AntisymmetricRNNCell,
        (; kwargs...) -> AntisymmetricRNNCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:ATRCell,
        (; kwargs...) -> ATRCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),]

function loss_loop(cell, x, p, st)
    (y, carry), st_ = cell(x, p, st)
    for _ in 1:3
        (y, carry), st_ = cell((x, carry), p, st_)
    end
    return sum(abs2, y)
end

function loss_loop_no_carry(cell, x, p, st)
    y, st_ = cell(x, p, st)
    for i in 1:3
        y, st_ = cell(x, p, st_)
    end
    return sum(abs2, y)
end

export loss_loop, loss_loop_no_carry, RECURRENT_CELLS

end

@testsetup module SharedTestSetup

import Reexport: @reexport

@reexport using LuxTestUtils, Lux

using MLDataDevices, LuxCUDA, StableRNGs,
    LinearAlgebra, JET

if !@isdefined(BACKEND_GROUP)
    const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "all"))
end

if BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda"
    using LuxCUDA
end

if BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu"
    using AMDGPU
end

cpu_testing() = BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
function cuda_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") &&
           MLDataDevices.functional(CUDADevice)
end
function amdgpu_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") &&
           MLDataDevices.functional(AMDGPUDevice)
end

const MODES = begin
    # Mode, Array Type, Device Function, GPU?
    modes = []
    cpu_testing() && push!(modes, ("cpu", Array, CPUDevice(), false))
    cuda_testing() && push!(modes, ("cuda", CuArray, CUDADevice(), true))
    amdgpu_testing() && push!(modes, ("amdgpu", ROCArray, AMDGPUDevice(), true))

    modes
end

LuxTestUtils.jet_target_modules!(["Lux", "LuxCore", "LuxLib"])
LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

# Some Helper Functions
function get_default_rng(mode::String)
    dev = if mode == "cpu"
        CPUDevice()
    elseif mode == "cuda"
        CUDADevice()
    elseif mode == "amdgpu"
        AMDGPUDevice()
    else
        nothing
    end
    rng = default_device_rng(dev)
    return rng isa TaskLocalRNG ? copy(rng) : deepcopy(rng)
end

maybe_rewrite_to_crosscor(layer) = layer

function maybe_rewrite_to_crosscor(mode, model)
    mode != "amdgpu" && return model
    return fmap(maybe_rewrite_to_crosscor, model)
end


export BACKEND_GROUP,
    MODES,
    cpu_testing,
    cuda_testing,
    amdgpu_testing,
    get_default_rng,
    StableRNG,
    maybe_rewrite_to_crosscor,
    check_approx,
    allow_unstable

end
