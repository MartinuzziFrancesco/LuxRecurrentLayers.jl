@testsetup module RecurrentLayersSetup

import Reexport: @reexport
@reexport using LuxRecurrentLayers

const RECURRENT_CELLS = [
    (:AntisymmetricRNNCell,
        (; kwargs...) -> AntisymmetricRNNCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:ATRCell,
        (; kwargs...) -> ATRCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:BRCell,
        (; kwargs...) -> BRCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:CFNCell,
        (; kwargs...) -> CFNCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:coRNNCell,
        (; kwargs...) -> coRNNCell(3 => 5; kwargs...),
        [:use_bias, :train_state, :train_memory]),
    (:FastGRNNCell,
        (; kwargs...) -> FastGRNNCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:FastRNNCell,
        (; kwargs...) -> FastRNNCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:GatedAntisymmetricRNNCell,
        (; kwargs...) -> GatedAntisymmetricRNNCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:IndRNNCell,
        (; kwargs...) -> IndRNNCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:JANETCell,
        (; kwargs...) -> JANETCell(3 => 5; kwargs...),
        [:use_bias, :train_state, :train_memory]),
    (:LEMCell,
        (; kwargs...) -> LEMCell(3 => 5; kwargs...),
        [:use_bias, :train_state, :train_memory]),
    (:LightRUCell,
        (; kwargs...) -> LightRUCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:LiGRUCell,
        (; kwargs...) -> LiGRUCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:MGUCell,
        (; kwargs...) -> MGUCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:MinimalRNNCell,
        (; kwargs...) -> MinimalRNNCell(3 => 5; kwargs...),
        [:use_bias, :train_state, :train_memory]),
    (:MultiplicativeLSTMCell,
        (; kwargs...) -> MultiplicativeLSTMCell(3 => 5; kwargs...),
        [:use_bias, :train_state, :train_memory]),
    (:MUT1Cell,
        (; kwargs...) -> MUT1Cell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:MUT2Cell,
        (; kwargs...) -> MUT2Cell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:MUT3Cell,
        (; kwargs...) -> MUT3Cell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:NASCell,
        (; kwargs...) -> NASCell(3 => 5; kwargs...),
        [:use_bias, :train_state, :train_memory]),
    (:NBRCell,
        (; kwargs...) -> NBRCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:PeepholeLSTMCell,
        (; kwargs...) -> PeepholeLSTMCell(3 => 5; kwargs...),
        [:use_bias, :train_state, :train_memory]),
    (:RANCell,
        (; kwargs...) -> RANCell(3 => 5; kwargs...),
        [:use_bias, :train_state, :train_memory]),
    (:SCRNCell,
        (; kwargs...) -> SCRNCell(3 => 5; kwargs...),
        [:use_bias, :train_state, :train_memory]),
    (:SGRNCell,
        (; kwargs...) -> SGRNCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:STARCell,
        (; kwargs...) -> STARCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    #    (:TGRUCell,
    #        (; kwargs...) -> TGRUCell(3 => 5; kwargs...),
    #        [:use_bias, :train_state]),
    #    (:TLSTMCell,
    #        (; kwargs...) -> TLSTMCell(3 => 5; kwargs...),
    #        [:use_bias, :train_state]),
    (:TRNNCell,
        (; kwargs...) -> TRNNCell(3 => 5; kwargs...),
        [:use_bias, :train_state]),
    (:UnICORNNCell,
        (; kwargs...) -> UnICORNNCell(3 => 5; kwargs...),
        [:use_bias, :train_state, :train_memory]),
    (:WMCLSTMCell,
        (; kwargs...) -> WMCLSTMCell(3 => 5; kwargs...),
        [:use_bias, :train_state, :train_memory])
]

format_knobs(kw::AbstractDict) =
    join(["$(String(k))=$(repr(v))"
          for (k, v) in sort!(collect(kw); by=x -> String(x[1]))], ", ")

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

export loss_loop, loss_loop_no_carry, format_knobs, RECURRENT_CELLS

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
