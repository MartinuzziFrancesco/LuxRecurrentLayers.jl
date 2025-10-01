module LuxRecurrentLayers

using Compat: @compat
using ConcreteStructs: @concrete
using LinearAlgebra: transpose, I
using Lux: Utils, init_rnn_hidden_state, init_trainable_rnn_hidden_state, match_eltype,
           safe_getproperty, bias_activation, fused_dense_bias_activation,
           AbstractRecurrentCell, zeros32, has_bias,
           has_train_state, init_rnn_weight,
           init_rnn_bias, replicate, fast_activation!!, known
import Lux: initialparameters, initialstates, parameterlength, statelength, multigate
using NNlib: NNlib, sigmoid_fast, tanh_fast, relu
using Random: AbstractRNG
using Static: StaticBool, StaticInt, StaticSymbol, True, False, static, Static

IntegerType = Utils.IntegerType
BoolType = Utils.BoolType

@compat(public, (initialparameters, initialstates, parameterlength, statelength))

export AntisymmetricRNNCell, ATRCell, BRCell, CFNCell, coRNNCell, FastGRNNCell,
       FastRNNCell, GatedAntisymmetricRNNCell, IndRNNCell, JANETCell, LEMCell, LightRUCell,
       LiGRUCell, MGUCell, MinimalRNNCell, MultiplicativeLSTMCell, MUT1Cell, MUT2Cell,
       MUT3Cell, NASCell, NBRCell, PeepholeLSTMCell, RANCell, SCRNCell, SGRNCell,
       STARCell, TGRUCell, TLSTMCell, TRNNCell, UnICORNNCell, WMCLSTMCell

include("generics.jl")

include("cells/antisymmetricrnn_cell.jl")
include("cells/atr_cell.jl")
include("cells/br_cell.jl")
include("cells/cfn_cell.jl")
include("cells/cornn_cell.jl")
include("cells/fastrnn_cell.jl")
include("cells/indrnn_cell.jl")
include("cells/janet_cell.jl")
include("cells/lem_cell.jl")
include("cells/lightru_cell.jl")
include("cells/ligru_cell.jl")
include("cells/mgu_cell.jl")
include("cells/minimalrnn_cell.jl")
include("cells/multiplicativelstm_cell.jl")
include("cells/mut_cell.jl")
include("cells/nas_cell.jl")
include("cells/peepholelstm_cell.jl")
include("cells/ran_cell.jl")
include("cells/scrn_cell.jl")
include("cells/sgrn_cell.jl")
include("cells/star_cell.jl")
include("cells/trnn_cell.jl")
include("cells/unicornn_cell.jl")
include("cells/wmclstm_cell.jl")

end
