module LuxRecurrentLayers

using Compat: @compat
using ConcreteStructs: @concrete
using LinearAlgebra: transpose, I
using Lux: Utils, init_rnn_hidden_state, init_trainable_rnn_hidden_state, match_eltype,
           safe_getproperty,
           fused_dense_bias_activation, AbstractRecurrentCell, zeros32, has_bias,
           has_train_state, init_rnn_weight,
           init_rnn_bias, replicate, fast_activation!!, multigate, known
import Lux: initialparameters, initialstates, parameterlength, statelength
using NNlib: NNlib, sigmoid_fast, tanh_fast
using Random: AbstractRNG
using Static: StaticBool, StaticInt, StaticSymbol, True, False, static

IntegerType = Utils.IntegerType
BoolType = Utils.BoolType

@compat(public, (initialparameters, initialstates, parameterlength, statelength))

export AntisymmetricRNNCell, ATRCell, CFNCell, coRNNCell, FastGRNNCell, FastRNNCell,
       JANETCell, LEMCell, MinimalRNNCell, PeepholeLSTMCell, SCRNCell, STARCell

include("generics.jl")

include("cells/antisymmetricrnn_cell.jl")
include("cells/atr_cell.jl")
include("cells/cfn_cell.jl")
include("cells/cornn_cell.jl")
include("cells/fastrnn_cell.jl")
include("cells/janet_cell.jl")
include("cells/lem_cell.jl")
include("cells/minimalrnn_cell.jl")
include("cells/peepholelstm_cell.jl")
include("cells/scrn_cell.jl")
include("cells/star_cell.jl")

end
