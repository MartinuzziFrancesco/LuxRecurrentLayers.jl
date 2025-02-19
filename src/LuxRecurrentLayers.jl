module LuxRecurrentLayers

using Compat: @compat
using ConcreteStructs: @concrete
using LinearAlgebra: transpose, I
using Lux: Utils, init_rnn_hidden_state, init_trainable_rnn_hidden_state, match_eltype, safe_getproperty,
fused_dense_bias_activation, AbstractRecurrentCell, zeros32, has_bias, has_train_state, init_rnn_weight,
init_rnn_bias, replicate, fast_activation!!
import Lux: initialparameters, initialstates, parameterlength, statelength
using NNlib: NNlib
using Random: AbstractRNG
using Static: StaticBool, StaticInt, StaticSymbol, True, False, static

IntegerType = Utils.IntegerType
BoolType = Utils.BoolType

@compat(public, (initialparameters, initialstates, parameterlength, statelength))

export AntisymmetricRNNCell
#export AntisymmetricRNN

include("cells/antisymmetricrnn_cell.jl")

end
