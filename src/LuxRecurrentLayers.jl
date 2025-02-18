module LuxRecurrentLayers

using Compat: @compat
using Lux: Utils, init_rnn_hidden_state, init_trainable_rnn_hidden_state, match_eltype, safe_getproperty,
fused_dense_bias_activation
using NNlib: NNlib
using Static: StaticBool, StaticInt, StaticSymbol, True, False, static

IntegerType = Utils.IntegerType

export AntisymmetricRNNCell
export AntisymmetricRNN

include("cells/antisymmetricrnn_cell.jl")

end
