#https://arxiv.org/pdf/1611.01578
@doc raw"""
    NASCell(in_dims => out_dims;
        use_bias=true, train_state=false, train_memory=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32, init_memory=zeros32)

[Neural Architecture Search unit](https://arxiv.org/pdf/1611.01578).

## Equations
```math
\begin{aligned}
\text{First Layer Outputs:} & \\
o_1 &= \sigma(W_i^{(1)} x_t + W_h^{(1)} h_{t-1} + b^{(1)}), \\
o_2 &= \text{ReLU}(W_i^{(2)} x_t + W_h^{(2)} h_{t-1} + b^{(2)}), \\
o_3 &= \sigma(W_i^{(3)} x_t + W_h^{(3)} h_{t-1} + b^{(3)}), \\
o_4 &= \text{ReLU}(W_i^{(4)} x_t \cdot W_h^{(4)} h_{t-1}), \\
o_5 &= \tanh(W_i^{(5)} x_t + W_h^{(5)} h_{t-1} + b^{(5)}), \\
o_6 &= \sigma(W_i^{(6)} x_t + W_h^{(6)} h_{t-1} + b^{(6)}), \\
o_7 &= \tanh(W_i^{(7)} x_t + W_h^{(7)} h_{t-1} + b^{(7)}), \\
o_8 &= \sigma(W_i^{(8)} x_t + W_h^{(8)} h_{t-1} + b^{(8)}). \\

\text{Second Layer Computations:} & \\
l_1 &= \tanh(o_1 \cdot o_2) \\
l_2 &= \tanh(o_3 + o_4) \\
l_3 &= \tanh(o_5 \cdot o_6) \\
l_4 &= \sigma(o_7 + o_8) \\

\text{Inject Cell State:} & \\
l_1 &= \tanh(l_1 + c_{\text{state}}) \\

\text{Final Layer Computations:} & \\
c_{\text{new}} &= l_1 \cdot l_2 \\
l_5 &= \tanh(l_3 + l_4) \\
h_{\text{new}} &= \tanh(c_{\text{new}} \cdot l_5)
\end{aligned}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension

## Keyword Arguments

  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.
    Default set to `false`.
  - `train_memory`: Flag to set the initial memory state as trainable.
    Default set to `false`.
  - `init_bias`: Initializer for bias. Must be a tuple containing 4 functions. If a single
    value is passed, it is copied into a 4 element tuple. If `nothing`, then we use
    uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(out_dims))`. Default set to `nothing`.
  - `init_recurrent_bias`: Initializer for recurrent bias. Must be a tuple containing 4 functions. If a single
    value is passed, it is copied into a 4 element tuple. If `nothing`, then we use
    uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(out_dims))`. Default set to `nothing`.
  - `init_weight`: Initializer for weight. Must be a tuple containing 4 functions. If a
    single value is passed, it is copied into a 4 element tuple. If `nothing`, then we use
    uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(out_dims))`. Default set to `nothing`.
  - `init_recurrent_weight`: Initializer for recurrent weight. Must be a tuple containing 3 functions. If a
    single value is passed, it is copied into a 3 element tuple. If `nothing`, then we use
    uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(out_dims))`. Default set to `nothing`.
  - `init_state`: Initializer for hidden state. Default set to `zeros32`.
  - `init_memory`: Initializer for memory. Default set to `zeros32`.

## Inputs

  - Case 1a: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `false`, `train_memory` is set to `false` - Creates a hidden state using
             `init_state`, hidden memory using `init_memory` and proceeds to Case 2.
  - Case 1b: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `true`, `train_memory` is set to `false` - Repeats `hidden_state` vector
             from the parameters to match the shape of `x`, creates hidden memory using
             `init_memory` and proceeds to Case 2.
  - Case 1c: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `false`, `train_memory` is set to `true` - Creates a hidden state using
             `init_state`, repeats the memory vector from parameters to match the shape of
             `x` and proceeds to Case 2.
  - Case 1d: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `true`, `train_memory` is set to `true` - Repeats the hidden state and
             memory vectors from the parameters to match the shape of  `x` and proceeds to
             Case 2.
  - Case 2: Tuple `(x, (h, c))` is provided, then the output and a tuple containing the 
            updated hidden state and memory is returned.

## Returns

  - Tuple Containing

      + Output ``h_{new}`` of shape `(out_dims, batch_size)`
      + Tuple containing new hidden state ``h_{new}`` and new memory ``c_{new}``

  - Updated model state

## Parameters

  - `weight_ih`: Concatenated Weights to map from input space
                 ``\{ W_{if}, W_{ic}, W_{ii}, W_{io} \}``.
  - `weight_hh`: Concatenated weights to map from hidden space
                 ``\{ W_{hf}, W_{hc}, W_{hi}, W_{ho} \}``
  - `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`)
  - `bias_hh`: Concatenated Bias vector for the hidden-hidden connection (not present if
    `use_bias=false`)
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
  - `memory`: Initial memory vector (not present if `train_memory=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct NASCell{TS <: StaticBool, TM <: StaticBool} <:
                 AbstractDoubleRecurrentCell{TS, TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_weight
    init_recurrent_weight
    init_state
    init_memory
    use_bias <: StaticBool
end

function NASCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_recurrent_bias=nothing,
        init_state=zeros32, init_memory=zeros32)
    init_weight isa NTuple{8} || (init_weight = ntuple(Returns(init_weight), 8))
    init_recurrent_weight isa NTuple{8} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 8))
    init_bias isa NTuple{8} || (init_bias = ntuple(Returns(init_bias), 8))
    init_recurrent_bias isa NTuple{8} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 8))
    return NASCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_recurrent_bias, init_weight, init_recurrent_weight,
        init_state, init_memory, static(use_bias))
end

function initialparameters(rng::AbstractRNG, nas::NASCell)
    return multi_initialparameters(rng, nas)
end

function parameterlength(nas::NASCell)
    return nas.in_dims * nas.out_dims * 8 + nas.out_dims * nas.out_dims * 8 +
           nas.out_dims * 16
end

function (nas::NASCell)(
        (inp,
            (state, c_state))::Tuple{
            <:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state, matched_cstate = match_eltype(
        nas, ps, st, inp, state, c_state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #gates
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    full_ghs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)
    gates = full_gxs .+ full_ghs
    split_gates = multigate(gates, Val(8))
    #computation
    #first layer
    layer1_1 = sigmoid_fast.(split_gates[1])
    layer1_2 = relu.(split_gates[2])
    layer1_3 = sigmoid_fast.(split_gates[3])
    layer1_4 = relu.(split_gates[4])
    layer1_5 = tanh_fast.(split_gates[5])
    layer1_6 = sigmoid_fast.(split_gates[6])
    layer1_7 = tanh_fast.(split_gates[7])
    layer1_8 = sigmoid_fast.(split_gates[8])

    #second layer
    l2_1 = @. tanh_fast(layer1_1 * layer1_2)
    l2_2 = @. tanh_fast(layer1_3 + layer1_4)
    l2_3 = @. tanh_fast(layer1_5 * layer1_6)
    l2_4 = @. sigmoid_fast(layer1_7 + layer1_8)

    #inject cell
    l2_1 = @. tanh_fast(l2_1 + c_state)

    # Third layer
    new_cstate = l2_1 .* l2_2
    l3_2 = @. tanh_fast(l2_3 + l2_4)

    new_state = @. tanh_fast(new_cstate * l3_2)
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, nas::NASCell)
    print(io, "NASCell($(nas.in_dims) => $(nas.out_dims)")
    has_bias(nas) || print(io, ", use_bias=false")
    has_train_state(nas) && print(io, ", train_state=true")
    known(nas.train_memory) && print(io, ", train_memory=true")
    print(io, ")")
end
