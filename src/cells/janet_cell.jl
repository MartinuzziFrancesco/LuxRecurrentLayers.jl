#https://arxiv.org/abs/1804.04849
@doc raw"""
    JANETCell(in_dims => out_dims;
        use_bias=true, train_state=false, train_memory=false,
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32, init_memory=zeros32, beta=1.0)

[Just another network unit](https://arxiv.org/abs/1804.04849).

## Equations
```math
\begin{aligned}
    \mathbf{s}_t &= \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{W}_f \mathbf{x}_t + \mathbf{b}_f \\
    \tilde{\mathbf{c}}_t &= \tanh (\mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{W}_c \mathbf{x}_t + \mathbf{b}_c) \\
    \mathbf{c}_t &= \sigma(\mathbf{s}_t) \odot \mathbf{c}_{t-1} + (1 - \sigma (\mathbf{s}_t - \beta)) \odot \tilde{\mathbf{c}}_t \\
    \mathbf{h}_t &= \mathbf{c}_t.
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
  - `init_bias`: Initializer for bias. Must be a tuple containing 2 functions. If a single
    value is passed, it is copied into a 2 element tuple. If `nothing`, then we use
    uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(out_dims))`. Default set to `nothing`.
  - `init_weight`: Initializer for weight. Must be a tuple containing 2 functions. If a
    single value is passed, it is copied into a 2 element tuple. If `nothing`, then we use
    uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(out_dims))`. Default set to `nothing`.
  - `init_recurrent_weight`: Initializer for recurrent weight. Must be a tuple containing 2 functions. If a
    single value is passed, it is copied into a 2 element tuple. If `nothing`, then we use
    uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(out_dims))`. Default set to `nothing`.
  - `init_state`: Initializer for hidden state. Default set to `zeros32`.
  - `init_memory`: Initializer for memory. Default set to `zeros32`.
  - `beta`: Control parameter over the input data flow.
    Default is 1.0.

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
                 ``\{ W_{if}, W_{ic} \}``.
  - `weight_hh`: Concatenated Weights to map from hidden space
                 ``\{ W_{hf}, W_{hc} \}``
  - `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`)
  - `bias_hh`: Concatenated Bias vector for the hidden-hidden connection (not present if
    `use_bias=false`)
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
  - `memory`: Initial memory vector (not present if `train_memory=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct JANETCell{TS <: StaticBool, TM <: StaticBool} <:
                 AbstractDoubleRecurrentCell{TS, TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_weight
    init_recurrent_weight
    init_state
    init_memory
    use_bias <: StaticBool
    beta
end

function JANETCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing, init_state=zeros32,
        init_memory=zeros32, beta::Number=1.0f0)
    init_weight isa NTuple{2} || (init_weight = ntuple(Returns(init_weight), 2))
    init_recurrent_weight isa NTuple{2} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 2))
    init_bias isa NTuple{2} || (init_bias = ntuple(Returns(init_bias), 2))
    return JANETCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_weight, init_recurrent_weight, init_state, init_memory, static(use_bias),
        beta)
end

function initialparameters(rng::AbstractRNG, janet::JANETCell)
    # weights
    weight_ih = multi_inits(
        rng, janet.init_weight, janet.out_dims, (janet.out_dims, janet.in_dims))
    weight_hh = multi_inits(
        rng, janet.init_recurrent_weight, janet.out_dims, (janet.out_dims, janet.out_dims))
    ps = (; weight_ih, weight_hh)
    # biases
    if has_bias(janet)
        bias_ih = multi_bias(rng, janet.init_bias, janet.out_dims, janet.out_dims)
        bias_hh = multi_bias(rng, janet.init_bias, janet.out_dims, janet.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    # trainable state and/or memory
    has_train_state(janet) &&
        (ps = merge(ps, (hidden_state=janet.init_state(rng, janet.out_dims),)))
    known(janet.train_memory) &&
        (ps = merge(ps, (memory=janet.init_memory(rng, janet.out_dims),)))
    return ps
end

function (janet::JANETCell)(
        (inp,
            (state, c_state))::Tuple{
            <:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state, matched_cstate = match_eltype(
        janet, ps, st, inp, state, c_state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #gates
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    full_ghs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)
    gxs = multigate(full_gxs, Val(2))
    ghs = multigate(full_ghs, Val(2))
    #computation
    linear_gate = gxs[1] .+ ghs[1]
    candidate_state = @. tanh_fast(gxs[2] + ghs[2])
    ones_vec = eltype(candidate_state)(1.0)
    new_cstate = @. sigmoid_fast(linear_gate) * c_state +
                    (ones_vec - sigmoid_fast(linear_gate - janet.beta)) *
                    candidate_state
    new_state = new_cstate
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, janet::JANETCell)
    print(io, "LSTMCell($(janet.in_dims) => $(janet.out_dims)")
    has_bias(janet) || print(io, ", use_bias=false")
    has_train_state(janet) && print(io, ", train_state=true")
    known(janet.train_memory) && print(io, ", train_memory=true")
    return print(io, ")")
end
