#https://arxiv.org/pdf/1705.07393
@doc raw"""
    RANCell(in_dims => out_dims;
        use_bias=true, train_state=false, train_memory=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32, init_memory=zeros32)

[Recurrent Additive Network cell](https://arxiv.org/pdf/1705.07393).

## Equations
```math
\begin{aligned}
\tilde{c}_t &= W_c x_t, \\
i_t         &= \sigma(W_i x_t + U_i h_{t-1} + b_i), \\
f_t         &= \sigma(W_f x_t + U_f h_{t-1} + b_f), \\
c_t         &= i_t \odot \tilde{c}_t + f_t \odot c_{t-1}, \\
h_t         &= g(c_t)
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
@concrete struct RANCell{TS <: StaticBool, TM <: StaticBool} <:
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

function RANCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_recurrent_bias=nothing,
        init_state=zeros32, init_memory=zeros32)
    init_weight isa NTuple{3} || (init_weight = ntuple(Returns(init_weight), 3))
    init_recurrent_weight isa NTuple{2} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 2))
    init_bias isa NTuple{3} || (init_bias = ntuple(Returns(init_bias), 3))
    init_recurrent_bias isa NTuple{2} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 2))
    return RANCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_recurrent_bias, init_weight, init_recurrent_weight,
        init_state, init_memory, static(use_bias))
end

function initialparameters(rng::AbstractRNG, ran::RANCell)
    return multi_initialparameters(rng, ran)
end

function parameterlength(ran::RANCell)
    return ran.in_dims * ran.out_dims * 3 + ran.out_dims * ran.out_dims * 2 +
           ran.out_dims * 5
end

function (ran::RANCell)(
        (inp,
            (state, c_state))::Tuple{
            <:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state, matched_cstate = match_eltype(
        ran, ps, st, inp, state, c_state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #gates
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    full_ghs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)
    gxs = multigate(full_gxs, Val(3))
    ghs = multigate(full_ghs, Val(2))
    #computation
    input_gate = @. sigmoid_fast(gxs[2] + ghs[1])
    forget_gate = @. sigmoid_fast(gxs[3] + ghs[2])
    new_cstate = @. input_gate * gxs[1] + forget_gate * matched_cstate
    new_state = @. tanh_fast(new_cstate)
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, ran::RANCell)
    print(io, "RANCell($(ran.in_dims) => $(ran.out_dims)")
    has_bias(ran) || print(io, ", use_bias=false")
    has_train_state(ran) && print(io, ", train_state=true")
    known(ran.train_memory) && print(io, ", train_memory=true")
    print(io, ")")
end
