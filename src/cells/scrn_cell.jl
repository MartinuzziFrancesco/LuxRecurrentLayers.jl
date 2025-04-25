#https://arxiv.org/pdf/1412.7753

@doc raw"""
    SCRNCell(in_dims => out_dims;
        use_bias=true, train_state=false, train_memory=false,
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_context_weight=nothing, init_state=zeros32, init_memory=zeros32)

[Structurally contraint recurrent unit](https://arxiv.org/pdf/1412.7753).

## Equations
```math
\begin{aligned}
s_t &= (1 - \alpha) W_s x_t + \alpha s_{t-1}, \\
h_t &= \sigma(W_h s_t + U_h h_{t-1} + b_h), \\
y_t &= f(U_y h_t + W_y s_t)
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
  - `init_context_weight`: Initializer for context weight. Must be a tuple containing 2 functions. If a
    single value is passed, it is copied into a 2 element tuple. If `nothing`, then we use
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
                 ``\{ W_{if}, W_{ic} \}``.
  - `weight_hh`: Concatenated Weights to map from hidden space
                 ``\{ W_{hf}, W_{hc} \}``
  - `weight_hh`: Concatenated Weights to map from context space
                 ``\{ W_{cf}, W_{cc} \}``
  - `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`)
  - `bias_hh`: Concatenated Bias vector for the hidden-hidden connection (not present if
    `use_bias=false`)
  - `alpha`: Initial context strength.
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
  - `memory`: Initial memory vector (not present if `train_memory=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct SCRNCell{TS <: StaticBool, TM <: StaticBool} <:
                 AbstractDoubleRecurrentCell{TS, TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_weight
    init_recurrent_weight
    init_context_weight
    init_state
    init_memory
    use_bias <: StaticBool
end

function SCRNCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_context_weight=nothing, init_state=zeros32, init_memory=zeros32)
    init_weight isa NTuple{2} || (init_weight = ntuple(Returns(init_weight), 2))
    init_recurrent_weight isa NTuple{2} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 2))
    init_context_weight isa NTuple{2} ||
        (init_context_weight = ntuple(Returns(init_context_weight), 2))
    init_bias isa NTuple{2} || (init_bias = ntuple(Returns(init_bias), 2))
    return SCRNCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_weight, init_recurrent_weight,
        init_context_weight, init_state, init_memory, static(use_bias))
end

function initialparameters(rng::AbstractRNG, scrn::SCRNCell)
    # weights
    weight_ih = multi_inits(
        rng, scrn.init_weight, scrn.out_dims, (scrn.out_dims, scrn.in_dims))
    weight_hh = multi_inits(
        rng, scrn.init_recurrent_weight, scrn.out_dims, (scrn.out_dims, scrn.out_dims))
    weight_ch = multi_inits(
        rng, scrn.init_context_weight, scrn.out_dims, (scrn.out_dims, scrn.out_dims))
    ps = (; weight_ih, weight_hh, weight_ch)
    # biases
    if has_bias(scrn)
        bias_ih = multi_bias(rng, scrn.init_bias, scrn.out_dims, scrn.out_dims)
        bias_hh = multi_bias(rng, scrn.init_bias, scrn.out_dims, scrn.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    # trainable state and/or memory
    has_train_state(scrn) &&
        (ps = merge(ps, (hidden_state=scrn.init_state(rng, scrn.out_dims),)))
    known(scrn.train_memory) &&
        (ps = merge(ps, (memory=scrn.init_memory(rng, scrn.out_dims),)))
    # any additional trainable parameters
    ps = merge(ps, (alpha=eltype(weight_ih)(0.0f0),))
    return ps
end

function parameterlength(scrn::SCRNCell)
    return scrn.in_dims * scrn.out_dims * 2 + scrn.out_dims * scrn.out_dims * 4 +
           scrn.out_dims * 2 + 1
end

function (scrn::SCRNCell)(
        (inp,
            (state, c_state))::Tuple{
            <:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state, matched_cstate = match_eltype(
        scrn, ps, st, inp, state, c_state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #gates
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    full_gcs = fused_dense_bias_activation(identity, ps.weight_ch, matched_state, bias_hh)
    gxs = multigate(full_gxs, Val(2))
    ghs = multigate(ps.weight_hh, Val(2))
    gcs = multigate(full_gcs, Val(2))
    #computation
    new_cstate = (eltype(ps.weight_hh)(1.0f0) .- ps.alpha) .* gxs[1] .+
                 ps.alpha .* matched_cstate
    hidden_layer = sigmoid_fast.(gxs[2] .+ ghs[1] * matched_state .+ gcs[1])
    new_state = tanh_fast.(ghs[2] * hidden_layer .+ gcs[2])
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, r::SCRNCell)
    print(io, "SCRNCell($(r.in_dims) => $(r.out_dims)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end
