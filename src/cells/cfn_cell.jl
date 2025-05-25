#https://arxiv.org/abs/1612.06212
@doc raw"""
    CFNCell(in_dims => out_dims, [activation];
        use_bias=true, train_state=false, init_bias=nothing,
        init_recurrent_bias=nothing, init_weight=nothing,
        init_recurrent_weight=nothing, init_state=zeros32,
        epsilon=1.0, gamma=0.0)


[Chaos free network unit](https://arxiv.org/abs/1612.06212).

## Equations

```math
\begin{aligned}
    h_t &= \theta_t \odot \tanh(h_{t-1}) + \eta_t \odot \tanh(W x_t), \\
    \theta_t &:= \sigma (U_\theta h_{t-1} + V_\theta x_t + b_\theta), \\
    \eta_t &:= \sigma (U_\eta h_{t-1} + V_\eta x_t + b_\eta).
\end{aligned}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension
  - `activation`: activation function. Default is `tanh`

# Keyword arguments


  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.
    Default set to `false`.
  - `train_memory`: Flag to set the initial memory state as trainable.
    Default set to `false`.
  - `init_bias`: Initializer for bias. Must be a tuple containing 2 functions. If a single
    value is passed, it is copied into a 2 element tuple. If `nothing`, then we use
    uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(out_dims))`. Default set to `nothing`.
  - `init_recurrent_bias`: Initializer for recurrent bias. Must be a tuple containing 2 functions.
    If a single value is passed, it is copied into a 2 element tuple. If `nothing`, then we use
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
             to `false` - Creates a hidden state using `init_state` and proceeds to Case 2.
  - Case 1b: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `true` - Repeats `hidden_state` from parameters to match the shape of `x`
             and proceeds to Case 2.
  - Case 2: Tuple `(x, (h, ))` is provided, then the output and a tuple containing the
            updated hidden state is returned.


## Returns

  - Tuple containing

      + Output ``h_{new}`` of shape `(out_dims, batch_size)`
      + Tuple containing new hidden state ``h_{new}``

  - Updated model state

## Parameters

  -  `weight_ih`: Concatenated Weights to map from input space
                 ``\{ W, W_{\theta},  W_{\eta} \}``.
  - `weight_hh`: Concatenated Weights to map from hidden space
                 ``\{ W_{\theta}, W_{\eta} \}``
  - `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`)
  - `bias_hh`: Bias vector for the hidden-hidden connection (not present if `use_bias=false`)
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct CFNCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
    train_state::TS
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_weight
    init_recurrent_weight
    init_state
    use_bias <: StaticBool
end

function CFNCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh;
        use_bias::BoolType=True(), train_state::BoolType=False(),
        init_bias=nothing, init_recurrent_bias=nothing, init_weight=nothing,
        init_recurrent_weight=nothing, init_state=zeros32)
    init_weight isa NTuple{3} || (init_weight = ntuple(Returns(init_weight), 3))
    init_recurrent_weight isa NTuple{2} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 2))
    init_bias isa NTuple{3} || (init_bias = ntuple(Returns(init_bias), 3))
    init_recurrent_bias isa NTuple{2} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 2))
    return CFNCell(static(train_state), activation, in_dims, out_dims,
        init_bias, init_recurrent_bias, init_weight,
        init_recurrent_weight, init_state, static(use_bias))
end

initialparameters(rng::AbstractRNG, cfn::CFNCell) = multi_initialparameters(rng, cfn)

initialstates(rng::AbstractRNG, ::CFNCell) = (rng=Utils.sample_replicate(rng),)

function parameterlength(cfn::CFNCell)
    return cfn.in_dims * cfn.out_dims * 3 + cfn.out_dims * cfn.out_dims * 2 +
           cfn.out_dims * 5
end

statelength(::CFNCell) = 1

function (cfn::CFNCell{False})(inp::AbstractMatrix, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = init_rnn_hidden_state(rng, cfn, inp)
    return cfn((inp, (hidden_state,)), ps, merge(st, (; rng)))
end

function (cfn::CFNCell{True})(inp::AbstractMatrix, ps, st::NamedTuple)
    hidden_state = init_trainable_rnn_hidden_state(ps.hidden_state, inp)
    return cfn((inp, (hidden_state,)), ps, st)
end

function (cfn::CFNCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(cfn, ps, st, inp, state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #gates
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    full_ghs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)
    gxs = multigate(full_gxs, Val(3))
    ghs = multigate(full_ghs, Val(2))
    #computation
    horizontal_gate = @. sigmoid_fast(gxs[1] + ghs[1])
    vertical_gate = @. sigmoid_fast(gxs[2] + ghs[2])
    new_state = @. horizontal_gate * tanh_fast(state) + vertical_gate * tanh_fast(gxs[3])

    return (new_state, (new_state,)), st
end

function Base.show(io::IO, r::CFNCell)
    print(io, "CFNCell($(r.in_dims) => $(r.out_dims)")
    (r.activation == identity) || print(io, ", $(r.activation)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end
