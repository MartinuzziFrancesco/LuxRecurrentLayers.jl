#https://www.mdpi.com/2079-9292/13/16/3204
@doc raw"""
    LightRUCell(in_dims => out_dims, [activation];
        use_bias=true, train_state=false, init_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)
    
[Light recurrent unit](https://www.mdpi.com/2079-9292/13/16/3204).

## Equations
```math
\begin{aligned}
    \tilde{h}_t &= \tanh(W_h x_t), \\
    f_t         &= \delta(W_f x_t + U_f h_{t-1} + b_f), \\
    h_t         &= (1 - f_t) \odot h_{t-1} + f_t \odot \tilde{h}_t.
\end{aligned}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension

## Keyword Arguments

  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.
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

  -  `weight_ih`: Weights to map from input space
                 ``\{W \}``.
  - `weight_hh`: Weights to map from hidden space
                 ``\{ w_h \}``
  - `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`)
  - `bias_hh`: Bias vector for the hidden-hidden connection (not present if `use_bias=false`)
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct LightRUCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
    train_state::TS
    in_dims <: IntegerType
    out_dims <: IntegerType
    activation
    init_bias
    init_recurrent_bias
    init_weight
    init_recurrent_weight
    init_state
    use_bias <: StaticBool
end

function LightRUCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh_fast;
        use_bias::BoolType=True(), train_state::BoolType=False(), init_bias=nothing,
        init_recurrent_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)
    init_weight isa NTuple{2} || (init_weight = ntuple(Returns(init_weight), 2))
    init_bias isa NTuple{2} || (init_bias = ntuple(Returns(init_bias), 2))
    return LightRUCell(
        static(train_state), in_dims, out_dims, activation, init_bias, init_recurrent_bias,
        init_weight, init_recurrent_weight, init_state, static(use_bias))
end

function initialparameters(rng::AbstractRNG, lightru::LightRUCell)
    weight_ih = multi_inits(
        rng, lightru.init_weight, lightru.out_dims, (lightru.out_dims, lightru.in_dims))
    weight_hh = init_rnn_weight(rng, lightru.init_recurrent_weight, lightru.out_dims,
        (lightru.out_dims, lightru.out_dims))
    ps = (; weight_ih, weight_hh)
    if has_bias(lightru)
        bias_ih = multi_bias(rng, lightru.init_bias, lightru.out_dims, lightru.out_dims)
        bias_hh = init_rnn_bias(
            rng, lightru.init_recurrent_bias, lightru.out_dims, lightru.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    has_train_state(lightru) &&
        (ps = merge(ps, (hidden_state=lightru.init_state(rng, lightru.out_dims),)))
    return ps
end

initialstates(rng::AbstractRNG, ::LightRUCell) = (rng=Utils.sample_replicate(rng),)

function parameterlength(lightru::LightRUCell)
    return lightru.in_dims * lightru.out_dims * 2 + lightru.out_dims * lightru.out_dims +
           lightru.out_dims * 3
end

function (lightru::LightRUCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(lightru, ps, st, inp, state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #computation
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    gxs = multigate(full_gxs, Val(2))
    gh = ps.weight_hh * state .+ bias_hh
    candidate_state = @. tanh_fast(gxs[1])
    forget_gate = sigmoid_fast.(gxs[2] .+ gh)
    new_state = @. (1 - forget_gate) * state + forget_gate * candidate_state
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, lightru::LightRUCell)
    print(io, "LightRUCell($(lightru.in_dims) => $(lightru.out_dims)")
    has_bias(lightru) || print(io, ", use_bias=false")
    has_train_state(lightru) && print(io, ", train_state=true")
    print(io, ")")
end
