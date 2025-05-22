#https://arxiv.org/pdf/1803.04831
@doc raw"""
    IndRNNCell(in_dims => out_dims, [activation];
        use_bias=true, train_state=false, init_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)
    
[Independently recurrent cell](https://arxiv.org/pdf/1803.04831).

## Equations
```math
\mathbf{h}_{t} = \sigma(\mathbf{W} \mathbf{x}_t + \mathbf{u} \odot \mathbf{h}_{t-1} +
    \mathbf{b})
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension
  - 'activation': Activation funcitons. Defaults to `tanh_fast`

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
@concrete struct IndRNNCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
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

function IndRNNCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh_fast;
        use_bias::BoolType=True(), train_state::BoolType=False(), init_bias=nothing,
        init_recurrent_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)
    return IndRNNCell(static(train_state), in_dims, out_dims, activation, init_bias, init_recurrent_bias,
        init_weight, init_recurrent_weight, init_state, static(use_bias))
end

function initialparameters(rng::AbstractRNG, indrnn::IndRNNCell)
    weight_ih = init_rnn_weight(
        rng, indrnn.init_weight, indrnn.out_dims, (indrnn.out_dims, indrnn.in_dims))
    weight_hh = vec(init_rnn_weight(rng, indrnn.init_recurrent_weight, indrnn.out_dims,
        (indrnn.out_dims, 1)))
    ps = (; weight_ih, weight_hh)
    if has_bias(indrnn)
        bias_ih = init_rnn_bias(rng, indrnn.init_bias, indrnn.out_dims, indrnn.out_dims)
        bias_hh = init_rnn_bias(rng, indrnn.init_recurrent_bias, indrnn.out_dims, indrnn.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    has_train_state(indrnn) &&
        (ps = merge(ps, (hidden_state=indrnn.init_state(rng, indrnn.out_dims),)))
    return ps
end

initialstates(rng::AbstractRNG, ::IndRNNCell) = (rng=Utils.sample_replicate(rng),)

function parameterlength(indrnn::IndRNNCell)
    return indrnn.in_dims * indrnn.out_dims + indrnn.out_dims +
           indrnn.out_dims * 2
end

function (indrnn::IndRNNCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(indrnn, ps, st, inp, state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #computation
    new_state = indrnn.activation.(ps.weight_ih * inp .+ ps.bias_ih .+ ps.weight_hh .* state .+ ps.bias_hh)
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, indrnn::IndRNNCell)
    print(io, "IndRNNCell($(indrnn.in_dims) => $(indrnn.out_dims)")
    has_bias(indrnn) || print(io, ", use_bias=false")
    has_train_state(indrnn) && print(io, ", train_state=true")
    print(io, ")")
end