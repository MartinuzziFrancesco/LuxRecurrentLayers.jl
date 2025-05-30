#https://arxiv.org/abs/1902.09689
@doc raw"""
    AntisymmetricRNNCell(in_dims => out_dims, [activation];
        use_bias=true, train_state=false, init_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32, epsilon=1.0, gamma=0.0)


[Antisymmetric recurrent cell](https://arxiv.org/abs/1902.09689).

## Equations
```math
h_t = h_{t-1} + \epsilon \tanh \left( (W_h - W_h^T - \gamma I) h_{t-1} + V_h x_t + b_h \right),
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
  - `epsilon`: step size. Default is 1.0.
  - `gamma`: strength of diffusion. Default is 0.0.

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

  - `weight_ih`: Maps the input to the hidden state.
  - `weight_hh`: Maps the hidden state to the hidden state.
  - `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`)
  - `bias_hh`: Bias vector for the hidden-hidden connection (not present if `use_bias=false`)
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct AntisymmetricRNNCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
    train_state::TS
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_weight
    init_recurrent_weight
    init_state
    use_bias <: StaticBool
    epsilon
    gamma
end

function AntisymmetricRNNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh;
        use_bias::BoolType=True(), train_state::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing, init_state=zeros32,
        epsilon=1.0f0, gamma=0.0f0)
    return AntisymmetricRNNCell(static(train_state), activation, in_dims, out_dims,
        init_bias, init_weight, init_recurrent_weight, init_state, static(use_bias), epsilon, gamma)
end

function initialparameters(rng::AbstractRNG, asymrnn::AntisymmetricRNNCell)
    return single_initialparameters(rng, asymrnn)
end

function parameterlength(asymrnn::AntisymmetricRNNCell)
    return asymrnn.in_dims * asymrnn.out_dims + asymrnn.out_dims * asymrnn.out_dims +
           asymrnn.out_dims
end

function (asymrnn::AntisymmetricRNNCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(asymrnn, ps, st, inp, state)
    #input linear transform
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    linear_input = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #recurrent linear transform
    asym_weight_hh = compute_asym_recurrent(ps.weight_hh, asymrnn.gamma)
    linear_recur = fused_dense_bias_activation(
        identity, asym_weight_hh, matched_state, bias_hh)
    #putting it all together
    half_new_state = fast_activation!!(asymrnn.activation, linear_input .+ linear_recur)
    new_state = matched_state .+ asymrnn.epsilon .* half_new_state
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, r::AntisymmetricRNNCell)
    print(io, "AntisymmetricRNNCell($(r.in_dims) => $(r.out_dims)")
    (r.activation == identity) || print(io, ", $(r.activation)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end

@doc raw"""
    GatedAntisymmetricRNNCell(in_dims => out_dims, [activation];
        use_bias=true, train_state=false, init_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32, epsilon=1.0, gamma=0.0)



[Antisymmetric recurrent cell with gating](https://arxiv.org/abs/1902.09689).

## Equations
```math
    z_t &= \sigma ( (W_h - W_h^T - \gamma I) h_{t-1} + V_z x_t + b_z ), \\
    h_t &= h_{t-1} + \epsilon z_t \odot \tanh ( (W_h - W_h^T - \gamma I) h_{t-1}
        + V_h x_t + b_h ).
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
  - `epsilon`: step size. Default is 1.0.
  - `gamma`: strength of diffusion. Default is 0.0.

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

  - `weight_ih`: Maps the input to the hidden state.
  - `weight_hh`: Maps the hidden state to the hidden state.
  - `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`)
  - `bias_hh`: Bias vector for the hidden-hidden connection (not present if `use_bias=false`)
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct GatedAntisymmetricRNNCell{TS <: StaticBool} <:
                 AbstractSingleRecurrentCell{TS}
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
    epsilon
    gamma
end

function GatedAntisymmetricRNNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh;
        use_bias::BoolType=True(), train_state::BoolType=False(),
        init_bias=nothing, init_recurrent_bias=nothing, init_weight=nothing,
        init_recurrent_weight=nothing, init_state=zeros32,
        epsilon=1.0f0, gamma=0.0f0)
    init_weight isa NTuple{2} || (init_weight = ntuple(Returns(init_weight), 2))
    init_bias isa NTuple{2} || (init_bias = ntuple(Returns(init_bias), 2))
    return GatedAntisymmetricRNNCell(static(train_state), activation, in_dims, out_dims,
        init_bias, init_recurrent_bias, init_weight, init_recurrent_weight, init_state,
        static(use_bias), epsilon, gamma)
end

function initialparameters(rng::AbstractRNG, asymrnn::GatedAntisymmetricRNNCell)
    weight_ih = multi_inits(
        rng, asymrnn.init_weight, asymrnn.out_dims, (asymrnn.out_dims, asymrnn.in_dims))
    weight_hh = init_rnn_weight(rng, asymrnn.init_recurrent_weight, asymrnn.out_dims,
        (asymrnn.out_dims, asymrnn.out_dims))
    ps = (; weight_ih, weight_hh)
    if has_bias(asymrnn)
        bias_ih = multi_bias(rng, asymrnn.init_bias, asymrnn.out_dims, asymrnn.out_dims)
        bias_hh = init_rnn_bias(
            rng, asymrnn.init_recurrent_bias, asymrnn.out_dims, asymrnn.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    has_train_state(asymrnn) &&
        (ps = merge(ps, (hidden_state=asymrnn.init_state(rng, asymrnn.out_dims),)))
    return ps
end

function parameterlength(asymrnn::GatedAntisymmetricRNNCell)
    return asymrnn.in_dims * asymrnn.out_dims * 2 + asymrnn.out_dims * asymrnn.out_dims +
           asymrnn.out_dims * 2
end

function (asymrnn::GatedAntisymmetricRNNCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    matched_inp, matched_state = match_eltype(asymrnn, ps, st, inp, state)
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    gxs = multigate(full_gxs, Val(2))
    asym_weight_hh = compute_asym_recurrent(ps.weight_hh, asymrnn.gamma)
    hs = fused_dense_bias_activation(identity, asym_weight_hh, matched_state, bias_hh)
    input_gate = @. sigmoid_fast(hs + gxs[1])
    half_new_state = @. tanh_fast(hs + gxs[2])
    new_state = @. matched_state .+ asymrnn.epsilon .* input_gate
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, r::GatedAntisymmetricRNNCell)
    print(io, "GatedAntisymmetricRNNCell($(r.in_dims) => $(r.out_dims)")
    (r.activation == identity) || print(io, ", $(r.activation)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end

function compute_asym_recurrent(weight_hh, gamma)
    gamma = eltype(weight_hh)(gamma)
    return weight_hh .- transpose(weight_hh) .-
           gamma .* Matrix{eltype(weight_hh)}(I, size(weight_hh, 1), size(weight_hh, 1))
end
