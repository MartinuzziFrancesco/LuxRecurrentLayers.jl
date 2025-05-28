#https://arxiv.org/abs/2103.05487
@doc raw"""
    UnICORNNCell(in_dims => out_dims;
        use_bias=true, train_state=false, train_memory=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32, init_memory=zeros32,
        dt=1.0, alpha=0.0)

[Undamped independent controlled oscillatory recurrent neural unit](https://arxiv.org/abs/2103.05487).

## Equations
```math
\begin{aligned}
    \mathbf{m}_t &= (\mathbf{W}_{mx} \mathbf{x}_t) \circ (\mathbf{W}_{mh}
        \mathbf{h}_{t-1}), \\
    \hat{\mathbf{h}}_t &= \mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hm}
        \mathbf{m}_t, \\
    \mathbf{i}_t &= \sigma(\mathbf{W}_{ix} \mathbf{x}_t + \mathbf{W}_{im}
        \mathbf{m}_t), \\
    \mathbf{o}_t &= \sigma(\mathbf{W}_{ox} \mathbf{x}_t + \mathbf{W}_{om}
        \mathbf{m}_t), \\
    \mathbf{f}_t &= \sigma(\mathbf{W}_{fx} \mathbf{x}_t + \mathbf{W}_{fm}
        \mathbf{m}_t), \\
    \mathbf{c}_t &= \mathbf{f}_t \circ \mathbf{c}_{t-1} + \mathbf{i}_t \circ
        \tanh(\hat{\mathbf{h}}_t), \\
    \mathbf{h}_t &= \tanh(\mathbf{c}_t) \circ \mathbf{o}_t.
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
                 ``\{ W_{hf}, W_{hc}, W_{hi}, W_{ho} \}```
  - `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`)
  - `bias_hh`: Concatenated Bias vector for the hidden-hidden connection (not present if
    `use_bias=false`)
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
  - `memory`: Initial memory vector (not present if `train_memory=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct UnICORNNCell{TS <: StaticBool, TM <: StaticBool} <:
                 AbstractDoubleRecurrentCell{TS, TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_weight
    init_recurrent_weight
    init_control_weight
    init_state
    init_memory
    use_bias <: StaticBool
    dt
    alpha
end

function UnICORNNCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_control_weight=nothing, init_recurrent_bias=nothing,
        init_state=zeros32, init_memory=zeros32, dt::Number=1.0f0, alpha::Number=0.0f0)
    return UnICORNNCell(static(train_state), static(train_memory), in_dims,
        out_dims, init_bias, init_recurrent_bias,
        init_weight, init_recurrent_weight, init_control_weight,
        init_state, init_memory, static(use_bias), dt, alpha)
end

function initialparameters(rng::AbstractRNG, unicornn::UnICORNNCell)
    # weights
    weight_ih = init_rnn_weight(
        rng, unicornn.init_weight, unicornn.out_dims, (unicornn.out_dims, unicornn.in_dims))
    weight_hh = vec(init_rnn_weight(
        rng, unicornn.init_recurrent_weight, unicornn.out_dims, (unicornn.out_dims, 1)))
    weight_ch = vec(init_rnn_weight(
        rng, unicornn.init_control_weight, unicornn.out_dims, (unicornn.out_dims, 1)))
    ps = (; weight_ih, weight_hh, weight_ch)
    # biases
    if has_bias(unicornn)
        bias_ih = init_rnn_bias(rng, unicornn.init_bias, unicornn.out_dims, unicornn.out_dims)
        ps = merge(ps, (; bias_ih))
    end
    # trainable state and/or memory
    has_train_state(unicornn) &&
        (ps = merge(ps, (hidden_state=unicornn.init_state(rng, unicornn.out_dims),)))
    known(unicornn.train_memory) &&
        (ps = merge(ps, (memory=unicornn.init_memory(rng, unicornn.out_dims),)))
    return ps
end

function parameterlength(unicornn::UnICORNNCell)
    return unicornn.in_dims * unicornn.out_dims + unicornn.out_dims * 3
end

function (unicornn::UnICORNNCell)(
        (inp,
            (state, c_state))::Tuple{
            <:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state, matched_cstate = match_eltype(
        unicornn, ps, st, inp, state, c_state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    #gates
    dt, alpha = unicornn.dt, unicornn.alpha
    new_cstate = matched_cstate .- dt .* sigmoid_fast.(ps.weight_ch) .*
                 (tanh_fast.(ps.weight_hh .* state .+ ps.weight_ih * inp .+ bias_ih) .+ alpha .* matched_state)
    new_state = state .+ dt .* sigmoid_fast.(ps.weight_ch) .* new_cstate
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, unicornn::UnICORNNCell)
    print(io, "UnICORNNCell($(unicornn.in_dims) => $(unicornn.out_dims)")
    has_bias(unicornn) || print(io, ", use_bias=false")
    has_train_state(unicornn) && print(io, ", train_state=true")
    known(unicornn.train_memory) && print(io, ", train_memory=true")
    print(io, ")")
end
