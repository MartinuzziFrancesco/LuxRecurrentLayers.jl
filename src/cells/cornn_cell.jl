#https://arxiv.org/abs/2010.00951
@doc raw"""
    coRNNCell(in_dims => out_dims;
        use_bias=true, train_state=false, train_memory=false,
        init_bias=nothing, init_recurrent_bias=nothing, init_cell_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_cell_weight=nothing, init_state=zeros32, init_memory=zeros32)
        gamma=0.0, epsilon=0.0, dt=1.0)

[Coupled oscillatory recurrent neural unit](https://arxiv.org/abs/2010.00951).

# Equations

```math
\begin{aligned}
    \mathbf{c}(t) &= \mathbf{c}(t-1) + \Delta t \, \sigma\left( 
        \mathbf{W}_{ih} \mathbf{x}(t) + \mathbf{b}_{ih} + 
        \mathbf{W}_{hh} \mathbf{h}(t-1) + \mathbf{b}_{hh} + 
        \mathbf{W}_{ch} \mathbf{c}(t-1) + \mathbf{b}_{ch} \right) 
        - \Delta t \, \gamma \, \mathbf{h}(t-1) - \Delta t \, \epsilon \,
        \mathbf{c}(t), \\
    \mathbf{h}(t) &= \mathbf{h}(t-1) + \Delta t \, \mathbf{c}(t)
\end{aligned}
```

# Arguments

- `in_dims`: Input Dimension
- `out_dims`: Output (Hidden State & Memory) Dimension


## Keyword Arguments

  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.
    Default set to `false`.
  - `train_memory`: Flag to set the initial memory state as trainable.
    Default set to `false`.
  - `init_bias`: Initializer for input to hidden bias
    $\mathbf{b}_{ih}$. If set to `nothing`, weights are initialized from a
    uniform distribution within `[-bound, bound]` where
    `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_recurrent_bias`: Initializer for hidden to hidden bias
    $\mathbf{b}_{hh}$. If set to `nothing`, weights are initialized from a
    uniform distribution within `[-bound, bound]` where
    `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_cell_bias`: Initializer for cell to hidden bias
    $\mathbf{b}_{ch}$. If set to `nothing`, weights are initialized from a
    uniform distribution within `[-bound, bound]` where
    `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_weight`: Initializer for input to hidden weight $\mathbf{W}_{ih}$.
    If set to `nothing`, weights are initialized from a uniform distribution
    within `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
    Default is `nothing`.
  - `init_recurrent_weight`: Initializer for hidden to hidden weight $\mathbf{W}_{hh}$.
    If set to `nothing`, weights are initialized from a uniform distribution
    within `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
    Default is `nothing`.
  - `init_cell_weight`: Initializer for cell to hidden weight $\mathbf{W}_{ch}$.
    If set to `nothing`, weights are initialized from a uniform distribution
    within `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
    Default is `nothing`.
  - `init_state`: Initializer for hidden state. Default set to `zeros32`.
  - `init_memory`: Initializer for memory. Default set to `zeros32`.
  - `dt`: time step. Default is 1.0.
  - `gamma`: Damping for state. Default is 0.0.
  - `epsilon`: Damping for candidate state. Default is 0.0.

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

  - `weight_ih`: Weights to map the input to the hidden state $\mathbf{W}_{ih}$.
  - `weight_hh`: Weights to map the hidden state to the hidden state $\mathbf{W}_{hh}$.
  - `weight_ch`: Weights to map the cell state to the hidden state $\mathbf{W}_{ch}$.
  - `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`)
    $\mathbf{b}_{ih}$
  - `bias_hh`: Bias vector for the hidden-hidden connection (not present if `use_bias=false`)
    $\mathbf{b}_{hh}$
  - `bias_ch`: Bias vector for the cell-hidden connection (not present if `use_bias=false`)
    $\mathbf{b}_{ch}$
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
  - `memory`: Initial memory vector (not present if `train_memory=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct coRNNCell{TS <: StaticBool, TM <: StaticBool} <:
                 AbstractDoubleRecurrentCell{TS, TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_cell_bias
    init_weight
    init_recurrent_weight
    init_cell_weight
    init_state
    use_bias <: StaticBool
    dt
    gamma
    epsilon
end

function coRNNCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_recurrent_bias=nothing, init_cell_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing, init_cell_weight=nothing,
        init_state=zeros32, dt::Number=1.0f0, gamma::Number=0.0f0, epsilon::Number=0.0f0)
    return coRNNCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_recurrent_bias, init_cell_bias, init_weight,
        init_recurrent_weight, init_cell_weight, init_state, static(use_bias),
        dt, gamma, epsilon)
end

function initialparameters(rng::AbstractRNG, cornn::coRNNCell)
    weight_ih = init_rnn_weight(
        rng, cornn.init_weight, cornn.out_dims, (cornn.out_dims, cornn.in_dims))
    weight_hh = init_rnn_weight(
        rng, cornn.init_recurrent_weight, cornn.out_dims, (cornn.out_dims, cornn.out_dims))
    weight_ch = init_rnn_weight(
        rng, cornn.init_cell_weight, cornn.out_dims, (cornn.out_dims, cornn.out_dims))
    ps = (; weight_ih, weight_hh, weight_ch)
    if has_bias(cornn)
        bias_ih = init_rnn_bias(rng, cornn.init_bias, cornn.out_dims, cornn.out_dims)
        bias_hh = init_rnn_bias(rng, cornn.init_recurrent_bias, cornn.out_dims, cornn.out_dims)
        bias_ch = init_rnn_bias(rng, cornn.init_cell_bias, cornn.out_dims, cornn.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh, bias_ch))
    end
    has_train_state(cornn) &&
        (ps = merge(ps, (hidden_state=cornn.init_state(rng, cornn.out_dims),)))
    return ps
end

function parameterlength(cornn::coRNNCell)
    return cornn.in_dims * cornn.out_dims + cornn.out_dims * cornn.out_dims * 2 +
           cornn.out_dims * 3
end

function (cornn::coRNNCell)(
        (inp,
            (state, c_state))::Tuple{
            <:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state, matched_cstate = match_eltype(
        cornn, ps, st, inp, state, c_state)
    dt, gamma, epsilon = cornn.dt, cornn.gamma, cornn.epsilon
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    bias_ch = safe_getproperty(ps, Val(:bias_ch))
    #computation
    xs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    hs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)
    zs = fused_dense_bias_activation(identity, ps.weight_ch, matched_cstate, bias_ch)
    pre_act = @. xs + hs + zs
    new_cstate = @. c_state + dt * (tanh_fast(pre_act) - gamma * state - epsilon * c_state)
    new_state = @. state + dt * new_cstate
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, r::coRNNCell)
    print(io, "coRNNCell($(r.in_dims) => $(r.out_dims)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end
