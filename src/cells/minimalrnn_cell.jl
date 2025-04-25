#https://arxiv.org/abs/1711.06788
@doc raw"""
    MinimalRNNCell(in_dims => out_dims;
        use_bias=true, train_state=false, init_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32,)

[Minimal recurrent neural network unit](https://arxiv.org/abs/1711.06788).

# Equations

```math
\begin{aligned}
    \mathbf{z}_t &= \Phi(\mathbf{x}_t) = \tanh(\mathbf{W}_x \mathbf{x}_t +
        \mathbf{b}_z), \\
    \mathbf{u}_t &= \sigma(\mathbf{U}_h \mathbf{h}_{t-1} + \mathbf{U}_z \mathbf{z}_t +
        \mathbf{b}_u), \\
    \mathbf{h}_t &= \mathbf{u}_t \circ \mathbf{h}_{t-1} + (1 - \mathbf{u}_t) \circ
        \mathbf{z}_t.
\end{aligned}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension

## Keyword arguments

  - `use_bias`: Set to false to deactivate bias
  - `train_state`: Trainable initial hidden state can be activated by setting this to `true`
  - `train_memory`: Trainable initial memory can be activated by setting this to `true`
  - `init_bias`: Initializer for bias. Must be a tuple containing 4 functions. If a single
    value is passed, it is copied into a 4 element tuple. If `nothing`, then we use
    uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(out_dims))`.
  - `init_weight`: Initializer for weight. Must be a tuple containing 4 functions. If a
    single value is passed, it is copied into a 4 element tuple. If `nothing`, then we use
    uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(out_dims))`.
  - `init_recurrent_weight`: Initializer for recurrent weight. Must be a tuple containing 4 functions. If a
    single value is passed, it is copied into a 4 element tuple. If `nothing`, then we use
    uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(out_dims))`.
  - `init_state`: Initializer for hidden state
  - `init_memory`: Initializer for memory

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

  - `weight_ih`: Weights to map from input space.
  - `weight_hh`: Weights to map from hidden space.
  - `weight_mm`: Weights to map from memory space.
  - `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`).
  - `bias_hh`: Bias vector for the hidden-hidden connection (not present if
    `use_bias=false`).
  - `bias_hh`: Bias vector for the mnemrory-memory connection (not present if
    `use_bias=false`).
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`).
  - `memory`: Initial memory vector (not present if `train_memory=false`).

## States

  - `rng`: Controls the randomness (if any) in the initial state generation
"""
@concrete struct MinimalRNNCell{TS <: StaticBool, TM <: StaticBool} <:
                 AbstractDoubleRecurrentCell{TS, TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_encoder_bias
    init_weight
    init_recurrent_weight
    init_memory_weight
    init_state
    init_memory
    use_bias <: StaticBool
end

function MinimalRNNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh;
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_encoder_bias=nothing, init_weight=nothing,
        init_recurrent_weight=nothing, init_memory_weight=nothing, init_state=zeros32,
        init_memory=zeros32)
    return MinimalRNNCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_encoder_bias, init_weight, init_recurrent_weight,
        init_memory_weight, init_state, init_memory, static(use_bias))
end

function initialparameters(rng::AbstractRNG, minimal::MinimalRNNCell)
    weight_ih = init_rnn_weight(rng, minimal.init_weight, minimal.out_dims,
        (minimal.out_dims, minimal.in_dims))
    weight_hh = init_rnn_weight(rng, minimal.init_recurrent_weight, minimal.out_dims,
        (minimal.out_dims, minimal.out_dims))
    weight_mm = init_rnn_weight(rng, minimal.init_memory_weight, minimal.out_dims,
        (minimal.out_dims, minimal.out_dims))
    ps = (; weight_ih, weight_hh, weight_mm)
    if has_bias(minimal)
        bias_ih = init_rnn_bias(
            rng, minimal.init_encoder_bias, minimal.out_dims, minimal.out_dims)
        bias_hh = init_rnn_bias(rng, minimal.init_bias, minimal.out_dims, minimal.out_dims)
        bias_mm = init_rnn_bias(rng, minimal.init_bias, minimal.out_dims, minimal.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh, bias_mm))
    end
    has_train_state(minimal) &&
        (ps = merge(ps, (hidden_state=minimal.init_state(rng, minimal.out_dims),)))
    known(minimal.train_memory) &&
        (ps = merge(ps, (hidden_state=minimal.init_memory(rng, minimal.out_dims),)))
    return ps
end

initialstates(rng::AbstractRNG, ::MinimalRNNCell) = (rng=Utils.sample_replicate(rng),)

function parameterlength(minimal::MinimalRNNCell)
    return minimal.in_dims * minimal.out_dims + minimal.out_dims * minimal.out_dims * 2 +
           minimal.out_dims * 3
end

statelength(::MinimalRNNCell) = 1

function (minimal::MinimalRNNCell)(
        (inp,
            (state, c_state))::Tuple{
            <:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state, matched_memory = match_eltype(
        minimal, ps, st, inp, state, c_state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    bias_mm = safe_getproperty(ps, Val(:bias_mm))
    #computation
    xs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    hs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)
    ms = fused_dense_bias_activation(identity, ps.weight_mm, matched_memory, bias_mm)

    new_cstate = tanh_fast.(xs)
    update_gate = @. sigmoid_fast(hs + ms)
    new_state = update_gate .* state .+
                (eltype(ps.weight_ih)(1.0) .- update_gate) .* new_cstate
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, minimal::MinimalRNNCell)
    print(io, "MinimalRNNCell($(minimal.in_dims) => $(minimal.out_dims)")
    has_bias(minimal) || print(io, ", use_bias=false")
    has_train_state(minimal) && print(io, ", train_state=true")
    known(minimal.train_memory) && print(io, ", train_memory=true")
    return print(io, ")")
end
