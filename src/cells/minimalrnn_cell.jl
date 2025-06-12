#https://arxiv.org/abs/1711.06788
@doc raw"""
    MinimalRNNCell(in_dims => out_dims;
        use_bias=true, train_state=false,
        init_encoder_bias=nothing, init_recurrent_bias=nothing,
        init_memory_bias=nothing, init_encoder_weight=nothing,
        init_recurrent_weight=nothing, init_memory_weight=nothing,
        init_state=zeros32,)

[Minimal recurrent neural network unit](https://arxiv.org/abs/1711.06788).

# Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \tanh\left( \mathbf{W}_{ih}^{z} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{z} \right), \\
    \mathbf{u}(t) &= \sigma\left( \mathbf{W}_{hh}^{u} \mathbf{h}(t-1) +
        \mathbf{W}_{zh}^{u} \mathbf{z}(t) + \mathbf{b}_{hh}^{u} \right), \\
    \mathbf{h}(t) &= \mathbf{u}(t) \circ \mathbf{h}(t-1) + \left(1 -
        \mathbf{u}(t)\right) \circ \mathbf{z}(t)
\end{aligned}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension

## Keyword arguments

  - `use_bias`: Set to false to deactivate bias
  - `train_state`: Trainable initial hidden state can be activated by setting this to `true`
  - `train_memory`: Trainable initial memory can be activated by setting this to `true`
  - `init_encoder_bias`: Initializer for encoder bias $\mathbf{b}_{ih}^{z}$.  
    Must be a single function. If `nothing`, initialized from a uniform distribution in
    `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
  - `init_recurrent_bias`: Initializer for recurrent bias $\mathbf{b}_{hh}^{u}$.  
    Must be a single function. If `nothing`, initialized from a uniform distribution in  
    `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
  - `init_memory_bias`: Initializer for memory bias $\mathbf{b}_{zh}^{u}$.  
    Must be a single function. If `nothing`, initialized from a uniform distribution in  
    `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
  - `init_encoder_weight`: Initializer for encoder weight $\mathbf{W}_{ih}^{z}$.  
    Must be a single function. If `nothing`, initialized from a uniform distribution in  
    `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
  - `init_recurrent_weight`: Initializer for recurrent weight $\mathbf{W}_{hh}^{u}$.  
    Must be a single function. If `nothing`, initialized from a uniform distribution in  
    `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
  - `init_memory_weight`: Initializer for memory weight $\mathbf{W}_{zh}^{u}$.  
    Must be a single function. If `nothing`, initialized from a uniform distribution in  
    `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
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

  - `weight_ih`: Encoder weights  
    ``\{ \mathbf{W}_{ih}^{z} \}``  
  - `weight_hh`: Recurrent weights  
    ``\{ \mathbf{W}_{hh}^{u} \}``  
  - `weight_mm`: Memory weights  
    ``\{ \mathbf{W}_{zh}^{u} \}``  
  - `bias_ih`: Encoder bias (if `use_bias=true`)  
    ``\{ \mathbf{b}_{ih}^{z} \}``  
  - `bias_hh`: Recurrent bias (if `use_bias=true`)  
    ``\{ \mathbf{b}_{hh}^{u} \}``  
  - `bias_mm`: Memory bias (if `use_bias=true`)  
    ``\{ \mathbf{b}_{zh}^{u} \}``  
  - `hidden_state`: Initial hidden state vector $\mathbf{h}(0)$  
    (not present if `train_state=false`).
  - `memory`: Initial memory vector $\mathbf{c}(0)$  
    (not present if `train_memory=false`).

## States

  - `rng`: Controls the randomness (if any) in the initial state generation
"""
@concrete struct MinimalRNNCell{TS <: StaticBool, TM <: StaticBool} <:
                 AbstractDoubleRecurrentCell{TS, TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_encoder_bias
    init_recurrent_bias
    init_memory_bias
    init_encoder_weight
    init_recurrent_weight
    init_memory_weight
    init_state
    init_memory
    use_bias <: StaticBool
end

function MinimalRNNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh;
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_encoder_bias=nothing, init_recurrent_bias=nothing, init_memory_bias=nothing,
        init_encoder_weight=nothing, init_recurrent_weight=nothing,
        init_memory_weight=nothing, init_state=zeros32, init_memory=zeros32)
    return MinimalRNNCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_encoder_bias, init_recurrent_bias, init_memory_bias, init_encoder_weight,
        init_recurrent_weight, init_memory_weight, init_state, init_memory,
        static(use_bias))
end

function initialparameters(rng::AbstractRNG, minimal::MinimalRNNCell)
    weight_ih = init_rnn_weight(rng, minimal.init_encoder_weight, minimal.out_dims,
        (minimal.out_dims, minimal.in_dims))
    weight_hh = init_rnn_weight(rng, minimal.init_recurrent_weight, minimal.out_dims,
        (minimal.out_dims, minimal.out_dims))
    weight_mm = init_rnn_weight(rng, minimal.init_memory_weight, minimal.out_dims,
        (minimal.out_dims, minimal.out_dims))
    ps = (; weight_ih, weight_hh, weight_mm)
    if has_bias(minimal)
        bias_ih = init_rnn_bias(
            rng, minimal.init_encoder_bias, minimal.out_dims, minimal.out_dims)
        bias_hh = init_rnn_bias(rng, minimal.init_recurrent_bias, minimal.out_dims, minimal.out_dims)
        bias_mm = init_rnn_bias(rng, minimal.init_memory_bias, minimal.out_dims, minimal.out_dims)
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
