#https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
@doc raw"""
    PeepholeLSTMCell(in_dims => out_dims;
        use_bias=true, train_state=false, train_memory=false,
        init_bias=nothing, init_recurrent_bias=nothing, init_peephole_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_peephole_weight=nothing, init_state=zeros32, init_memory=zeros32)

[Peephole long short term memory](https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf).

## Equations
```math
\begin{aligned}
    \mathbf{z}(t) &= \tanh\left(
        \mathbf{W}_{ih}^{z} \mathbf{x}(t) + \mathbf{b}_{ih}^{z} +
        \mathbf{W}_{hh}^{z} \mathbf{h}(t-1) + \mathbf{b}_{hh}^{z}
        \right), \\
    \mathbf{i}(t) &= \sigma\left(
        \mathbf{W}_{ih}^{i} \mathbf{x}(t) + \mathbf{b}_{ih}^{i} +
        \mathbf{W}_{hh}^{i} \mathbf{h}(t-1) + \mathbf{b}_{hh}^{i} +
        \mathbf{p}^{i} \circ \mathbf{c}(t-1) + \mathbf{b}_{ph}^{i}
        \right), \\
    \mathbf{f}(t) &= \sigma\left(
        \mathbf{W}_{ih}^{f} \mathbf{x}(t) + \mathbf{b}_{ih}^{f} +
        \mathbf{W}_{hh}^{f} \mathbf{h}(t-1) + \mathbf{b}_{hh}^{f} +
        \mathbf{p}^{f} \circ \mathbf{c}(t-1) + \mathbf{b}_{ph}^{f}
        \right), \\
    \mathbf{c}(t) &= \mathbf{f}(t) \circ \mathbf{c}(t-1) + \mathbf{i}(t) \circ
        \mathbf{z}(t), \\
    \mathbf{o}(t) &= \sigma\left(
        \mathbf{W}_{ih}^{o} \mathbf{x}(t) + \mathbf{b}_{ih}^{o} +
        \mathbf{W}_{hh}^{o} \mathbf{h}(t-1) + \mathbf{b}_{hh}^{o} +
        \mathbf{p}^{o} \circ \mathbf{c}(t) + \mathbf{b}_{ph}^{o}
        \right), \\
    \mathbf{h}(t) &= \mathbf{o}(t) \circ \tanh\left( \mathbf{c}(t) \right)
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
  - `init_bias`: Initializer for input-to-hidden biases  
    $\mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{i}, \mathbf{b}_{ih}^{f}, \mathbf{b}_{ih}^{o}$.  
    Must be a tuple containing 4 functions. If a single value is passed, it is
    copied into a 4-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`, where `bound = inv(sqrt(out_dims))`.  
    The functions are applied in order: the first initializes $\mathbf{b}_{ih}^{z}$,
    the second $\mathbf{b}_{ih}^{i}$, the third $\mathbf{b}_{ih}^{f}$,
    and the fourth $\mathbf{b}_{ih}^{o}$. Default set to `nothing`.
  - `init_recurrent_bias`: Initializer for hidden-to-hidden biases  
    $\mathbf{b}_{hh}^{z}, \mathbf{b}_{hh}^{i}, \mathbf{b}_{hh}^{f}, \mathbf{b}_{hh}^{o}$.  
    Must be a tuple containing 4 functions. If a single value is passed, it is
    copied into a 4-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`, where
    `bound = inv(sqrt(out_dims))`. The functions are applied in order:  
    the first initializes $\mathbf{b}_{hh}^{z}$, the second $\mathbf{b}_{hh}^{i}$,  
    the third $\mathbf{b}_{hh}^{f}$, and the fourth $\mathbf{b}_{hh}^{o}$.  
    Default set to `nothing`.
  - `init_peephole_bias`: Initializer for peephole biases  
    $\mathbf{b}_{ph}^{i}, \mathbf{b}_{ph}^{f}, \mathbf{b}_{ph}^{o}$.  
    Must be a tuple containing 3 functions. If a single value is passed, it is
    copied into a 3-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`, 
    where `bound = inv(sqrt(out_dims))`. The functions are applied in order:  
    the first initializes $\mathbf{b}_{ph}^{i}$, the second $\mathbf{b}_{ph}^{f}$,  
    and the third $\mathbf{b}_{ph}^{o}$.  
    Default set to `nothing`.
  - `init_weight`: Initializer for input-to-hidden weights  
    $\mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{i}, \mathbf{W}_{ih}^{f}, \mathbf{W}_{ih}^{o}$.  
    Must be a tuple containing 4 functions. If a single value is passed, it is
    copied into a 4-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = inv(sqrt(out_dims))`. The functions are applied in order:  
    the first initializes $\mathbf{W}_{ih}^{z}$, the second $\mathbf{W}_{ih}^{i}$,  
    the third $\mathbf{W}_{ih}^{f}$, and the fourth $\mathbf{W}_{ih}^{o}$.  
    Default set to `nothing`.
  - `init_recurrent_weight`: Initializer for hidden-to-hidden weights  
    $\mathbf{W}_{hh}^{z}, \mathbf{W}_{hh}^{i}, \mathbf{W}_{hh}^{f}, \mathbf{W}_{hh}^{o}$.  
    Must be a tuple containing 4 functions. If a single value is passed, it is
    copied into a 4-element tuple. If set to `nothing`, weights are initialized 
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = inv(sqrt(out_dims))`. The functions are applied in order:  
    the first initializes $\mathbf{W}_{hh}^{z}$, the second $\mathbf{W}_{hh}^{i}$,  
    the third $\mathbf{W}_{hh}^{f}$, and the fourth $\mathbf{W}_{hh}^{o}$.  
    Default set to `nothing`.
  - `init_peephole_weight`: Initializer for peephole weights  
    $\mathbf{p}^{i}, \mathbf{p}^{f}, \mathbf{p}^{o}$.  
    Must be a tuple containing 3 functions. If a single value is passed, it is
    copied into a 3-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = inv(sqrt(out_dims))`. The functions are applied in order:  
    the first initializes $\mathbf{p}^{i}$, the second $\mathbf{p}^{f}$,  
    and the third $\mathbf{p}^{o}$.  
    Default set to `nothing`.
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

  - `weight_ih`: Input-to-hidden weights  
    ``\{ \mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{i}, \mathbf{W}_{ih}^{f}, \mathbf{W}_{ih}^{o} \}``  
  - `weight_hh`: Hidden-to-hidden weights  
    ``\{ \mathbf{W}_{hh}^{z}, \mathbf{W}_{hh}^{i}, \mathbf{W}_{hh}^{f}, \mathbf{W}_{hh}^{o} \}``  
  - `weight_ph`: Peephole weights  
    ``\{ \mathbf{p}^{i}, \mathbf{p}^{f}, \mathbf{p}^{o} \}``  
  - `bias_ih`: Input-to-hidden biases (if `use_bias=true`)  
    ``\{ \mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{i}, \mathbf{b}_{ih}^{f}, \mathbf{b}_{ih}^{o} \}``  
  - `bias_hh`: Hidden-to-hidden biases (if `use_bias=true`)  
    ``\{ \mathbf{b}_{hh}^{z}, \mathbf{b}_{hh}^{i}, \mathbf{b}_{hh}^{f}, \mathbf{b}_{hh}^{o} \}``  
  - `bias_ph`: Peephole biases (if `use_bias=true`)  
    ``\{ \mathbf{b}_{ph}^{i}, \mathbf{b}_{ph}^{f}, \mathbf{b}_{ph}^{o} \}``  
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)  
  - `memory`: Initial memory vector (not present if `train_memory=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct PeepholeLSTMCell{TS <: StaticBool, TM <: StaticBool} <:
                 AbstractDoubleRecurrentCell{TS, TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_peephole_bias
    init_weight
    init_recurrent_weight
    init_peephole_weight
    init_state
    init_memory
    use_bias <: StaticBool
end

function PeepholeLSTMCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_peephole_weight=nothing, init_recurrent_bias=nothing, init_peephole_bias=nothing,
        init_state=zeros32, init_memory=zeros32)
    init_weight isa NTuple{4} || (init_weight = ntuple(Returns(init_weight), 4))
    init_recurrent_weight isa NTuple{4} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 4))
    init_peephole_weight isa NTuple{3} ||
        (init_peephole_weight = ntuple(Returns(init_peephole_weight), 3))
    init_bias isa NTuple{4} || (init_bias = ntuple(Returns(init_bias), 4))
    init_recurrent_bias isa NTuple{4} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 4))
    init_peephole_bias isa NTuple{3} ||
        (init_peephole_bias = ntuple(Returns(init_peephole_bias), 3))
    return PeepholeLSTMCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_recurrent_bias, init_peephole_bias, init_weight, init_recurrent_weight,
        init_peephole_weight, init_state, init_memory, static(use_bias))
end

function initialparameters(rng::AbstractRNG, lstm::PeepholeLSTMCell)
    # weights
    weight_ih = multi_inits(
        rng, lstm.init_weight, lstm.out_dims, (lstm.out_dims, lstm.in_dims))
    weight_hh = multi_inits(
        rng, lstm.init_recurrent_weight, lstm.out_dims, (lstm.out_dims, lstm.out_dims))
    weight_ph = multi_inits(
        rng, lstm.init_peephole_weight, lstm.out_dims, (lstm.out_dims, lstm.out_dims))
    ps = (; weight_ih, weight_hh, weight_ph)
    # biases
    if has_bias(lstm)
        bias_ih = multi_bias(rng, lstm.init_bias, lstm.out_dims, lstm.out_dims)
        bias_hh = multi_bias(rng, lstm.init_recurrent_bias, lstm.out_dims, lstm.out_dims)
        bias_ph = multi_bias(rng, lstm.init_peephole_bias, lstm.out_dims, lstm.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh, bias_ph))
    end
    # trainable state and/or memory
    has_train_state(lstm) &&
        (ps = merge(ps, (hidden_state=lstm.init_state(rng, lstm.out_dims),)))
    known(lstm.train_memory) &&
        (ps = merge(ps, (memory=lstm.init_memory(rng, lstm.out_dims),)))
    return ps
end

function parameterlength(lstm::PeepholeLSTMCell)
    return lstm.in_dims * lstm.out_dims * 4 + lstm.out_dims * lstm.out_dims * 12 +
           lstm.out_dims * 8
end

function (lstm::PeepholeLSTMCell)(
        (inp,
            (state, c_state))::Tuple{
            <:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state, matched_cstate = match_eltype(
        lstm, ps, st, inp, state, c_state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    bias_ph = safe_getproperty(ps, Val(:bias_ph))
    #gates
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    full_ghs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)
    full_gps = fused_dense_bias_activation(identity, ps.weight_ph, matched_cstate, bias_ph)
    gates = full_gxs .+ full_ghs
    input, forget, cell, output = multigate(gates, Val(4))
    gpeep = multigate(full_gps, Val(3))
    #computation
    new_cstate = @. sigmoid_fast(forget + gpeep[1]) * matched_cstate +
                    sigmoid_fast(input + gpeep[2]) * tanh_fast(cell)
    new_state = @. sigmoid_fast(output + gpeep[3]) * tanh_fast(new_cstate)
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, lstm::PeepholeLSTMCell)
    print(io, "PeepholeLSTMCell($(lstm.in_dims) => $(lstm.out_dims)")
    has_bias(lstm) || print(io, ", use_bias=false")
    has_train_state(lstm) && print(io, ", train_state=true")
    known(lstm.train_memory) && print(io, ", train_memory=true")
    print(io, ")")
end
