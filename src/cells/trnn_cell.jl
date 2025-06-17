#https://arxiv.org/abs/1911.11033
@doc raw"""
    TRNNCell(in_dims => out_dims;
        use_bias=true, train_state=false, init_bias=nothing,
        init_weight=nothing, init_state=zeros32)

[Strongly typed recurrent unit](https://arxiv.org/abs/1602.02218).

## Equations
```math
\begin{aligned}
    \mathbf{z}(t) &= \mathbf{W}_{ih}^{z} \mathbf{x}(t) + \mathbf{b}_{ih}^{z}, \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}_{ih}^{f} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{f} \right), \\
    \mathbf{h}(t) &= \mathbf{f}(t) \circ \mathbf{h}(t-1) + \left(1 -
        \mathbf{f}(t)\right) \circ \mathbf{z}(t).
\end{aligned}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension

# Keyword arguments


  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.  
    Default set to `false`.
  - `init_bias`: Initializer for input-to-hidden biases  
    $\mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{f}$.  
    Must be a tuple containing 2 functions. If a single value is passed, it is
    copied into a 2-element tuple. If set to `nothing`, biases are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
    The functions are applied in order: the first initializes  
    $\mathbf{b}_{ih}^{z}$, the second $\mathbf{b}_{ih}^{f}$.  
    Default set to `nothing`.
  - `init_weight`: Initializer for input-to-hidden weights  
    $\mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{f}$.  
    Must be a tuple containing 2 functions. If a single value is passed, it is
    copied into a 2-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
    The functions are applied in order: the first initializes  
    $\mathbf{W}_{ih}^{z}$, the second $\mathbf{W}_{ih}^{f}$.  
    Default set to `nothing`.
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

  - `weight_ih`: Concatenated weights to map from input space  
    ``\{ \mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{f} \}``
  - `bias_ih`: Concatenated bias vector for input-hidden connections  
    ``\{ \mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{f} \}``  
    (not present if `use_bias=false`)
  - `hidden_state`: Initial hidden state vector  
    (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct TRNNCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
    train_state::TS
    in_dims <: IntegerType
    out_dims <: IntegerType
    activation
    init_bias
    init_weight
    init_state
    use_bias <: StaticBool
end

function TRNNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh_fast;
        use_bias::BoolType=True(), train_state::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_state=zeros32)
    init_weight isa NTuple{2} || (init_weight = ntuple(Returns(init_weight), 2))
    init_bias isa NTuple{2} || (init_bias = ntuple(Returns(init_bias), 2))
    return TRNNCell(static(train_state), in_dims, out_dims, activation,
        init_bias, init_weight, init_state, static(use_bias))
end

function initialparameters(rng::AbstractRNG, trnn::TRNNCell)
    weight_ih = multi_inits(
        rng, trnn.init_weight, trnn.out_dims, (trnn.out_dims, trnn.in_dims))
    ps = (; weight_ih)
    if has_bias(trnn)
        bias_ih = multi_bias(rng, trnn.init_bias, trnn.out_dims, trnn.out_dims)
        ps = merge(ps, (; bias_ih))
    end
    has_train_state(trnn) &&
        (ps = merge(ps, (hidden_state=trnn.init_state(rng, trnn.out_dims),)))
    return ps
end

initialstates(rng::AbstractRNG, ::TRNNCell) = (rng=Utils.sample_replicate(rng),)

function parameterlength(trnn::TRNNCell)
    return trnn.in_dims * trnn.out_dims * 2 + trnn.out_dims * 2
end

statelength(::TRNNCell) = 1

function (trnn::TRNNCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(trnn, ps, st, inp, state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    #computation
    t_ones = eltype(bias_ih)(1.0f0)
    full_xs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    gxs = multigate(full_xs, Val(2))

    forget_gate = trnn.activation.(gxs[1])
    new_state = @. forget_gate * matched_state + (t_ones - forget_gate) * gxs[2]
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, trnn::TRNNCell)
    print(io, "TRNNCell($(trnn.in_dims) => $(trnn.out_dims)")
    has_bias(trnn) || print(io, ", use_bias=false")
    has_train_state(trnn) && print(io, ", train_state=true")
    print(io, ")")
end

#https://arxiv.org/abs/2109.00020
@doc raw"""
    TGRUCell(in_dims => out_dims;
        use_bias=true, train_state=false, train_memory=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32, init_memory=zeros32)


[Strongly typed gated recurrent unit](https://arxiv.org/abs/1602.02218).

## Equations
```math
\begin{aligned}
    \mathbf{z}(t) &= \mathbf{W}_{ih}^{z} \, \mathbf{x}(t) + \mathbf{b}_{ih}^{z}
        + \mathbf{W}_{mh}^{z} \, \mathbf{x}(t-1) + \mathbf{b}_{mh}^{z}, \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}_{ih}^{f} \, \mathbf{x}(t) +
        \mathbf{b}_{ih}^{f} + \mathbf{W}_{mh}^{f} \, \mathbf{x}(t-1) +
        \mathbf{b}_{mh}^{f} \right), \\
    \mathbf{o}(t) &= \tau\left( \mathbf{W}_{ih}^{o} \, \mathbf{x}(t) +
        \mathbf{b}_{ih}^{o} + \mathbf{W}_{mh}^{o} \, \mathbf{x}(t-1) +
        \mathbf{b}_{mh}^{o} \right), \\
    \mathbf{h}(t) &= \mathbf{f}(t) \circ \mathbf{h}(t-1) + \mathbf{z}(t) \circ
        \mathbf{o}(t)
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
    $\mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{f}, \mathbf{b}_{ih}^{o}$.  
    Must be a tuple containing 3 functions. If a single value is passed, it is
    copied into a 3-element tuple. If set to `nothing`, biases are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
    The functions are applied in order: the first initializes  
    $\mathbf{b}_{ih}^{z}$, then $\mathbf{b}_{ih}^{f}$, and $\mathbf{b}_{ih}^{o}$.  
    Default set to `nothing`.
  - `init_recurrent_bias`: Initializer for hidden-to-hidden biases  
    $\mathbf{b}_{mh}^{z}, \mathbf{b}_{mh}^{f}, \mathbf{b}_{mh}^{o}$.
    Must be a tuple containing 3 functions. If a single value is passed, it is
    copied into a 3-element tuple. If set to `nothing`, biases are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
    Default set to `nothing`.
  - `init_weight`: Initializer for input-to-hidden weights  
    $\mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{f}, \mathbf{W}_{ih}^{o}$.  
    Must be a tuple containing 3 functions. If a single value is passed, it is
    copied into a 3-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
    The functions are applied in order: the first initializes  
    $\mathbf{W}_{ih}^{z}$, then $\mathbf{W}_{ih}^{f}$, and $\mathbf{W}_{ih}^{o}$.  
    Default set to `nothing`.
  - `init_recurrent_weight`: Initializer for hidden-to-hidden weights  
    $\mathbf{W}_{mh}^{z}, \mathbf{W}_{mh}^{f}, \mathbf{W}_{mh}^{o}$.  
    Must be a tuple containing 3 functions. If a single value is passed, it is
    copied into a 3-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
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

  - `weight_ih`: Concatenated weights for input-to-hidden transformations  
    ``\{ \mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{f}, \mathbf{W}_{ih}^{o} \}``.
  - `weight_hh`: Concatenated weights for hidden-to-hidden transformations  
    ``\{ \mathbf{W}_{mh}^{z}, \mathbf{W}_{mh}^{f}, \mathbf{W}_{mh}^{o} \}``.
  - `bias_ih`: Input-to-hidden bias vector
    ``\{ \mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{f}, \mathbf{b}_{ih}^{o} \}``.
    (not present if `use_bias=false`).
  - `bias_hh`: Hidden-to-hidden bias vector
    ``\{ \mathbf{b}_{mh}^{z}, \mathbf{b}_{mh}^{f}, \mathbf{b}_{mh}^{o} \}``. 
    (not present if `use_bias=false`).
  - `hidden_state`: Initial hidden state vector  
    (not present if `train_state=false`).
  - `memory`: Initial memory vector  
    (not present if `train_memory=false`).

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct TGRUCell{TS <: StaticBool, TM <: StaticBool} <:
                 AbstractDoubleRecurrentCell{TS, TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_weight
    init_recurrent_weight
    init_state
    init_memory
    use_bias <: StaticBool
end

function TGRUCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_recurrent_bias=nothing,
        init_state=zeros32, init_memory=zeros32)
    init_weight isa NTuple{3} || (init_weight = ntuple(Returns(init_weight), 3))
    init_recurrent_weight isa NTuple{3} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 3))
    init_bias isa NTuple{3} || (init_bias = ntuple(Returns(init_bias), 3))
    init_recurrent_bias isa NTuple{3} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 3))
    return TGRUCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_recurrent_bias, init_weight, init_recurrent_weight,
        init_state, init_memory, static(use_bias))
end

function initialparameters(rng::AbstractRNG, tgru::TGRUCell)
    return multi_initialparameters(rng, tgru)
end

function parameterlength(tgru::TGRUCell)
    return tgru.in_dims * tgru.out_dims * 3 + tgru.out_dims * tgru.out_dims * 3 +
           tgru.out_dims * 6
end

function (tgru::TGRUCell)(
        (inp,
            (state, c_state))::Tuple{
            <:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state, matched_cstate = match_eltype(
        tgru, ps, st, inp, state, c_state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    bias_ph = safe_getproperty(ps, Val(:bias_ph))
    #gates
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    full_ghs = fused_dense_bias_activation(identity, ps.weight_hh, matched_cstate, bias_hh)
    fused_gates = @. full_gxs + full_ghs
    gates = multigate(fused_gates, Val(3))

    update_gate = @. sigmoid_fast(gates[2])
    candidate_state = @. tanh_fast(gates[3])
    new_state = @. update_gate * matched_state + gates[1] * candidate_state
    new_cstate = matched_inp
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, tgru::TGRUCell)
    print(io, "TGRUCell($(tgru.in_dims) => $(tgru.out_dims)")
    has_bias(tgru) || print(io, ", use_bias=false")
    has_train_state(tgru) && print(io, ", train_state=true")
    known(tgru.train_memory) && print(io, ", train_memory=true")
    print(io, ")")
end

#https://arxiv.org/abs/2109.00020
@doc raw"""
    TLSTMCell(in_dims => out_dims;
        use_bias=true, train_state=false, train_memory=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32, init_memory=zeros32)


[Strongly typed long short term memory cell](https://arxiv.org/abs/1602.02218).

## Equations
```math
\begin{aligned}
    \mathbf{z}(t) &= \mathbf{W}_{mh}^{z} \mathbf{x}(t{-}1) + \mathbf{b}_{mh}^{z} + 
        \mathbf{W}_{ih}^{z} \mathbf{x}(t) + \mathbf{b}_{ih}^{z} \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}_{mh}^{f} \mathbf{x}(t{-}1) + \mathbf{b}_{mh}^{f} +
        \mathbf{W}_{ih}^{f} \mathbf{x}(t) + \mathbf{b}_{ih}^{f} \right) \\
    \mathbf{o}(t) &= \tau\left( \mathbf{W}_{mh}^{o} \mathbf{x}(t{-}1) + \mathbf{b}_{mh}^{o} +
        \mathbf{W}_{ih}^{o} \mathbf{x}(t) + \mathbf{b}_{ih}^{o} \right) \\
    \mathbf{c}(t) &= \mathbf{f}(t) \circ \mathbf{c}(t{-}1) + \left(1 -
        \mathbf{f}(t)\right) \circ \mathbf{z}(t) \\
    \mathbf{h}(t) &= \mathbf{c}(t) \circ \mathbf{o}(t)
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
  $\mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{f}, \mathbf{b}_{ih}^{o}$.  
  Must be a tuple containing 3 functions. If a single value is passed, it is
  copied into a 3-element tuple. If set to `nothing`, biases are initialized
  from a uniform distribution within `[-bound, bound]`,  
  where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
  The functions are applied in order to initialize  
  $\mathbf{b}_{ih}^{z}$, $\mathbf{b}_{ih}^{f}$, $\mathbf{b}_{ih}^{o}$$.  
  Default set to `nothing`.
- `init_recurrent_bias`: Initializer for memory-to-hidden biases  
  $\mathbf{b}_{mh}^{z}, \mathbf{b}_{mh}^{f}, \mathbf{b}_{mh}^{o}$.  
  Must be a tuple containing 3 functions. If a single value is passed, it is
  copied into a 3-element tuple. If set to `nothing`, biases are initialized
  from a uniform distribution within `[-bound, bound]`,  
  where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
  The functions are applied in order to initialize  
  $\mathbf{b}_{mh}^{z}$, $\mathbf{b}_{mh}^{f}$, $\mathbf{b}_{mh}^{o}$$.  
  Default set to `nothing`.
- `init_weight`: Initializer for input-to-hidden weights  
  $\mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{f}, \mathbf{W}_{ih}^{o}$.  
  Must be a tuple containing 3 functions. If a single value is passed, it is
  copied into a 3-element tuple. If set to `nothing`, weights are initialized
  from a uniform distribution within `[-bound, bound]`,  
  where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
  The functions are applied in order to initialize  
  $\mathbf{W}_{ih}^{z}$, $\mathbf{W}_{ih}^{f}$, $\mathbf{W}_{ih}^{o}$$.  
  Default set to `nothing`.
- `init_recurrent_weight`: Initializer for memory-to-hidden weights  
  $\mathbf{W}_{mh}^{z}, \mathbf{W}_{mh}^{f}, \mathbf{W}_{mh}^{o}$.  
  Must be a tuple containing 3 functions. If a single value is passed, it is
  copied into a 3-element tuple. If set to `nothing`, weights are initialized
  from a uniform distribution within `[-bound, bound]`,  
  where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
  The functions are applied in order to initialize  
  $\mathbf{W}_{mh}^{z}$, $\mathbf{W}_{mh}^{f}$, $\mathbf{W}_{mh}^{o}$$.  
  Default set to `nothing`.
- `init_state`: Initializer for hidden state.  
  Default set to `zeros32`.
- `init_memory`: Initializer for memory.  
  Default set to `zeros32`.


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

  - `weight_ih`: Concatenated weights for input-to-hidden transformations  
    ``\{ \mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{f}, \mathbf{W}_{ih}^{o} \}``.
  - `weight_mh`: Concatenated weights for memory-to-hidden transformations  
    ``\{ \mathbf{W}_{mh}^{z}, \mathbf{W}_{mh}^{f}, \mathbf{W}_{mh}^{o} \}``.
  - `bias_ih`: Concatenated bias vector for input-to-hidden transformations  
    ``\{ \mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{f}, \mathbf{b}_{ih}^{o} \}``.  
    Not present if `use_bias = false`.
  - `bias_mh`: Concatenated bias vector for memory-to-hidden transformations  
    ``\{ \mathbf{b}_{mh}^{z}, \mathbf{b}_{mh}^{f}, \mathbf{b}_{mh}^{o} \}``.  
    Not present if `use_bias = false`.
  - `hidden_state`: Initial hidden state vector.  
    Not present if `train_state = false`.
  - `memory`: Initial memory state vector.  
    Not present if `train_memory = false`.


## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct TLSTMCell{TS <: StaticBool, TM <: StaticBool} <:
                 AbstractDoubleRecurrentCell{TS, TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_weight
    init_recurrent_weight
    init_state
    init_memory
    use_bias <: StaticBool
end

function TLSTMCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_recurrent_bias=nothing, init_state=zeros32, init_memory=zeros32)
    init_weight isa NTuple{3} || (init_weight = ntuple(Returns(init_weight), 3))
    init_recurrent_weight isa NTuple{3} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 3))
    init_bias isa NTuple{3} || (init_bias = ntuple(Returns(init_bias), 3))
    init_recurrent_bias isa NTuple{3} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 3))
    return TLSTMCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_recurrent_bias, init_weight, init_recurrent_weight,
        init_state, init_memory, static(use_bias))
end

function initialparameters(rng::AbstractRNG, lstm::TLSTMCell)
    weight_ih = multi_inits(
        rng, lstm.init_weight, lstm.out_dims, (lstm.out_dims, lstm.in_dims))
    weight_hh = multi_inits(rng, lstm.init_recurrent_weight, lstm.out_dims,
        (lstm.out_dims, lstm.in_dims))
    ps = (; weight_ih, weight_hh)
    if has_bias(lstm)
        bias_ih = multi_bias(rng, lstm.init_bias, lstm.out_dims, lstm.out_dims)
        bias_hh = multi_bias(
            rng, lstm.init_recurrent_bias, lstm.out_dims, lstm.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    has_train_state(lstm) &&
        (ps = merge(ps, (hidden_state=lstm.init_state(rng, lstm.out_dims),)))
    known(lstm.train_memory) &&
        (ps = merge(ps, (memory=lstm.init_memory(rng, lstm.out_dims),)))
    return ps
end

function parameterlength(lstm::TLSTMCell)
    return lstm.in_dims * lstm.out_dims * 3 + lstm.out_dims * lstm.out_dims * 3 +
           lstm.out_dims * 6
end

function (lstm::TLSTMCell)(
        (inp,
            (state, c_state, prev_inp)),
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(
        lstm, ps, st, inp, state, c_state)
    matched_previnp, mateched_cstate, = match_eltype(
        lstm, ps, st, prev_inp, c_state, c_state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #gates
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    full_ghs = fused_dense_bias_activation(identity, ps.weight_hh, matched_previnp, bias_hh)
    full_gates = @. full_gxs + full_ghs
    t_ones = eltype(bias_ih)(1.0f0)
    gates = multigate(full_gates, Val(3))
    update_gate = @. sigmoid_fast(gates[2])
    candidate_state = @. tanh_fast(gates[3])
    new_cstate = @. update_gate * mateched_cstate + (t_ones - update_gate) * gates[1]
    new_state = @. new_cstate * candidate_state

    return (new_state, (new_state, new_cstate, matched_inp)), st
end

function (rcell::TLSTMCell{False, False})(inp::AbstractMatrix,
        ps, st::NamedTuple)
    rng = replicate(st.rng)
    state = init_rnn_hidden_state(rng, rcell, inp)
    c_state = init_rnn_hidden_state(rng, rcell, inp)
    return rcell((inp, (state, c_state, inp)), ps, merge(st, (; rng)))
end

function (rcell::TLSTMCell{True, False})(inp::AbstractMatrix,
        ps, st::NamedTuple)
    rng = replicate(st.rng)
    state = init_trainable_rnn_hidden_state(ps.hidden_state, inp)
    c_state = init_rnn_hidden_state(rng, rcell, inp)
    return rcell((inp, (state, c_state, inp)), ps, merge(st, (; rng)))
end

function (rcell::TLSTMCell{False, True})(inp::AbstractMatrix,
        ps, st::NamedTuple)
    rng = replicate(st.rng)
    state = init_rnn_hidden_state(rng, rcell, inp)
    c_state = init_trainable_rnn_hidden_state(ps.hidden_state, inp)
    return rcell((inp, (state, c_state, inp)), ps, merge(st, (; rng)))
end

function (rcell::TLSTMCell{True, True})(inp::AbstractMatrix,
        ps, st::NamedTuple)
    state = init_trainable_rnn_hidden_state(ps.hidden_state, inp)
    c_state = init_trainable_rnn_hidden_state(ps.hidden_state, inp)
    return rcell((inp, (state, c_state, inp)), ps, st)
end

function Base.show(io::IO, lstm::TLSTMCell)
    print(io, "TLSTMCell($(lstm.in_dims) => $(lstm.out_dims)")
    has_bias(lstm) || print(io, ", use_bias=false")
    has_train_state(lstm) && print(io, ", train_state=true")
    known(lstm.train_memory) && print(io, ", train_memory=true")
    print(io, ")")
end
