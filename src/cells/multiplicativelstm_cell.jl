#https://arxiv.org/abs/1609.07959
@doc raw"""
    MultiplicativeLSTMCell(in_dims => out_dims;
        use_bias=true, train_state=false, train_memory=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_multiplicative_bias=nothing, init_weight=nothing,
        init_recurrent_weight=nothing, init_multiplicative_weight=nothing,
        init_state=zeros32, init_memory=zeros32)

[Multiplicative long short term memory cell](https://arxiv.org/abs/1609.07959).

## Equations
```math
\begin{aligned}
    \mathbf{m}(t) &= \left( \mathbf{W}_{ih}^{m} \mathbf{x}(t) + \mathbf{b}_{ih}^{m}
        \right) \circ \left( \mathbf{W}_{hh}^{m} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{m} \right) \\
    \hat{\mathbf{h}}(t) &= \mathbf{W}_{ih}^{h} \mathbf{x}(t) + \mathbf{b}_{ih}^{h}
        + \mathbf{W}_{mh}^{h} \mathbf{m}(t) + \mathbf{b}_{mh}^{h} \\
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}_{ih}^{i} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{i} + \mathbf{W}_{mh}^{i} \mathbf{m}(t) +
        \mathbf{b}_{mh}^{i} \right) \\
    \mathbf{o}(t) &= \sigma\left( \mathbf{W}_{ih}^{o} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{o} + \mathbf{W}_{mh}^{o} \mathbf{m}(t) +
        \mathbf{b}_{mh}^{o} \right) \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}_{ih}^{f} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{f} + \mathbf{W}_{mh}^{f} \mathbf{m}(t) +
        \mathbf{b}_{mh}^{f} \right) \\
    \mathbf{c}(t) &= \mathbf{f}(t) \circ \mathbf{c}(t-1) + \mathbf{i}(t)
        \circ \tanh\left( \hat{\mathbf{h}}(t) \right) \\
    \mathbf{h}(t) &= \tanh\left( \mathbf{c}(t) \right) \circ \mathbf{o}(t)
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
    $\mathbf{b}_{ih}^{m}, \mathbf{b}_{ih}^{h}, \mathbf{b}_{ih}^{i}, \mathbf{b}_{ih}^{o}, \mathbf{b}_{ih}^{f}$.  
    Must be a tuple containing 5 functions. If a single value is passed, it is copied into a 5-element tuple.  
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`  
    where `bound = inv(sqrt(out_dims))`.  
    The functions are applied in order:  
    the first initializes $\mathbf{b}_{ih}^{m}$, the second $\mathbf{b}_{ih}^{h}$, the third $\mathbf{b}_{ih}^{i}$,  
    the fourth $\mathbf{b}_{ih}^{o}$, and the fifth $\mathbf{b}_{ih}^{f}$.
  - `init_recurrent_bias`: Initializer for hidden-to-hidden biases  
    $\mathbf{b}_{hh}^{m}$.  
    Must be a tuple containing 1 function. If a single value is passed, it is used directly.  
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`  
    where `bound = inv(sqrt(out_dims))`.  
  - `init_multiplicative_bias`: Initializer for multiplicative-to-hidden biases  
    $\mathbf{b}_{mh}^{h}, \mathbf{b}_{mh}^{i}, \mathbf{b}_{mh}^{o}, \mathbf{b}_{mh}^{f}$.  
    Must be a tuple containing 4 functions. If a single value is passed, it is copied into a 4-element tuple.  
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`  
    where `bound = inv(sqrt(out_dims))`.  
    The functions are applied in order:  
    the first initializes $\mathbf{b}_{mh}^{h}$, the second $\mathbf{b}_{mh}^{i}$,  
    the third $\mathbf{b}_{mh}^{o}$, and the fourth $\mathbf{b}_{mh}^{f}$.
  - `init_weight`: Initializer for input-to-hidden weights  
    $\mathbf{W}_{ih}^{m}, \mathbf{W}_{ih}^{h}, \mathbf{W}_{ih}^{i}, \mathbf{W}_{ih}^{o}, \mathbf{W}_{ih}^{f}$.  
    Must be a tuple containing 5 functions. If a single value is passed, it is copied into a 5-element tuple.  
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`  
    where `bound = inv(sqrt(out_dims))`.  
    The functions are applied in order:  
    the first initializes $\mathbf{W}_{ih}^{m}$, the second $\mathbf{W}_{ih}^{h}$,  
    the third $\mathbf{W}_{ih}^{i}$, the fourth $\mathbf{W}_{ih}^{o}$, and the fifth $\mathbf{W}_{ih}^{f}$.
  - `init_recurrent_weight`: Initializer for hidden-to-hidden weights  
    $\mathbf{W}_{hh}^{m}$.  
    Must be a tuple containing 1 function. If a single value is passed, it is used directly.  
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`  
    where `bound = inv(sqrt(out_dims))`.  
  - `init_multiplicative_weight`: Initializer for multiplicative-to-hidden weights  
    $\mathbf{W}_{mh}^{h}, \mathbf{W}_{mh}^{i}, \mathbf{W}_{mh}^{o}, \mathbf{W}_{mh}^{f}$.  
    Must be a tuple containing 4 functions. If a single value is passed, it is copied into a 4-element tuple.  
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`  
    where `bound = inv(sqrt(out_dims))`.  
    The functions are applied in order:  
    the first initializes $\mathbf{W}_{mh}^{h}$, the second $\mathbf{W}_{mh}^{i}$,  
    the third $\mathbf{W}_{mh}^{o}$, and the fourth $\mathbf{W}_{mh}^{f}$.
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
    ``\{ \mathbf{W}_{ih}^{m}, \mathbf{W}_{ih}^{h}, \mathbf{W}_{ih}^{i}, \mathbf{W}_{ih}^{o}, \mathbf{W}_{ih}^{f} \}``  
  - `weight_hh`: Hidden-to-hidden weights  
    ``\{ \mathbf{W}_{hh}^{m} \}``  
  - `weight_mh`: Multiplicative-to-hidden weights  
    ``\{ \mathbf{W}_{mh}^{h}, \mathbf{W}_{mh}^{i}, \mathbf{W}_{mh}^{o}, \mathbf{W}_{mh}^{f} \}``  
  - `bias_ih`: Input-to-hidden biases (if `use_bias=true`)  
    ``\{ \mathbf{b}_{ih}^{m}, \mathbf{b}_{ih}^{h}, \mathbf{b}_{ih}^{i}, \mathbf{b}_{ih}^{o}, \mathbf{b}_{ih}^{f} \}``  
  - `bias_hh`: Hidden-to-hidden biases (if `use_bias=true`)  
    ``\{ \mathbf{b}_{hh}^{m} \}``  
  - `bias_mh`: Multiplicative-to-hidden biases (if `use_bias=true`)  
    ``\{ \mathbf{b}_{mh}^{h}, \mathbf{b}_{mh}^{i}, \mathbf{b}_{mh}^{o}, \mathbf{b}_{mh}^{f} \}``  
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
  - `memory`: Initial memory vector (not present if `train_memory=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct MultiplicativeLSTMCell{TS <: StaticBool, TM <: StaticBool} <:
                 AbstractDoubleRecurrentCell{TS, TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_multiplicative_bias
    init_weight
    init_recurrent_weight
    init_multiplicative_weight
    init_state
    init_memory
    use_bias <: StaticBool
end

function MultiplicativeLSTMCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_multiplicative_weight=nothing, init_recurrent_bias=nothing, init_multiplicative_bias=nothing,
        init_state=zeros32, init_memory=zeros32)
    init_weight isa NTuple{5} || (init_weight = ntuple(Returns(init_weight), 5))
    init_multiplicative_weight isa NTuple{4} ||
        (init_multiplicative_weight = ntuple(Returns(init_multiplicative_weight), 4))
    init_bias isa NTuple{5} || (init_bias = ntuple(Returns(init_bias), 5))
    init_multiplicative_bias isa NTuple{4} ||
        (init_multiplicative_bias = ntuple(Returns(init_multiplicative_bias), 4))
    return MultiplicativeLSTMCell(
        static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_recurrent_bias, init_multiplicative_bias, init_weight, init_recurrent_weight,
        init_multiplicative_weight, init_state, init_memory, static(use_bias))
end

function initialparameters(rng::AbstractRNG, lstm::MultiplicativeLSTMCell)
    # weights
    weight_ih = multi_inits(
        rng, lstm.init_weight, lstm.out_dims, (lstm.out_dims, lstm.in_dims))
    weight_hh = init_rnn_weight(
        rng, lstm.init_recurrent_weight, lstm.out_dims, (lstm.out_dims, lstm.out_dims))
    weight_mh = multi_inits(
        rng, lstm.init_multiplicative_weight, lstm.out_dims, (lstm.out_dims, lstm.out_dims))
    ps = (; weight_ih, weight_hh, weight_mh)
    # biases
    if has_bias(lstm)
        bias_ih = multi_bias(rng, lstm.init_bias, lstm.out_dims, lstm.out_dims)
        bias_hh = init_rnn_bias(rng, lstm.init_recurrent_bias, lstm.out_dims, lstm.out_dims)
        bias_mh = multi_bias(
            rng, lstm.init_multiplicative_bias, lstm.out_dims, lstm.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh, bias_mh))
    end
    # trainable state and/or memory
    has_train_state(lstm) &&
        (ps = merge(ps, (hidden_state=lstm.init_state(rng, lstm.out_dims),)))
    known(lstm.train_memory) &&
        (ps = merge(ps, (memory=lstm.init_memory(rng, lstm.out_dims),)))
    return ps
end

function parameterlength(lstm::MultiplicativeLSTMCell)
    return lstm.in_dims * lstm.out_dims * 5 + lstm.out_dims * lstm.out_dims * 5 +
           lstm.out_dims * 10
end

function (lstm::MultiplicativeLSTMCell)(
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
    bias_mh = safe_getproperty(ps, Val(:bias_mh))
    #gates
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    ghs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)
    gxs = multigate(full_gxs, Val(5))
    multiplicative_state = gxs[1] .* ghs
    full_gms = fused_dense_bias_activation(
        identity, ps.weight_mh, multiplicative_state, bias_mh)
    gms = multigate(full_gms, Val(4))
    input_gate = @. sigmoid_fast(gxs[2] + gms[1])
    output_gate = @. sigmoid_fast(gxs[3] + gms[2])
    forget_gate = @. sigmoid_fast(gxs[4] + gms[3])
    candidate_state = @. tanh_fast(gxs[5] + gms[4])
    new_cstate = @. forget_gate * matched_cstate + input_gate * candidate_state
    new_state = @. tanh_fast(candidate_state) * output_gate
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, lstm::MultiplicativeLSTMCell)
    print(io, "MultiplicativeLSTMCell($(lstm.in_dims) => $(lstm.out_dims)")
    has_bias(lstm) || print(io, ", use_bias=false")
    has_train_state(lstm) && print(io, ", train_state=true")
    known(lstm.train_memory) && print(io, ", train_memory=true")
    print(io, ")")
end
