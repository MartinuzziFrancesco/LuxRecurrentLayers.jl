#https://arxiv.org/abs/2109.00020
@doc raw"""
    WMCLSTMCell(in_dims => out_dims;
        use_bias=true, train_state=false, train_memory=false,
        init_bias=nothing, init_recurrent_bias=nothing, init_memory_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_memory_weight=nothing, init_state=zeros32, init_memory=zeros32)


[Long short term memory cell with working memory
connections](https://arxiv.org/abs/2109.00020).

## Equations
```math
\begin{aligned}
    \mathbf{i}(t) &= \sigma\left( \mathbf{W}_{ih}^{i} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{i} + \mathbf{W}_{hh}^{i} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{i} + \mathbf{W}_{mh}^{i} \mathbf{c}(t-1) +
        \mathbf{b}_{mh}^{i} \right), \\
    \mathbf{f}(t) &= \sigma\left( \mathbf{W}_{ih}^{f} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{f} + \mathbf{W}_{hh}^{f} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{f} + \mathbf{W}_{mh}^{f} \mathbf{c}(t-1) +
        \mathbf{b}_{mh}^{f} \right), \\
    \mathbf{c}(t) &= \mathbf{f}(t) \circ \mathbf{c}(t-1) + \mathbf{i}(t) \circ
        \sigma_c\left( \mathbf{W}_{ih}^{c} \mathbf{x}(t) + \mathbf{b}_{ih}^{c}
        \right), \\
    \mathbf{o}(t) &= \sigma\left( \mathbf{W}_{ih}^{o} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{o} + \mathbf{W}_{hh}^{o} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{o} + \mathbf{W}_{mh}^{o} \mathbf{c}(t) +
        \mathbf{b}_{mh}^{o} \right), \\
    \mathbf{h}(t) &= \mathbf{o}(t) \circ \sigma_h\left( \mathbf{c}(t) \right)
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
    $\mathbf{b}_{ih}^{i}, \mathbf{b}_{ih}^{f}, \mathbf{b}_{ih}^{c}, \mathbf{b}_{ih}^{o}$.  
    Must be a tuple containing 4 functions. If a single value is passed, it is
    copied into a 4-element tuple. If set to `nothing`, biases are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
    The functions are applied in order:  
    the first initializes $\mathbf{b}_{ih}^{i}$, the second $\mathbf{b}_{ih}^{f}$,  
    the third $\mathbf{b}_{ih}^{c}$, the fourth $\mathbf{b}_{ih}^{o}$.  
    Default set to `nothing`.
  - `init_recurrent_bias`: Initializer for hidden-to-hidden biases  
    $\mathbf{b}_{hh}^{i}, \mathbf{b}_{hh}^{f}, \mathbf{b}_{hh}^{o}$.  
    Must be a tuple containing 3 functions. If a single value is passed, it is
    copied into a 3-element tuple. If set to `nothing`, biases are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
    The functions are applied in order:  
    the first initializes $\mathbf{b}_{hh}^{i}$, the second $\mathbf{b}_{hh}^{f}$,  
    the third $\mathbf{b}_{hh}^{o}$.  
    Default set to `nothing`.
  - `init_memory_bias`: Initializer for memory-to-hidden biases  
    $\mathbf{b}_{mh}^{i}, \mathbf{b}_{mh}^{f}, \mathbf{b}_{mh}^{o}$.  
    Must be a tuple containing 3 functions. If a single value is passed, it is
    copied into a 3-element tuple. If set to `nothing`, biases are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
    The functions are applied in order:  
    the first initializes $\mathbf{b}_{mh}^{i}$, the second $\mathbf{b}_{mh}^{f}$,  
    the third $\mathbf{b}_{mh}^{o}$.  
    Default set to `nothing`.
  - `init_weight`: Initializer for input-to-hidden weights  
    $\mathbf{W}_{ih}^{i}, \mathbf{W}_{ih}^{f}, \mathbf{W}_{ih}^{c}, \mathbf{W}_{ih}^{o}$.  
    Must be a tuple containing 4 functions. If a single value is passed, it is
    copied into a 4-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
    The functions are applied in order:  
    the first initializes $\mathbf{W}_{ih}^{i}$, the second $\mathbf{W}_{ih}^{f}$,  
    the third $\mathbf{W}_{ih}^{c}$, the fourth $\mathbf{W}_{ih}^{o}$.  
    Default set to `nothing`.
  - `init_recurrent_weight`: Initializer for hidden-to-hidden weights  
    $\mathbf{W}_{hh}^{i}, \mathbf{W}_{hh}^{f}, \mathbf{W}_{hh}^{o}$.  
    Must be a tuple containing 3 functions. If a single value is passed, it is
    copied into a 3-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
    The functions are applied in order:  
    the first initializes $\mathbf{W}_{hh}^{i}$, the second $\mathbf{W}_{hh}^{f}$,  
    the third $\mathbf{W}_{hh}^{o}$.  
    Default set to `nothing`.
  - `init_memory_weight`: Initializer for memory-to-hidden weights  
    $\mathbf{W}_{mh}^{i}, \mathbf{W}_{mh}^{f}, \mathbf{W}_{mh}^{o}$.  
    Must be a tuple containing 3 functions. If a single value is passed, it is
    copied into a 3-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = \mathrm{inv}(\sqrt{\mathrm{out\_dims}})`.  
    The functions are applied in order:  
    the first initializes $\mathbf{W}_{mh}^{i}$, the second $\mathbf{W}_{mh}^{f}$,  
    the third $\mathbf{W}_{mh}^{o}$.  
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

  - `weight_ih`: Concatenated weights to map from input space  
    ``\{ \mathbf{W}_{ih}^{f}, \mathbf{W}_{ih}^{c}, \mathbf{W}_{ih}^{i}, \mathbf{W}_{ih}^{o} \}``.
  - `weight_hh`: Concatenated weights to map from hidden space  
    ``\{ \mathbf{W}_{hh}^{f}, \mathbf{W}_{hh}^{c}, \mathbf{W}_{hh}^{i}, \mathbf{W}_{hh}^{o} \}``.
  - `weight_mh`: Concatenated weights to map from memory space  
    ``\{ \mathbf{W}_{mh}^{f}, \mathbf{W}_{mh}^{c}, \mathbf{W}_{mh}^{i} \}``.
  - `bias_ih`: Concatenated bias vector for the input-hidden connection (not present if `use_bias=false`)  
    ``\{ \mathbf{b}_{ih}^{f}, \mathbf{b}_{ih}^{c}, \mathbf{b}_{ih}^{i}, \mathbf{b}_{ih}^{o} \}``.
  - `bias_hh`: Concatenated bias vector for the hidden-hidden connection (not present if `use_bias=false`)  
    ``\{ \mathbf{b}_{hh}^{f}, \mathbf{b}_{hh}^{i}, \mathbf{b}_{hh}^{o} \}``.
  - `bias_mh`: Concatenated bias vector for the memory-hidden connection (not present if `use_bias=false`)  
    ``\{ \mathbf{b}_{mh}^{f}, \mathbf{b}_{mh}^{i}, \mathbf{b}_{mh}^{o} \}``.
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
  - `memory`: Initial memory vector (not present if `train_memory=false`)



## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct WMCLSTMCell{TS <: StaticBool, TM <: StaticBool} <:
                 AbstractDoubleRecurrentCell{TS, TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_memory_bias
    init_weight
    init_recurrent_weight
    init_memory_weight
    init_state
    init_memory
    use_bias <: StaticBool
end

function WMCLSTMCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_memory_weight=nothing, init_recurrent_bias=nothing, init_memory_bias=nothing,
        init_state=zeros32, init_memory=zeros32)
    init_weight isa NTuple{4} || (init_weight = ntuple(Returns(init_weight), 4))
    init_recurrent_weight isa NTuple{4} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 4))
    init_memory_weight isa NTuple{3} ||
        (init_memory_weight = ntuple(Returns(init_memory_weight), 3))
    init_bias isa NTuple{4} || (init_bias = ntuple(Returns(init_bias), 4))
    init_recurrent_bias isa NTuple{4} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 4))
    init_memory_bias isa NTuple{3} ||
        (init_memory_bias = ntuple(Returns(init_memory_bias), 3))
    return WMCLSTMCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_recurrent_bias, init_memory_bias, init_weight, init_recurrent_weight,
        init_memory_weight, init_state, init_memory, static(use_bias))
end

function initialparameters(rng::AbstractRNG, lstm::WMCLSTMCell)
    # weights
    weight_ih = multi_inits(
        rng, lstm.init_weight, lstm.out_dims, (lstm.out_dims, lstm.in_dims))
    weight_hh = multi_inits(
        rng, lstm.init_recurrent_weight, lstm.out_dims, (lstm.out_dims, lstm.out_dims))
    weight_mh = multi_inits(
        rng, lstm.init_memory_weight, lstm.out_dims, (lstm.out_dims, lstm.out_dims))
    ps = (; weight_ih, weight_hh, weight_mh)
    # biases
    if has_bias(lstm)
        bias_ih = multi_bias(rng, lstm.init_bias, lstm.out_dims, lstm.out_dims)
        bias_hh = multi_bias(rng, lstm.init_recurrent_bias, lstm.out_dims, lstm.out_dims)
        bias_mh = multi_bias(rng, lstm.init_memory_bias, lstm.out_dims, lstm.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh, bias_mh))
    end
    # trainable state and/or memory
    has_train_state(lstm) &&
        (ps = merge(ps, (hidden_state=lstm.init_state(rng, lstm.out_dims),)))
    known(lstm.train_memory) &&
        (ps = merge(ps, (memory=lstm.init_memory(rng, lstm.out_dims),)))
    return ps
end

function parameterlength(lstm::WMCLSTMCell)
    return lstm.in_dims * lstm.out_dims * 4 + lstm.out_dims * lstm.out_dims * 7 +
           lstm.out_dims * 11
end

function (lstm::WMCLSTMCell)(
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
    fused_gates = @. full_gxs + full_ghs
    memory_matrices = multigate(ps.weight_mh, Val(3))
    memory_gates = memory_matrices[1] * matched_cstate, memory_matrices[2] * matched_cstate
    gates = multigate(fused_gates, Val(4))

    input_gate = @. sigmoid_fast(gates[1] + tanh_fast(memory_gates[1]))
    forget_gate = @. sigmoid_fast(gates[2] + tanh_fast(memory_gates[2]))
    cell_gate = @. tanh_fast(gates[4])
    new_cstate = @. forget_gate * matched_cstate + input_gate * cell_gate
    memory_gate = memory_matrices[3] * new_cstate
    output_gate = @. sigmoid_fast(gates[3] + tanh_fast(memory_gate))
    new_state = @. output_gate * tanh_fast(new_cstate)
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, lstm::WMCLSTMCell)
    print(io, "WMCLSTMCell($(lstm.in_dims) => $(lstm.out_dims)")
    has_bias(lstm) || print(io, ", use_bias=false")
    has_train_state(lstm) && print(io, ", train_state=true")
    known(lstm.train_memory) && print(io, ", train_memory=true")
    print(io, ")")
end
