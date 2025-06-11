#https://arxiv.org/pdf/1412.7753

@doc raw"""
    LEMCell(in_dims => out_dims;
        use_bias=true, train_state=false, train_memory=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32, init_memory=zeros32, dt=1.0)

[Long expressive memory unit](https://arxiv.org/pdf/2110.04744).

# Equations
```math
\begin{aligned}
    \boldsymbol{\Delta t}(t) &= \Delta t \cdot \hat{\sigma} \left( 
        \mathbf{W}_{ih}^{1} \mathbf{x}(t) + \mathbf{b}_{ih}^{1} + 
        \mathbf{W}_{hh}^{1} \mathbf{h}(t-1) + \mathbf{b}_{hh}^{1} \right), \\
    \overline{\boldsymbol{\Delta t}}(t) &= \Delta t \cdot \hat{\sigma} \left( 
        \mathbf{W}_{ih}^{2} \mathbf{x}(t) + \mathbf{b}_{ih}^{2} + 
        \mathbf{W}_{hh}^{2} \mathbf{h}(t-1) + \mathbf{b}_{hh}^{2} \right), \\
    \mathbf{c}(t) &= \left(1 - \boldsymbol{\Delta t}(t)\right) \circ \mathbf{c}(t-1) + 
        \boldsymbol{\Delta t}(t) \circ \sigma\left( 
        \mathbf{W}_{ih}^{c} \mathbf{x}(t) + \mathbf{b}_{ih}^{c} + 
        \mathbf{W}_{hh}^{c} \mathbf{h}(t-1) + \mathbf{b}_{hh}^{c} \right), \\
    \mathbf{h}(t) &= \left(1 - \boldsymbol{\Delta t}(t)\right) \circ \mathbf{h}(t-1) + 
        \boldsymbol{\Delta t}(t) \circ \sigma\left( 
        \mathbf{W}_{ih}^{h} \mathbf{x}(t) + \mathbf{b}_{ih}^{h} + 
        \mathbf{W}_{ch} \mathbf{c}(t) + \mathbf{b}_{ch} \right)
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
    $\{ \mathbf{b}_{ih}^{1}, \mathbf{b}_{ih}^{2}, \mathbf{b}_{ih}^{c}, \mathbf{b}_{ih}^{h} \}$.
    Must be a tuple of 4 functions, e.g., `(glorot_uniform, kaiming_uniform, lecun_normal, zeros)`.
    If a single function is passed, it is expanded to a 4-element tuple.
    If set to `nothing`, biases are initialized from a uniform distribution within
    `[-bound, bound]` where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_recurrent_bias`: Initializer for hidden-to-hidden biases
    $\{ \mathbf{b}_{hh}^{1}, \mathbf{b}_{hh}^{2}, \mathbf{b}_{hh}^{c}, \mathbf{b}_{hh}^{h} \}$.
    Must be a tuple of 3 functions.
    If a single function is passed, it is expanded to a 3-element tuple.
    If set to `nothing`, biases are initialized from a uniform distribution within
    `[-bound, bound]` where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_cell_bias`: Initializer for bias $\mathbf{b}_{ch}$. If set to
    `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_weight`: Initializer for input-to-hidden weights
    $\{ \mathbf{W}_{ih}^{1}, \mathbf{W}_{ih}^{2}, \mathbf{W}_{ih}^{c}, \mathbf{W}_{ih}^{h} \}$.
    Must be a tuple of 4 functions.
    If a single function is passed, it is expanded to a 4-element tuple.
    If set to `nothing`, weights are initialized from a uniform distribution within
    `[-bound, bound]` where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_recurrent_weight`: Initializer for hidden-to-hidden weights
    $\{ \mathbf{W}_{hh}^{1}, \mathbf{W}_{hh}^{2}, \mathbf{W}_{hh}^{c}, \mathbf{W}_{hh}^{h} \}$.
    Must be a tuple of 3 functions.
    If a single function is passed, it is expanded to a 3-element tuple.
    If set to `nothing`, weights are initialized from a uniform distribution within
    `[-bound, bound]` where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
    - `init_cell_weight`: Initializer for input to hidden weight $\mathbf{W}_{ch}$. If set to
    `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_state`: Initializer for hidden state. Default set to `zeros32`.
  - `init_memory`: Initializer for memory. Default set to `zeros32`.
  - `dt`: timestep. Defaul is 1.0.

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

  - `weight_ih`: Concatenated weights mapping from input to internal units  
    ``\{ \mathbf{W}_{ih}^{1}, \mathbf{W}_{ih}^{2}, \mathbf{W}_{ih}^{c}, \mathbf{W}_{ih}^{h} \}``  
    The functions provided in `init_weight` are applied in order:  
    the first initializes $\mathbf{W}_{ih}^{1}$, the second $\mathbf{W}_{ih}^{2}$,
    the third $\mathbf{W}_{ih}^{c}$, and the fourth $\mathbf{W}_{ih}^{h}$.
  - `weight_hh`: Concatenated weights mapping from hidden state to internal units  
    ``\{ \mathbf{W}_{hh}^{1}, \mathbf{W}_{hh}^{2}, \mathbf{W}_{hh}^{c} \}``  
    The functions provided in `init_recurrent_weight` are applied in order:  
    the first initializes $\mathbf{W}_{hh}^{1}$, the second $\mathbf{W}_{hh}^{2}$,
    and the third $\mathbf{W}_{hh}^{c}$.
  -  `weight_ch`: Weights to map from cell space $\mathbf{W}_{ch}$.
  - `bias_ih`: Concatenated input-to-hidden bias vectors (not present if `use_bias=false`)  
    ``\{ \mathbf{b}_{ih}^{1}, \mathbf{b}_{ih}^{2}, \mathbf{b}_{ih}^{c}, \mathbf{b}_{ih}^{h} \}``  
    The functions provided in `init_bias` are applied in order:  
    the first initializes $\mathbf{b}_{ih}^{1}$, the second $\mathbf{b}_{ih}^{2}$,
    the third $\mathbf{b}_{ih}^{c}$, and the fourth $\mathbf{b}_{ih}^{h}$.
  - `bias_hh`: Concatenated hidden-to-hidden bias vectors (not present if `use_bias=false`)  
    ``\{ \mathbf{b}_{hh}^{1}, \mathbf{b}_{hh}^{2}, \mathbf{b}_{hh}^{c} \}``  
    The functions provided in `init_recurrent_bias` are applied in order:  
    the first initializes $\mathbf{b}_{hh}^{1}$, the second $\mathbf{b}_{hh}^{2}$,
    and the third $\mathbf{b}_{hh}^{c}$.
  - `bias_ch`: Bias vector for the cell-hidden connection $\mathbf{b}_{ch}$
    (not present if `use_bias=false`)
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
  - `memory`: Initial memory vector (not present if `train_memory=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct LEMCell{TS <: StaticBool, TM <: StaticBool} <:
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
    init_memory
    use_bias <: StaticBool
    dt
end

function LEMCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_recurrent_bias=nothing, init_cell_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing, init_cell_weight=nothing,
        init_state=zeros32, init_memory=zeros32, dt=1.0)
    init_weight isa NTuple{4} ||
        (init_weight = ntuple(Returns(init_weight), 4))
    init_recurrent_weight isa NTuple{3} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 3))
    init_bias isa NTuple{4} || (init_bias = ntuple(Returns(init_bias), 4))
    init_recurrent_bias isa NTuple{3} || (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 3))
    return LEMCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_recurrent_bias, init_cell_bias, init_weight, init_recurrent_weight,
        init_cell_weight, init_state, init_memory, static(use_bias), dt)
end

function initialparameters(rng::AbstractRNG, lem::LEMCell)
    # weights
    weight_ih = multi_inits(
        rng, lem.init_weight, lem.out_dims, (lem.out_dims, lem.in_dims))
    weight_hh = multi_inits(
        rng, lem.init_recurrent_weight, lem.out_dims, (lem.out_dims, lem.out_dims))
    weight_ch = init_rnn_weight(rng, lem.init_cell_weight, lem.out_dims, (lem.out_dims, lem.out_dims))
    ps = (; weight_ih, weight_hh, weight_ch)
    # biases
    if has_bias(lem)
        bias_ih = multi_bias(rng, lem.init_bias, lem.out_dims, lem.out_dims)
        bias_hh = multi_bias(rng, lem.init_recurrent_bias, lem.out_dims, lem.out_dims)
        bias_ch = init_rnn_bias(rng, lem.init_cell_bias, lem.out_dims, lem.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh, bias_ch))
    end
    # trainable state and/or memory
    has_train_state(lem) &&
        (ps = merge(ps, (hidden_state=lem.init_state(rng, lem.out_dims),)))
    known(lem.train_memory) &&
        (ps = merge(ps, (memory=lem.init_memory(rng, lem.out_dims),)))
    return ps
end

function parameterlength(lem::LEMCell)
    return lem.in_dims * lem.out_dims * 2 + lem.out_dims * lem.out_dims * 4 +
           lem.out_dims * 2 + 1
end

function (lem::LEMCell)(
        (inp,
            (state, c_state))::Tuple{
            <:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state, matched_cstate = match_eltype(
        lem, ps, st, inp, state, c_state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    bias_ch = safe_getproperty(ps, Val(:bias_ch))
    #computation
    gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    xs = multigate(gxs, Val(4))
    ghs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)
    hs = multigate(ghs, Val(3))
    cs = fused_dense_bias_activation(identity, ps.weight_ch, matched_cstate, bias_ch)
    t_ones = eltype(bias_ih)(1.0)

    msdt_bar = @. lem.dt * sigmoid_fast(xs[1] + hs[1])
    ms_dt = @. lem.dt * sigmoid_fast(xs[2] + hs[2])
    new_cstate = @. (t_ones - ms_dt) * matched_cstate + ms_dt * tanh_fast(xs[3] + hs[3])
    new_state = @. (t_ones - msdt_bar) * matched_state + msdt_bar * tanh_fast(xs[4] + cs)
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, lem::LEMCell)
    print(io, "LEMCell($(lem.in_dims) => $(lem.out_dims)")
    has_bias(lem) || print(io, ", use_bias=false")
    has_train_state(lem) && print(io, ", train_state=true")
    print(io, ")")
end