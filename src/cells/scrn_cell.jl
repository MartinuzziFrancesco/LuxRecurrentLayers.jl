#https://arxiv.org/pdf/1412.7753

@doc raw"""
    SCRNCell(in_dims => out_dims;
        use_bias=true, train_state=false, train_memory=false,
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_context_weight=nothing, init_state=zeros32, init_memory=zeros32)

[Structurally contraint recurrent unit](https://arxiv.org/pdf/1412.7753).

## Equations
```math
\begin{aligned}
    \mathbf{s}(t) &= (1 - \alpha) \left( \mathbf{W}_{ih}^{s} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{s} \right) + \alpha \, \mathbf{s}(t-1), \\
    \mathbf{h}(t) &= \sigma\left(
        \mathbf{W}_{ch}^{h} \mathbf{s}(t) + \mathbf{b}_{ch}^{h} +
        \mathbf{W}_{ih}^{h} \mathbf{x}(t) + \mathbf{b}_{ih}^{h} +
        \mathbf{W}_{hh}^{h} \mathbf{h}(t-1) + \mathbf{b}_{hh}^{h}
        \right), \\
    \mathbf{y}(t) &= f\left(
        \mathbf{W}_{ch}^{y} \mathbf{s}(t) + \mathbf{b}_{ch}^{y} +
        \mathbf{W}_{hh}^{y} \mathbf{h}(t) + \mathbf{b}_{hh}^{y}
    \right)
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
    $\mathbf{b}_{ih}^{s}, \mathbf{b}_{ih}^{h}$.
    Must be a tuple containing 2 functions. If a single value is passed, it is
    copied into a 2-element tuple. If set to `nothing`, biases are initialized
    from a uniform distribution within `[-bound, bound]`,
    where `bound = inv(sqrt(out_dims))`.
    The functions are applied in order: the first initializes $\mathbf{b}_{ih}^{s}$, the second $\mathbf{b}_{ih}^{h}$.
    Default set to `nothing`.
  - `init_recurrent_bias`: Initializer for hidden-to-hidden biases
    $\mathbf{b}_{hh}^{h}, \mathbf{b}_{hh}^{y}$.
    Must be a tuple containing 2 functions. If a single value is passed, it is
    copied into a 2-element tuple. If set to `nothing`, biases are initialized
    from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`. Default set to `nothing`.
  - `init_context_bias`: Initializer for context biases
    $\mathbf{b}_{ch}^{h}, \mathbf{b}_{ch}^{y}$.
    Must be a tuple containing 2 functions. If a single value is passed, it is
    copied into a 2-element tuple. If set to `nothing`, biases are initialized
    from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`. Default set to `nothing`.
  - `init_weight`: Initializer for input-to-hidden weights
    $\mathbf{W}_{ih}^{s}, \mathbf{W}_{ih}^{h}$.
    Must be a tuple containing 2 functions. If a single value is passed, it is
    copied into a 2-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`,
    where `bound = inv(sqrt(out_dims))`.
    The functions are applied in order: the first initializes
    $\mathbf{W}_{ih}^{s}$, the second $\mathbf{W}_{ih}^{h}$.
    Default set to `nothing`.
  - `init_recurrent_weight`: Initializer for hidden-to-hidden weights
    $\mathbf{W}_{hh}^{h}, \mathbf{W}_{hh}^{y}$.
    Must be a tuple containing 2 functions. If a single value is passed, it is
    copied into a 2-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`. Default set to `nothing`.
  - `init_context_weight`: Initializer for context weights
    $\mathbf{W}_{ch}^{h}, \mathbf{W}_{ch}^{y}$.
    Must be a tuple containing 2 functions. If a single value is passed, it is
    copied into a 2-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`. Default set to `nothing`.
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

  - `weight_ch`: Context-to-hidden weights
    ``\{ \mathbf{W}_{ch}^{h}, \mathbf{W}_{ch}^{y} \}``
  - `weight_ih`: Input-to-hidden weights
    ``\{ \mathbf{W}_{ih}^{s}, \mathbf{W}_{ih}^{h} \}``
  - `weight_hh`: Hidden-to-hidden weights
    ``\{ \mathbf{W}_{hh}^{h}, \mathbf{W}_{hh}^{y} \}``
  - `bias_ch`: Context-to-hidden biases (not present if `use_bias=false`)
    ``\{ \mathbf{b}_{ch}^{h}, \mathbf{b}_{ch}^{y} \}``
  - `bias_ih`: Input-to-hidden biases (not present if `use_bias=false`)
    ``\{ \mathbf{b}_{ih}^{s}, \mathbf{b}_{ih}^{h} \}``
  - `bias_hh`: Hidden-to-hidden biases (not present if `use_bias=false`)
    ``\{ \mathbf{b}_{hh}^{h}, \mathbf{b}_{hh}^{y} \}``
  - `alpha`: Initial context strength
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
  - `memory`: Initial memory vector (not present if `train_memory=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct SCRNCell{TS<:StaticBool,TM<:StaticBool} <:
                 AbstractDoubleRecurrentCell{TS,TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_context_bias
    init_weight
    init_recurrent_weight
    init_context_weight
    init_state
    init_memory
    use_bias <: StaticBool
end

function SCRNCell((in_dims, out_dims)::Pair{<:IntegerType,<:IntegerType};
    use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
    init_bias=nothing, init_recurrent_bias=nothing, init_context_bias=nothing,
    init_weight=nothing, init_recurrent_weight=nothing,
    init_context_weight=nothing, init_state=zeros32, init_memory=zeros32)
    init_weight isa NTuple{2} || (init_weight = ntuple(Returns(init_weight), 2))
    init_recurrent_weight isa NTuple{2} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 2))
    init_context_weight isa NTuple{2} ||
        (init_context_weight = ntuple(Returns(init_context_weight), 2))
    init_bias isa NTuple{2} || (init_bias = ntuple(Returns(init_bias), 2))
    init_recurrent_bias isa NTuple{2} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 2))
    init_context_bias isa NTuple{2} ||
        (init_context_bias = ntuple(Returns(init_context_bias), 2))
    return SCRNCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_recurrent_bias, init_context_bias, init_weight,
        init_recurrent_weight, init_context_weight, init_state, init_memory,
        static(use_bias))
end

function initialparameters(rng::AbstractRNG, scrn::SCRNCell)
    # weights
    weight_ih = multi_inits(
        rng, scrn.init_weight, scrn.out_dims, (scrn.out_dims, scrn.in_dims))
    weight_hh = multi_inits(
        rng, scrn.init_recurrent_weight, scrn.out_dims, (scrn.out_dims, scrn.out_dims))
    weight_ch = multi_inits(
        rng, scrn.init_context_weight, scrn.out_dims, (scrn.out_dims, scrn.out_dims))
    ps = (; weight_ih, weight_hh, weight_ch)
    # biases
    if has_bias(scrn)
        bias_ih = multi_bias(rng, scrn.init_bias, scrn.out_dims, scrn.out_dims)
        bias_hh = multi_bias(rng, scrn.init_recurrent_bias, scrn.out_dims, scrn.out_dims)
        bias_ch = multi_bias(rng, scrn.init_context_bias, scrn.out_dims, scrn.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh, bias_ch))
    end
    # trainable state and/or memory
    has_train_state(scrn) &&
        (ps = merge(ps, (hidden_state=scrn.init_state(rng, scrn.out_dims),)))
    known(scrn.train_memory) &&
        (ps = merge(ps, (memory=scrn.init_memory(rng, scrn.out_dims),)))
    # any additional trainable parameters
    ps = merge(ps, (alpha=eltype(weight_ih)(0.0f0),))
    return ps
end

function parameterlength(scrn::SCRNCell)
    return scrn.in_dims * scrn.out_dims * 2 + scrn.out_dims * scrn.out_dims * 4 +
           scrn.out_dims * 2 + 1
end

function (scrn::SCRNCell)(
    (inp,
        (state, c_state))::Tuple{
        <:AbstractMatrix,Tuple{<:AbstractMatrix,<:AbstractMatrix}},
    ps, st::NamedTuple)
    #type match
    matched_inp, matched_state, matched_cstate = match_eltype(
        scrn, ps, st, inp, state, c_state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    bias_ch = safe_getproperty(ps, Val(:bias_ch))
    #gates
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    full_gcs = fused_dense_bias_activation(identity, ps.weight_ch, matched_state, bias_ch)
    gxs = multigate(full_gxs, Val(2))
    ghs = multigate(ps.weight_hh, Val(2))
    bhs = multigate(bias_hh, Val(2))
    gcs = multigate(full_gcs, Val(2))
    t_ones = one(eltype(ps.weight_hh))
    #computation
    new_cstate = (t_ones .- ps.alpha) .* gxs[1] .+
                 ps.alpha .* matched_cstate
    hidden_layer = sigmoid_fast.(gxs[2] .+ ghs[1] * matched_state .+ gcs[1] .+ bhs[1])
    new_state = tanh_fast.(ghs[2] * hidden_layer .+ gcs[2] .+ bhs[2])
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, r::SCRNCell)
    print(io, "SCRNCell($(r.in_dims) => $(r.out_dims)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end
