#https://arxiv.org/abs/1612.06212
@doc raw"""
    CFNCell(in_dims => out_dims, [activation];
        use_bias=true, train_state=false, init_bias=nothing,
        init_recurrent_bias=nothing, init_weight=nothing,
        init_recurrent_weight=nothing, init_state=zeros32)


[Chaos free network unit](https://arxiv.org/abs/1612.06212).

## Equations

```math
\begin{aligned}
    \boldsymbol{\theta}(t) &= \sigma\left(\mathbf{W}_{ih}^{\theta} \mathbf{x}(t)
        + \mathbf{b}_{ih}^{\theta} + \mathbf{W}_{hh}^{\theta} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{\theta}\right) \\
    \boldsymbol{\eta}(t) &= \sigma\left(\mathbf{W}_{ih}^{\eta} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{\eta} + \mathbf{W}_{hh}^{\eta} \mathbf{h}(t-1) + 
        \mathbf{b}_{hh}^{\eta} \right) \\
    \mathbf{h}(t) &= \boldsymbol{\theta}(t) \circ \tanh(\mathbf{h}(t-1)) +
        \boldsymbol{\eta}(t) \circ \tanh(\mathbf{W}_{ih}^h \mathbf{x}(t) +
        \mathbf{b}_{ih}^{h})
\end{aligned}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension
  - `activation`: activation function. Default is `tanh`

# Keyword arguments


  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.
    Default set to `false`.
  - `init_bias`: Initializer for input to hidden bias
    $\mathbf{b}_{ih}^{\theta}, \mathbf{b}_{ih}^{\eta}, \mathbf{b}_{ih}^{h}$.
    Must be a tuple containing 3 functions, e.g., `(glorot_normal, kaiming_uniform)`.
    If a single function `fn` is provided, it is automatically expanded into a 3-element
    tuple (fn, fn, fn). If set to `nothing`, weights are initialized from a uniform
    distribution within `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
    Default is `nothing`.
  - `init_recurrent_bias`: Initializer for hidden to hidden bias
    $\mathbf{b}_{hh}^{\theta}, \mathbf{b}_{hh}^{\eta}$.
    Must be a tuple containing 2 functions, e.g., `(glorot_normal, kaiming_uniform)`.
    If a single function `fn` is provided, it is automatically expanded into a 2-element
    tuple (fn, fn). If set to `nothing`, weights are initialized from a uniform
    distribution within `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
    Default is `nothing`.
  - `init_weight`: Initializer for input to hidden weights
    $\mathbf{W}_{ih}^{\theta}, \mathbf{W}_{ih}^{\eta}, \mathbf{W}_{ih}^{h}$.
    Must be a tuple containing 3 functions, e.g., `(glorot_normal, kaiming_uniform)`.
    If a single function `fn` is provided, it is automatically expanded into
    a 3-element tuple (fn, fn, fn). If set to `nothing`, weights are initialized from
    a uniform distribution within `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
    Default is `nothing`.
  - `init_recurrent_weight`: Initializer for input to hidden weights
    $\mathbf{W}_{hh}^{\theta}, \mathbf{W}_{hh}^{\eta}$.
    Must be a tuple containing 2 functions, e.g., `(glorot_normal, kaiming_uniform)`.
    If a single function `fn` is provided, it is automatically expanded into
    a 2-element tuple (fn, fn). If set to `nothing`, weights are initialized from
    a uniform distribution within `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
    Default is `nothing`.
  - `init_state`: Initializer for hidden state. Default set to `zeros32`.

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

  - `weight_ih`: Concatenated weights to map from input to the hidden state.
                 ``\{ \mathbf{W}_{ih}^{\theta}, \mathbf{W}_{ih}^{\eta}, \mathbf{W}_{ih}^{h} \}``
    The initializers in `init_weight` are applied in the order they appear:
    the first function is used for $\mathbf{W}_{ih}^{\theta}$, the second for
    $\mathbf{W}_{ih}^{\eta}$, and the third for $\mathbf{W}_{ih}^h$.
  - `weight_hh`: Concatenated weights to map from hidden to hidden state.
                 ``\{ \mathbf{W}_{hh}^{\theta}, \mathbf{W}_{hh}^{\eta} \}``
    The initializers in `init_recurrent_weight` are applied in the order they appear:
    the first function is used for $\mathbf{W}_{hh}^{\theta}$, and the second for
    $\mathbf{W}_{hh}^{\eta}$.
  - `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`)
                 ``\{ \mathbf{b}_{ih}^{\theta}, \mathbf{b}_{ih}^{\eta}, \mathbf{b}_{ih}^{h} \}``
    The initializers in `init_bias` are applied in the order they appear:
    the first function is used for $\mathbf{b}_{ih}^{\theta}$, the second for
      $\mathbf{b}_{ih}^{\eta}$, and the third for $\mathbf{b}_{ih}^{h}$.
  - `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`)
                 ``\{ \mathbf{b}_{hh}^{\theta}, \mathbf{b}_{hh}^{\eta} \}``
    The initializers in `init_recurrent_bias` are applied in the order they appear:
    the first function is used for $\mathbf{b}_{hh}^{\theta}$, and the second for
      $\mathbf{b}_{hh}^{\eta}$.
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct CFNCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
    train_state::TS
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_weight
    init_recurrent_weight
    init_state
    use_bias <: StaticBool
end

function CFNCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh;
        use_bias::BoolType=True(), train_state::BoolType=False(),
        init_bias=nothing, init_recurrent_bias=nothing, init_weight=nothing,
        init_recurrent_weight=nothing, init_state=zeros32)
    init_weight isa NTuple{3} || (init_weight = ntuple(Returns(init_weight), 3))
    init_recurrent_weight isa NTuple{2} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 2))
    init_bias isa NTuple{3} || (init_bias = ntuple(Returns(init_bias), 3))
    init_recurrent_bias isa NTuple{2} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 2))
    return CFNCell(static(train_state), activation, in_dims, out_dims,
        init_bias, init_recurrent_bias, init_weight,
        init_recurrent_weight, init_state, static(use_bias))
end

initialparameters(rng::AbstractRNG, cfn::CFNCell) = multi_initialparameters(rng, cfn)

initialstates(rng::AbstractRNG, ::CFNCell) = (rng=Utils.sample_replicate(rng),)

function parameterlength(cfn::CFNCell)
    return cfn.in_dims * cfn.out_dims * 3 + cfn.out_dims * cfn.out_dims * 2 +
           cfn.out_dims * 5
end

statelength(::CFNCell) = 1

function (cfn::CFNCell{False})(inp::AbstractMatrix, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = init_rnn_hidden_state(rng, cfn, inp)
    return cfn((inp, (hidden_state,)), ps, merge(st, (; rng)))
end

function (cfn::CFNCell{True})(inp::AbstractMatrix, ps, st::NamedTuple)
    hidden_state = init_trainable_rnn_hidden_state(ps.hidden_state, inp)
    return cfn((inp, (hidden_state,)), ps, st)
end

function (cfn::CFNCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(cfn, ps, st, inp, state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #gates
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    full_ghs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)
    gxs = multigate(full_gxs, Val(3))
    ghs = multigate(full_ghs, Val(2))
    #computation
    horizontal_gate = @. sigmoid_fast(gxs[1] + ghs[1])
    vertical_gate = @. sigmoid_fast(gxs[2] + ghs[2])
    new_state = @. horizontal_gate * tanh_fast(state) + vertical_gate * tanh_fast(gxs[3])

    return (new_state, (new_state,)), st
end

function Base.show(io::IO, r::CFNCell)
    print(io, "CFNCell($(r.in_dims) => $(r.out_dims)")
    (r.activation == identity) || print(io, ", $(r.activation)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end
