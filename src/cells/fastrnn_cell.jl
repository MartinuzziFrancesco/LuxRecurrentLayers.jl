#https://arxiv.org/abs/1901.02358
@doc raw"""
    FastRNNCell(in_dims => out_dims, [activation];
        use_bias=true, train_state=false, init_bias=nothing,
        init_recurrent_bias=nothing, init_weight=nothing,
        init_recurrent_weight=nothing, init_state=zeros32,
        init_alpha=-3.0, init_beta=3.0)

[Fast recurrent neural network cell](https://arxiv.org/abs/1901.02358).

# Arguments

  - `in_dims`: Input dimension
  - `out_dims`: Output (Hidden State & Memory) dimension
  - `activation`: activation function. Default is `tanh`

# Keyword arguments

  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.
    Default set to `false`.
  - `init_bias`: Initializer for bias $\mathbf{b}_{ih}$. If set to
    `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_recurrent_bias`: Initializer for hidden to hidden bias
    $\mathbf{b}_{hh}^z, \mathbf{b}_{hh}^h$. If set to
    `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_weight`: Initializer for weight $\mathbf{W}_{ih}$. If set to
    `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_recurrent_weight`: Initializer for recurrent weight $\mathbf{W}_{hh}$. If set to
    `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_state`: Initializer for hidden state. Default set to `zeros32`.
  - `init_alpha`: initializer for the $\alpha$ learnable parameter.
    Default is -3.0.
  - `init_beta`: initializer for the $\beta$ learnable parameter.
    Default is 3.0.

# Equations

```math
\begin{aligned}
    \tilde{\mathbf{h}}(t) &= \sigma\left( \mathbf{W}_{ih} \mathbf{x}(t) +
        \mathbf{b}_{ih} + \mathbf{W}_{hh} \mathbf{h}(t-1) + \mathbf{b}_{hh}
        \right), \\
    \mathbf{h}(t) &= \alpha \, \tilde{\mathbf{h}}(t) + \beta \,
        \mathbf{h}(t-1)
\end{aligned}
```

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

  - `weight_ih`: Concatenated weights to map from input to the hidden state $\mathbf{W}_{ih}$.
  - `weight_hh`: Concatenated weights to map from hidden to the hidden state $\mathbf{W}_{hh}$.
  - `bias_ih`: Bias vector for the input-hidden connection (not present if
      `use_bias=false`) $\mathbf{b}_{ih}$.
  - `bias_hh`: Bias vector for the hidden-hidden connection (not present if
      `use_bias=false`) $\mathbf{b}_{hh}$.
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
  - `alpha`: Learnable scalar to modulate candidate state.
  - `beta`: Learnable scalar to modulate previous state.

## States

  - `rng`: Controls the randomness (if any) in the initial state generation
"""
@concrete struct FastRNNCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
    train_state::TS
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_weight
    init_recurrent_weight
    init_state
    init_alpha
    init_beta
    use_bias <: StaticBool
end

function FastRNNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh_fast;
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_recurrent_bias=nothing, init_weight=nothing,
        init_recurrent_weight=nothing, init_state=zeros32,
        init_alpha=-3.0f0, init_beta=3.0f0)
    return FastRNNCell(static(train_state), activation, in_dims, out_dims,
        init_bias, init_weight, init_recurrent_weight, init_state, init_alpha, init_beta,
        static(use_bias))
end

function initialparameters(rng::AbstractRNG, fastrnn::FastRNNCell)
    ps = single_initialparameters(rng, fastrnn)
    # any additional trainable parameters
    alpha = fastrnn.init_alpha .* ones(1)
    beta = fastrnn.init_beta .* ones(1)
    ps = merge(ps, (; alpha, beta))
    return ps
end

function parameterlength(fastrnn::FastRNNCell)
    return fastrnn.in_dims * fastrnn.out_dims + fastrnn.out_dims * fastrnn.out_dims +
           fastrnn.out_dims * 2 + 2
end

function (fastrnn::FastRNNCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(fastrnn, ps, st, inp, state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #computation
    xs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    hs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)

    candidate_state = @. fastrnn.activation(xs + hs)
    new_state = @. ps.alpha * candidate_state + ps.beta * state
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, r::FastRNNCell)
    print(io, "FastRNNCell($(r.in_dims) => $(r.out_dims)")
    (r.activation == identity) || print(io, ", $(r.activation)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end

@doc raw"""
    FastGRNNCell(input_size => hidden_size, [activation];
        use_bias=true, train_state=false, init_bias=nothing,
        init_recurrent_bias=nothing, init_weight=nothing,
        init_recurrent_weight=nothing, init_state=zeros32,
        init_zeta=1.0, init_nu=4.0)

[Fast gated recurrent neural network cell](https://arxiv.org/abs/1901.02358).

# Arguments

  - `in_dims`: Input dimension
  - `out_dims`: Output (Hidden State & Memory) dimension
  - `activation`: activation function. Default is `tanh`

# Keyword arguments

  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.
    Default set to `false`.
  - `init_bias`: Initializer for input to hidden bias
    $\mathbf{b}_{ih}^z, \mathbf{b}_{ih}^h$. Must be a tuple containing
    2 functions, e.g., `(glorot_normal, kaiming_uniform)`.
    If a single function `fn` is provided, it is automatically expanded into a 2-element
    tuple (fn, fn). If set to `nothing`, weights are initialized from a uniform
    distribution within `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
    Default is `nothing`.
  - `init_recurrent_bias`: Initializer for hidden to hidden bias
    $\mathbf{b}_{hh}^z, \mathbf{b}_{hh}^h$. Must be a tuple containing
    2 functions, e.g., `(glorot_normal, kaiming_uniform)`.
    If a single function `fn` is provided, it is automatically expanded into a 2-element
    tuple (fn, fn). If set to `nothing`, weights are initialized from a uniform
    distribution within `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
    Default is `nothing`.
  - `init_weight`: Initializer for weight $\mathbf{W}_{ih}$. If set to
    `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_recurrent_weight`: Initializer for recurrent weight $\mathbf{W}_{hh}$. If set to
    `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_state`: Initializer for hidden state. Default set to `zeros32`.
  - `init_zeta`: initializer for the $\zeta$ learnable parameter.
    Default is 1.0.
  - `init_nu`: initializer for the $\nu$ learnable parameter.
    Default is 4.0.

# Equations
```math
\begin{aligned}
    \mathbf{z}(t) &= \sigma\left( \mathbf{W}_{ih} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{z} + \mathbf{W}_{hh} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{z} \right), \\
    \tilde{\mathbf{h}}(t) &= \tanh\left( \mathbf{W}_{ih} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{h} + \mathbf{W}_{hh} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{h} \right), \\
    \mathbf{h}(t) &= \left( \left( \zeta (1 - \mathbf{z}(t)) + \nu \right)
        \circ \tilde{\mathbf{h}}(t) \right) + \mathbf{z}(t) \circ
        \mathbf{h}(t-1)
\end{aligned}
```

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

  - `weight_ih`: Concatenated weights to map from input to the hidden state $\mathbf{W}_{ih}$.
  - `weight_hh`: Concatenated weights to map from hidden to the hidden state $\mathbf{W}_{hh}$.
  - `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`)
                 ``\{ \mathbf{b}_{ih}^z, \mathbf{b}_{ih}^h \}``
    The initializers in `init_bias` are applied in the order they appear:
    the first function is used for $\mathbf{b}_{ih}^z$, and the second for $\mathbf{b}_{ih}^h$.
  - `bias_hh`: Bias vector for the hidden-hidden connection (not present if `use_bias=false`)
                 ``\{ \mathbf{b}_{hh}^z, \mathbf{b}_{hh}^h \}``
    The initializers in `init_bias` are applied in the order they appear:
    the first function is used for $\mathbf{b}_{hh}^z$, and the second for $\mathbf{b}_{hh}^h$.
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
  - `zeta`: Learnable scalar to modulate candidate state.
  - `nu`: Learnable scalar to modulate previous state.

## States

  - `rng`: Controls the randomness (if any) in the initial state generation
"""
@concrete struct FastGRNNCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
    train_state::TS
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_weight
    init_recurrent_weight
    init_state
    init_zeta
    init_nu
    use_bias <: StaticBool
end

function FastGRNNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh_fast;
        use_bias::BoolType=True(), train_state::BoolType=False(),
        init_bias=nothing, init_recurrent_bias=nothing, init_weight=nothing,
        init_recurrent_weight=nothing, init_state=zeros32, init_zeta=1.0f0,
        init_nu=-4.0f0)
    init_bias isa NTuple{2} || (init_bias = ntuple(Returns(init_bias), 2))
    init_recurrent_bias isa NTuple{2} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 2))
    return FastGRNNCell(static(train_state), activation, in_dims, out_dims,
        init_bias, init_recurrent_bias, init_weight, init_recurrent_weight,
        init_state, init_zeta, init_nu, static(use_bias))
end

function initialparameters(rng::AbstractRNG, fastrnn::FastGRNNCell)
    #matrices
    weight_ih = init_rnn_weight(
        rng, fastrnn.init_weight, fastrnn.out_dims, (fastrnn.out_dims, fastrnn.in_dims))
    weight_hh = init_rnn_weight(
        rng, fastrnn.init_recurrent_weight, fastrnn.out_dims,
        (fastrnn.out_dims, fastrnn.out_dims))
    ps = (; weight_ih, weight_hh)
    #biases
    if has_bias(fastrnn)
        bias_ih = multi_bias(rng, fastrnn.init_bias, fastrnn.out_dims, fastrnn.out_dims)
        bias_hh = multi_bias(rng, fastrnn.init_bias, fastrnn.out_dims, fastrnn.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    # trainable state or memory
    has_train_state(fastrnn) &&
        (ps = merge(ps, (hidden_state=fastrnn.init_state(rng, fastrnn.out_dims),)))
    # any additional trainable parameters
    zeta = fastrnn.init_zeta .* ones(1)
    nu = fastrnn.init_nu .* ones(1)
    ps = merge(ps, (; zeta, nu))
    return ps
end

function parameterlength(fastrnn::FastGRNNCell)
    return fastrnn.in_dims * fastrnn.out_dims + fastrnn.out_dims * fastrnn.out_dims +
           fastrnn.out_dims * 2 + 2
end

function (fastrnn::FastGRNNCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(fastrnn, ps, st, inp, state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_ihs = multigate(bias_ih, Val(2))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    bias_hhs = multigate(bias_hh, Val(2))
    #computation
    xsz = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ihs[1])
    xsh = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ihs[2])
    hsz = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hhs[1])
    hsh = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hhs[2])

    gate = @. fastrnn.activation(xsz + hsz)
    candidate_state = @. tanh_fast(xsh + hsh)
    ones_arr = ones(eltype(gate), size(gate))
    new_state = @. (ps.zeta * (ones_arr - gate) + ps.nu) * candidate_state +
                   gate * state
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, r::FastGRNNCell)
    print(io, "FastGRNNCell($(r.in_dims) => $(r.out_dims)")
    (r.activation == identity) || print(io, ", $(r.activation)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end
