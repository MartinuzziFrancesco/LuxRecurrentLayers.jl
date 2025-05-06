#https://arxiv.org/abs/1901.02358
@doc raw"""
    FastRNNCell(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Fast recurrent neural network cell](https://arxiv.org/abs/1901.02358).

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: the activation function, defaults to `tanh_fast`.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.

# Equations
```math
\begin{aligned}
\tilde{h}_t &= \sigma(W_h x_t + U_h h_{t-1} + b), \\
h_t &= \alpha \tilde{h}_t + \beta h_{t-1}
\end{aligned}
```

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
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing, init_state=zeros32,
        init_alpha=-3.0f0, init_beta=3.0f0)
    return FastRNNCell(static(train_state), activation, in_dims, out_dims,
        init_bias, init_weight, init_recurrent_weight, init_state, init_alpha, init_beta,
        static(use_bias))
end

function initialparameters(rng::AbstractRNG, fastrnn::FastRNNCell)
    #matrices
    weight_ih = init_rnn_weight(
        rng, fastrnn.init_weight, fastrnn.out_dims, (fastrnn.out_dims, fastrnn.in_dims))
    weight_hh = init_rnn_weight(
        rng, fastrnn.init_recurrent_weight, fastrnn.out_dims,
        (fastrnn.out_dims, fastrnn.out_dims))
    ps = (; weight_ih, weight_hh)
    #biases
    if has_bias(fastrnn)
        bias_ih = init_rnn_bias(rng, fastrnn.init_bias, fastrnn.out_dims, fastrnn.out_dims)
        bias_hh = init_rnn_bias(rng, fastrnn.init_bias, fastrnn.out_dims, fastrnn.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    # trainable state or memory
    has_train_state(fastrnn) &&
        (ps = merge(ps, (hidden_state=fastrnn.init_state(rng, fastrnn.out_dims),)))
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
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Fast gated recurrent neural network cell](https://arxiv.org/abs/1901.02358).

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer.
- `activation`: the activation function, defaults to `tanh_fast`.

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights.
    Default is `glorot_uniform`.
- `init_recurrent_kernel`: initializer for the hidden to hidden weights.
    Default is `glorot_uniform`.
- `bias`: include a bias or not. Default is `true`.

# Equations
```math
\begin{aligned}
z_t &= \sigma(W x_t + U h_{t-1} + b_z), \\
\tilde{h}_t &= \tanh(W x_t + U h_{t-1} + b_h), \\
h_t &= \big((\zeta (1 - z_t) + \nu) \odot \tilde{h}_t\big) + z_t \odot h_{t-1}
\end{aligned}
```

"""
@concrete struct FastGRNNCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
    train_state::TS
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
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
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing, init_state=zeros32,
        init_zeta=1.0f0, init_nu=-4.0f0)
    return FastGRNNCell(static(train_state), activation, in_dims, out_dims,
        init_bias, init_weight, init_recurrent_weight, init_state, init_zeta, init_nu,
        static(use_bias))
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
        bias_ih = init_rnn_bias(rng, fastrnn.init_bias, fastrnn.out_dims, fastrnn.out_dims)
        bias_hh = init_rnn_bias(rng, fastrnn.init_bias, fastrnn.out_dims, fastrnn.out_dims)
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
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #computation
    xs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    hs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)

    gate = @. fastrnn.activation(xs + hs)
    candidate_state = @. tanh_fast(xs + hs)
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
