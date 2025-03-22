#https://arxiv.org/abs/2010.00951
@doc raw"""
    coRNNCell(input_size => hidden_size, [dt];
        gamma=0.0, epsilon=0.0,
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true)

[Coupled oscillatory recurrent neural unit](https://arxiv.org/abs/2010.00951).

# Equations

```math
\begin{aligned}
\mathbf{y}_n &= y_{n-1} + \Delta t \mathbf{z}_n, \\
\mathbf{z}_n &= z_{n-1} + \Delta t \sigma \left( \mathbf{W} y_{n-1} +
    \mathcal{W} z_{n-1} + \mathbf{V} u_n + \mathbf{b} \right) -
    \Delta t \gamma y_{n-1} - \Delta t \epsilon \mathbf{z}_n,
\end{aligned}
```

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `dt`: time step. Default is 1.0.

# Keyword arguments

- `gamma`: damping for state. Default is 0.0.
- `epsilon`: damping for candidate state. Default is 0.0.
- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`


"""
@concrete struct coRNNCell{TS <: StaticBool, TM <: StaticBool} <:
                 AbstractDoubleRecurrentCell{TS, TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_weight
    init_recurrent_weight
    init_state
    use_bias <: StaticBool
    dt
    gamma
    epsilon
end

function coRNNCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing, init_state=zeros32,
        dt::Number=1.0f0, gamma::Number=0.0f0, epsilon::Number=0.0f0)
    return coRNNCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_weight, init_recurrent_weight, init_state, static(use_bias),
        dt, gamma, epsilon)
end

function initialparameters(rng::AbstractRNG, cornn::coRNNCell)
    weight_ih = init_rnn_weight(
        rng, cornn.init_weight, cornn.out_dims, (cornn.out_dims, cornn.in_dims))
    weight_hh = init_rnn_weight(
        rng, cornn.init_recurrent_weight, cornn.out_dims, (cornn.out_dims, cornn.out_dims))
    weight_zh = init_rnn_weight(
        rng, cornn.init_recurrent_weight, cornn.out_dims, (cornn.out_dims, cornn.out_dims))
    ps = (; weight_ih, weight_hh, weight_zh)
    if has_bias(cornn)
        bias_ih = init_rnn_bias(rng, cornn.init_bias, cornn.out_dims, cornn.out_dims)
        bias_hh = init_rnn_bias(rng, cornn.init_bias, cornn.out_dims, cornn.out_dims)
        bias_zh = init_rnn_bias(rng, cornn.init_bias, cornn.out_dims, cornn.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh, bias_zh))
    end
    has_train_state(cornn) &&
        (ps = merge(ps, (hidden_state=cornn.init_state(rng, cornn.out_dims),)))
    return ps
end

function parameterlength(cornn::coRNNCell)
    return cornn.in_dims * cornn.out_dims + cornn.out_dims * cornn.out_dims * 2 +
           cornn.out_dims * 3
end

function (cornn::coRNNCell)(
        (inp,
            (state, c_state))::Tuple{
            <:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state, matched_cstate = match_eltype(
        cornn, ps, st, inp, state, c_state)
    dt, gamma, epsilon = cornn.dt, cornn.gamma, cornn.epsilon
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    bias_zh = safe_getproperty(ps, Val(:bias_zh))
    #computation
    xs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    hs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)
    zs = fused_dense_bias_activation(identity, ps.weight_zh, matched_cstate, bias_zh)
    pre_act = @. xs + hs + zs
    new_cstate = @. c_state + dt * (tanh_fast(pre_act) - gamma * state - epsilon * c_state)
    new_state = @. state + dt * new_cstate
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, r::coRNNCell)
    print(io, "coRNNCell($(r.in_dims) => $(r.out_dims)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end
