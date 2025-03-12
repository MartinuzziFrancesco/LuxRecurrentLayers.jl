#https://arxiv.org/abs/1612.06212
@doc raw"""
    CFNCell(in_dims => out_dims;
        init_weight = nothing, init_recurrent_weight = nothing,
        bias = true)


[Chaos free network unit](https://arxiv.org/abs/1612.06212).

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer

# Keyword arguments

- `init_weight`: initializer for the input to hidden weights
- `init_recurrent_weight`: initializer for the hidden to hidden weights
- `init_bias`: initializer for the input to hidden bias
- `init_recurrent_bias`: initializer for the hidden to hidden bias

```math
\begin{aligned}
    h_t &= \theta_t \odot \tanh(h_{t-1}) + \eta_t \odot \tanh(W x_t), \\
    \theta_t &:= \sigma (U_\theta h_{t-1} + V_\theta x_t + b_\theta), \\
    \eta_t &:= \sigma (U_\eta h_{t-1} + V_\eta x_t + b_\eta).
\end{aligned}
```

"""
@concrete struct CFNCell <: AbstractRecurrentCell
    train_state <: StaticBool
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

function initialparameters(rng::AbstractRNG, cfn::CFNCell)
    weight_ih = multi_inits(rng, cfn.init_weight, cfn.out_dims, cfn.in_dims)
    weight_hh = multi_inits(rng, cfn.init_recurrent_weight, cfn.out_dims, cfn.out_dims)
    ps = (; weight_ih, weight_hh)
    if has_bias(cfn)
        bias_ih = multi_bias(rng, cfn.init_bias, cfn.out_dims, cfn.out_dims)
        bias_hh = multi_bias(rng, cfn.init_recurrent_bias, cfn.out_dims, cfn.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    has_train_state(cfn) &&
        (ps = merge(ps, (hidden_state=cfn.init_state(rng, cfn.out_dims),)))
    return ps
end

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
