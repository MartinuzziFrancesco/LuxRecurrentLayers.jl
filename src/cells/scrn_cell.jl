#https://arxiv.org/pdf/1412.7753

@doc raw"""
    SCRNCell(input_size => hidden_size;)

[Structurally contraint recurrent unit](https://arxiv.org/pdf/1412.7753).


# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights


# Equations
```math
\begin{aligned}
s_t &= (1 - \alpha) W_s x_t + \alpha s_{t-1}, \\
h_t &= \sigma(W_h s_t + U_h h_{t-1} + b_h), \\
y_t &= f(U_y h_t + W_y s_t)
\end{aligned}
```

"""
@concrete struct SCRNCell <: AbstractRecurrentCell
    train_state <: StaticBool
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_weight
    init_recurrent_weight
    init_context_weight
    init_state
    use_bias <: StaticBool
end

function SCRNCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing, init_context_weight=nothing, init_state=zeros32)
    init_weight isa NTuple{2} || (init_weight = ntuple(Returns(init_weight), 2))
    init_recurrent_weight isa NTuple{2} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 2))
    init_context_weight isa NTuple{2} ||
        (init_context_weight = ntuple(Returns(init_context_weight), 2))
    init_bias isa NTuple{2} || (init_bias = ntuple(Returns(init_bias), 2))
    return SCRNCell(static(train_state), in_dims, out_dims,
        init_bias, init_weight, init_recurrent_weight, init_context_weight, init_state, static(use_bias))
end

function initialparameters(rng::AbstractRNG, scrn::SCRNCell)
    weight_ih = multi_inits(rng, scrn.init_weight, scrn.out_dims, scrn.in_dims)
    weight_hh = multi_inits(rng, scrn.init_recurrent_weight, scrn.out_dims, scrn.out_dims)
    weight_ch = multi_inits(rng, scrn.init_context_weight, scrn.out_dims, scrn.out_dims)
    ps = (; weight_ih, weight_hh, weight_ch)
    if has_bias(scrn)
        bias_ih = multi_bias(rng, scrn.init_bias, scrn.out_dims, scrn.out_dims)
        bias_hh = multi_bias(rng, scrn.init_bias, scrn.out_dims, scrn.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    has_train_state(scrn) &&
        (ps = merge(ps, (hidden_state=scrn.init_state(rng, scrn.out_dims),)))
    ps = merge(ps, (alpha=eltype(weight_ih)(0.0f0),))
    return ps
end

initialstates(rng::AbstractRNG, ::SCRNCell) = (rng=Utils.sample_replicate(rng),)

function parameterlength(scrn::SCRNCell)
    return scrn.in_dims * scrn.out_dims * 2 + scrn.out_dims * scrn.out_dims * 4 + scrn.out_dims * 2 + 1 
end

statelength(::SCRNCell) = 1

function (scrn::SCRNCell{False})(inp::AbstractMatrix, ps, st::NamedTuple)
    rng = replicate(st.rng)
    state = init_rnn_hidden_state(rng, scrn, inp)
    c_state = init_rnn_hidden_state(rng, scrn, inp)
    return scrn((inp, (state, c_state)), ps, merge(st, (; rng)))
end

function (scrn::SCRNCell{True})(inp::AbstractMatrix, ps, st::NamedTuple)
    state = init_trainable_rnn_hidden_state(ps.hidden_state, inp)
    c_state = init_rnn_hidden_state(rng, scrn, inp)
    return scrn((inp, (state, c_state)), ps, st)
end

function (scrn::SCRNCell)(
        (inp, (state, c_state))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix},},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state, matched_cstate = match_eltype(scrn, ps, st, inp, state, c_state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #gates
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    full_gcs = fused_dense_bias_activation(identity, ps.weight_ch, matched_state, bias_hh)
    gxs = multigate(full_gxs, Val(2))
    ghs =  multigate(ps.weight_hh, Val(2))
    gcs = multigate(full_gcs, Val(2))
    #computation
    new_cstate = (eltype(ps.weight_hh)(1.0f0) .- ps.alpha) .* gxs[1] .+ ps.alpha .* c_state
    hidden_layer = sigmoid_fast.(gxs[2] .+ ghs[1] * matched_state .+ gcs[1])
    new_state = tanh_fast.(ghs[2] * hidden_layer .+ gcs[2])
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, r::SCRNCell)
    print(io, "SCRNCell($(r.in_dims) => $(r.out_dims)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end