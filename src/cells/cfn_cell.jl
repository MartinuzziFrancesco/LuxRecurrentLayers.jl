#https://arxiv.org/abs/1902.09689
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
- `bias`: include a bias or not. Default is `true`

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
    init_weight
    init_recurrent_weight
    init_state
    use_bias <: StaticBool
end

function CFNCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh;
        use_bias::BoolType=True(), train_state::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing, init_state=zeros32)
    return CFNCell(static(train_state), activation, in_dims, out_dims,
        init_bias, init_weight, init_recurrent_weight, init_state, static(use_bias))
end

function initialparameters(rng::AbstractRNG, asymrnn::AntisymmetricRNNCell)
    weight_ih = init_rnn_weight(
        rng, asymrnn.init_weight, asymrnn.out_dims, (asymrnn.out_dims, asymrnn.in_dims))
    weight_hh = init_rnn_weight(
        rng, asymrnn.init_recurrent_weight, asymrnn.out_dims, (asymrnn.out_dims, asymrnn.out_dims))
    ps = (; weight_ih, weight_hh)
    if has_bias(asymrnn)
        bias_ih = init_rnn_bias(rng, asymrnn.init_bias, asymrnn.out_dims, asymrnn.out_dims)
        bias_hh = init_rnn_bias(rng, asymrnn.init_bias, asymrnn.out_dims, asymrnn.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    has_train_state(asymrnn) &&
        (ps = merge(ps, (hidden_state=asymrnn.init_state(rng, asymrnn.out_dims),)))
    return ps
end

initialstates(rng::AbstractRNG, ::AntisymmetricRNNCell) = (rng=Utils.sample_replicate(rng),)

function parameterlength(asymrnn::AntisymmetricRNNCell)
    return asymrnn.in_dims * asymrnn.out_dims + asymrnn.out_dims * asymrnn.out_dims + asymrnn.out_dims 
end

statelength(::AntisymmetricRNNCell) = 1

function (asymrnn::AntisymmetricRNNCell{False})(inp::AbstractMatrix, ps, st::NamedTuple)
    rng = replicate(st.rng)
    hidden_state = init_rnn_hidden_state(rng, asymrnn, inp)
    return asymrnn((inp, (hidden_state,)), ps, merge(st, (; rng)))
end

function (asymrnn::AntisymmetricRNNCell{True})(inp::AbstractMatrix, ps, st::NamedTuple)
    hidden_state = init_trainable_rnn_hidden_state(ps.hidden_state, inp)
    return asymrnn((inp, (hidden_state,)), ps, st)
end

function (asymrnn::AntisymmetricRNNCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(asymrnn, ps, st, inp, state)
    #input linear transform
    bias_ih = safe_getproperty(ps, Val(:bias_ih))

    return (new_state, (new_state,)), st
end

function Base.show(io::IO, r::AntisymmetricRNNCell)
    print(io, "AntisymmetricRNNCell($(r.in_dims) => $(r.out_dims)")
    (r.activation == identity) || print(io, ", $(r.activation)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end