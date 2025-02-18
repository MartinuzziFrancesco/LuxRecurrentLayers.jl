#https://arxiv.org/abs/1902.09689
@doc raw"""
    AntisymmetricRNNCell(input_size => hidden_size, [activation];
        init_kernel = glorot_uniform,
        init_recurrent_kernel = glorot_uniform,
        bias = true, epsilon=1.0)


[Antisymmetric recurrent cell](https://arxiv.org/abs/1902.09689).
See [`AntisymmetricRNN`](@ref) for a layer that processes entire sequences.

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `activation`: activation function. Default is `tanh`

# Keyword arguments

- `init_kernel`: initializer for the input to hidden weights
- `init_recurrent_kernel`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`
- `epsilon`: step size. Default is 1.0.
- `gamma`: strength of diffusion. Default is 0.0

# Equations
```math
h_t = h_{t-1} + \epsilon \tanh \left( (W_h - W_h^T - \gamma I) h_{t-1} + V_h x_t + b_h \right),
```

# Forward

    asymrnncell(inp, state)
    asymrnncell(inp)

## Arguments
- `inp`: The input to the asymrnncell. It should be a vector of size `input_size`
  or a matrix of size `input_size x batch_size`.
- `state`: The hidden state of the AntisymmetricRNNCell. It should be a vector of size
  `hidden_size` or a matrix of size `hidden_size x batch_size`.
  If not provided, it is assumed to be a vector of zeros,
  initialized by [`Flux.initialstates`](@extref).

## Returns
- A tuple `(output, state)`, where both elements are given by the updated state
  `new_state`, a tensor of size `hidden_size` or `hidden_size x batch_size`.
"""
@concrete struct AntisymmetricRNNCell <: AbstractRecurrentCell
    train_state <: StaticBool
    activation
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_weight
    init_recurrent_weight
    init_state
    use_bias <: StaticBool
    epsilon
    gamma
end

function AntisymmetricRNNCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh;
        use_bias::BoolType=True(), train_state::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing, init_state=zeros32)
    return AntisymmetricRNNCell(static(train_state), activation, in_dims, out_dims,
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
        (inp, (hidden_state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    y, hidden_stateₙ = match_eltype(asymrnn, ps, st, inp, hidden_state)

    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    z₁ = fused_dense_bias_activation(identity, ps.weight_hh, hidden_stateₙ, bias_hh)

    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    z₂ = fused_dense_bias_activation(identity, ps.weight_ih, y, bias_ih)

    hₙ = fast_activation!!(asymrnn.activation, z₁ .+ z₂)
    return (hₙ, (hₙ,)), st
end

function Base.show(io::IO, r::AntisymmetricRNNCell)
    print(io, "AntisymmetricRNNCell($(r.in_dims) => $(r.out_dims)")
    (r.activation == identity) || print(io, ", $(r.activation)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end

function compute_asym_recurrent(Wh, gamma)
    return Wh .- transpose(Wh) .- gamma .* Matrix{eltype(Wh)}(I, size(Wh, 1), size(Wh, 1))
end
