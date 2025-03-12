#https://arxiv.org/abs/1902.09689
@doc raw"""
    AntisymmetricRNNCell(in_dims => out_dims, [activation];
        init_weight = nothing, init_recurrent_weight = nothing,
        bias = true, epsilon=1.0)


[Antisymmetric recurrent cell](https://arxiv.org/abs/1902.09689).

# Arguments

- `input_size => hidden_size`: input and inner dimension of the layer
- `activation`: activation function. Default is `tanh`

# Keyword arguments

- `init_weight`: initializer for the input to hidden weights
- `init_recurrent_weight`: initializer for the hidden to hidden weights
- `bias`: include a bias or not. Default is `true`
- `epsilon`: step size. Default is 1.0.
- `gamma`: strength of diffusion. Default is 0.0

# Equations
```math
h_t = h_{t-1} + \epsilon \tanh \left( (W_h - W_h^T - \gamma I) h_{t-1} + V_h x_t + b_h \right),
```

"""
@concrete struct AntisymmetricRNNCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
    train_state::TS
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

function AntisymmetricRNNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh;
        use_bias::BoolType=True(), train_state::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing, init_state=zeros32,
        epsilon=1.0f0, gamma=0.0f0)
    return AntisymmetricRNNCell(static(train_state), activation, in_dims, out_dims,
        init_bias, init_weight, init_recurrent_weight, init_state, static(use_bias), epsilon, gamma)
end

function initialparameters(rng::AbstractRNG, asymrnn::AntisymmetricRNNCell)
    weight_ih = init_rnn_weight(
        rng, asymrnn.init_weight, asymrnn.out_dims, (asymrnn.out_dims, asymrnn.in_dims))
    weight_hh = init_rnn_weight(
        rng, asymrnn.init_recurrent_weight, asymrnn.out_dims,
        (asymrnn.out_dims, asymrnn.out_dims))
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

function parameterlength(asymrnn::AntisymmetricRNNCell)
    return asymrnn.in_dims * asymrnn.out_dims + asymrnn.out_dims * asymrnn.out_dims +
           asymrnn.out_dims
end

function (asymrnn::AntisymmetricRNNCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(asymrnn, ps, st, inp, state)
    #input linear transform
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    linear_input = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #recurrent linear transform
    asym_weight_hh = compute_asym_recurrent(ps.weight_hh, asymrnn.gamma)
    linear_recur = fused_dense_bias_activation(
        identity, asym_weight_hh, matched_state, bias_hh)
    #putting it all together
    half_new_state = fast_activation!!(asymrnn.activation, linear_input .+ linear_recur)
    new_state = state .+ asymrnn.epsilon .* half_new_state
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, r::AntisymmetricRNNCell)
    print(io, "AntisymmetricRNNCell($(r.in_dims) => $(r.out_dims)")
    (r.activation == identity) || print(io, ", $(r.activation)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end

function compute_asym_recurrent(weight_hh, gamma)
    gamma = eltype(weight_hh)(gamma)
    return weight_hh .- transpose(weight_hh) .-
           gamma .* Matrix{eltype(weight_hh)}(I, size(weight_hh, 1), size(weight_hh, 1))
end
