#https://arxiv.org/pdf/1611.09913
@doc raw"""
    UGRNNCell(in_dims => out_dims;
        use_bias=true, use_recurrent_bias=true, train_state=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)

[Update gate recurrent unit](https://arxiv.org/pdf/1611.09913).

## Equations
```math
\begin{aligned}
    \mathbf{c}(t) &= \tanh\left(
        \mathbf{W}_{ih}^{c} \mathbf{x}(t) + \mathbf{b}_{ih}^{c} +
        \mathbf{W}_{hh}^{c} \mathbf{h}(t-1) + \mathbf{b}_{hh}^{c} \right), \\
    \mathbf{g}(t) &= \sigma\left(
        \mathbf{W}_{ih}^{g} \mathbf{x}(t) + \mathbf{b}_{ih}^{g} +
        \mathbf{W}_{hh}^{g} \mathbf{h}(t-1) + \mathbf{b}_{hh}^{g} \right), \\
    \mathbf{h}(t) &= \mathbf{g}(t) \circ \mathbf{h}(t-1) +
        \left(1 - \mathbf{g}(t)\right) \circ \mathbf{c}(t)
\end{aligned}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension

## Keyword Arguments

  - `use_bias`: Flag to use bias $\mathbf{b}_{ih}$ in the computation.
    Default set to `true`.
  - `use_recurrent_bias`: Flag to use recurrent bias $\mathbf{b}_{hh}$ in the computation.
    Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.
    Default set to `false`.
  - `init_bias`: Initializer for input-to-hidden biases
    $\mathbf{b}_{ih}^{c}, \mathbf{b}_{ih}^{g}$.
    Must be a tuple of 2 functions. If a single function is provided, it is
    expanded to 2 copies. If set to `nothing`, biases are initialized from a
    uniform distribution in `[-bound, bound]` where
    `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_recurrent_bias`: Initializer for hidden-to-hidden biases
    $\mathbf{b}_{hh}^{c}, \mathbf{b}_{hh}^{g}$.
    Must be a tuple of 2 functions. If a single function is provided,
    it is expanded to 2 copies. If set to `nothing`, biases are initialized
    from a uniform distribution in `[-bound, bound]` where
    `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_weight`: Initializer for input-to-hidden weights
    $\mathbf{W}_{ih}^{c}, \mathbf{W}_{ih}^{g}$.
    Must be a tuple of 2 functions. If a single function is provided, it is
    expanded to 2 copies. If set to `nothing`, weights are initialized from a
    uniform distribution in `[-bound, bound]` where
    `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_recurrent_weight`: Initializer for hidden-to-hidden weights
    $\mathbf{W}_{hh}^{c}, \mathbf{W}_{hh}^{g}$.
    Must be a tuple of 2 functions. If a single function is provided, it is
    expanded to 2 copies. If set to `nothing`, weights are initialized from a
    uniform distribution in `[-bound, bound]` where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
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

  - `weight_ih`: Concatenated input-to-hidden weights
    ``\{ \mathbf{W}_{ih}^{c}, \mathbf{W}_{ih}^{g} \}``
    The functions from `init_weight` are applied in order:
    the first initializes $\mathbf{W}_{ih}^{c}$, the second $\mathbf{W}_{ih}^{g}$.
  - `weight_hh`: Concatenated hidden-to-hidden weights
    ``\{ \mathbf{W}_{hh}^{c}, \mathbf{W}_{hh}^{g} \}``
    The functions from `init_recurrent_weight` are applied in order:
    the first initializes $\mathbf{W}_{hh}^{c}$, the second $\mathbf{W}_{hh}^{g}$.
  - `bias_ih`: Concatenated input-to-hidden biases (if `use_bias=true`)
    ``\{ \mathbf{b}_{ih}^{c}, \mathbf{b}_{ih}^{g} \}``
    The functions from `init_bias` are applied in order:
    the first initializes $\mathbf{b}_{ih}^{c}$, the second $\mathbf{b}_{ih}^{g}$.
  - `bias_hh`: Concatenated hidden-to-hidden biases (if `use_recurrent_bias=true`)
    ``\{ \mathbf{b}_{hh}^{c}, \mathbf{b}_{hh}^{g} \}``
    The functions from `init_recurrent_bias` are applied in order:
    the first initializes $\mathbf{b}_{hh}^{c}$, the second $\mathbf{b}_{hh}^{g}$.
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct UGRNNCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
    train_state::TS
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_weight
    init_recurrent_weight
    init_state
    use_bias <: StaticBool
    use_recurrent_bias <: StaticBool
end

function UGRNNCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), use_recurrent_bias::BoolType=True(),
        train_state::BoolType=False(), init_bias=nothing,
        init_recurrent_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)
    init_weight isa NTuple{2} || (init_weight = ntuple(Returns(init_weight), 2))
    init_recurrent_weight isa NTuple{2} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 2))
    init_bias isa NTuple{2} || (init_bias = ntuple(Returns(init_bias), 2))
    init_recurrent_bias isa NTuple{2} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 2))
    return UGRNNCell(
        static(train_state), in_dims, out_dims, init_bias, init_recurrent_bias,
        init_weight, init_recurrent_weight, init_state, static(use_bias),
        static(use_recurrent_bias))
end

initialparameters(rng::AbstractRNG, ugrnn::UGRNNCell) = multi_initialparameters(rng, ugrnn)

function (ugrnn::UGRNNCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    matched_inp, matched_state = match_eltype(ugrnn, ps, st, inp, state)
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    t_ones = one(eltype(matched_inp))
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    full_ghs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)
    gxs = multigate(full_gxs, Val(2))
    ghs = multigate(full_ghs, Val(2))
    candidate_state = @. tanh_fast(gxs[1] + ghs[1])
    update_gate = @. sigmoid_fast(gxs[2] + ghs[2])
    new_state = @. update_gate * matched_state + (t_ones - update_gate) * candidate_state
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, ugrnn::UGRNNCell)
    print(io, "UGRNNCell($(ugrnn.in_dims) => $(ugrnn.out_dims)")
    has_bias(ugrnn) || print(io, ", use_bias=false")
    has_train_state(ugrnn) && print(io, ", train_state=true")
    print(io, ")")
end
