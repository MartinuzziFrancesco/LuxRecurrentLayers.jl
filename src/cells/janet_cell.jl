#https://arxiv.org/abs/1804.04849
@doc raw"""
    JANETCell(in_dims => out_dims;
        use_bias=true, use_recurrent_bias=true, use_integration_bias=false,
        train_state=false, train_memory=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_integration_bias=nothing, init_weight=nothing,
        init_recurrent_weight=nothing, init_state=zeros32, init_memory=zeros32,
        beta=1.0, integration_mode=AdditiveIntegration())

[Just another network unit](https://arxiv.org/abs/1804.04849).

## Equations
```math
\begin{aligned}
    \mathbf{s}(t) &= \mathbf{W}_{ih}^{f} \mathbf{x}(t) + \mathbf{b}_{ih}^{f} +
        \mathbf{W}_{hh}^{f} \mathbf{h}(t-1) + \mathbf{b}_{hh}^{f}, \\
    \tilde{\mathbf{c}}(t) &= \tanh\left( \mathbf{W}_{ih}^{c} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{c} + \mathbf{W}_{hh}^{c} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{c} \right), \\
    \mathbf{c}(t) &= \sigma(\mathbf{s}(t)) \circ \mathbf{c}(t-1) + \left(1 -
        \sigma(\mathbf{s}(t) - \beta)\right) \circ \tilde{\mathbf{c}}(t), \\
    \mathbf{h}(t) &= \mathbf{c}(t)
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
  - `use_integration_bias`: Flag to use integration bias $\mathbf{b}_{mi}$ in the computation.
    This bias is only useful for multiplicative integration. Check the docs page on multiplicative
    integration for more details. Default set to `false`.
  - `train_state`: Flag to set the initial hidden state as trainable. Default set to `false`.
  - `train_memory`: Flag to set the initial memory state as trainable. Default set to `false`.
  - `init_bias`: Initializer for input-to-hidden biases
    $\mathbf{b}_{ih}^{f}$ and $\mathbf{b}_{ih}^{c}$.
    Must be a tuple of 2 functions, e.g., `(glorot_uniform, kaiming_uniform)`.
    If a single function `fn` is provided, it is expanded to `(fn, fn)`.
    If set to `nothing`, each bias is initialized from a uniform distribution
    within `[-bound, bound]` where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_recurrent_bias`: Initializer for hidden-to-hidden biases
    $\mathbf{b}_{hh}^{f}$ and $\mathbf{b}_{hh}^{c}$.
    Must be a tuple of 2 functions, e.g., `(glorot_uniform, kaiming_uniform)`.
    If a single function `fn` is provided, it is expanded to `(fn, fn)`.
    If set to `nothing`, each bias is initialized from a uniform distribution
    within `[-bound, bound]` where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_integration_bias`: Initializer for integration bias $\mathbf{b}_{mi}$. If set to
    `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_weight`: Initializer for input-to-hidden weights
    $\mathbf{W}_{ih}^{f}$ and $\mathbf{W}_{ih}^{c}$.
    Must be a tuple of 2 functions, e.g., `(glorot_uniform, kaiming_uniform)`.
    If a single function `fn` is provided, it is expanded to `(fn, fn)`.
    If set to `nothing`, each weight is initialized from a uniform distribution
    within `[-bound, bound]` where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_recurrent_weight`: Initializer for hidden-to-hidden weights
    $\mathbf{W}_{hh}^{f}$ and $\mathbf{W}_{hh}^{c}$.
    Must be a tuple of 2 functions, e.g., `(glorot_uniform, kaiming_uniform)`.
    If a single function `fn` is provided, it is expanded to `(fn, fn)`.
    If set to `nothing`, each weight is initialized from a uniform distribution
    within `[-bound, bound]` where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_state`: Initializer for hidden state. Default set to `zeros32`.
  - `init_memory`: Initializer for memory. Default set to `zeros32`.
  - `beta`: Control parameter over the input data flow. Default is `1.0`.
  - `integration_mode`: integration type for the recurrent forward pass.
    Default is [`AdditiveIntegration()`](@ref).

## Inputs

  - Case 1a: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `false`, `train_memory` is set to `false` - Creates a hidden state using
             `init_state`, hidden memory using `init_memory` and proceeds to Case 2.
  - Case 1b: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `true`, `train_memory` is set to `false` - Repeats `hidden_state` vector
             from the parameters to match the shape of `x`, creates hidden memory using
             `init_memory` and proceeds to Case 2.
  - Case 1c: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `false`, `train_memory` is set to `true` - Creates a hidden state using
             `init_state`, repeats the memory vector from parameters to match the shape of
             `x` and proceeds to Case 2.
  - Case 1d: Only a single input `x` of shape `(in_dims, batch_size)`, `train_state` is set
             to `true`, `train_memory` is set to `true` - Repeats the hidden state and
             memory vectors from the parameters to match the shape of  `x` and proceeds to
             Case 2.
  - Case 2: Tuple `(x, (h, c))` is provided, then the output and a tuple containing the
            updated hidden state and memory is returned.

## Returns

  - Tuple Containing

      + Output ``h_{new}`` of shape `(out_dims, batch_size)`
      + Tuple containing new hidden state ``h_{new}`` and new memory ``c_{new}``

  - Updated model state

## Parameters

  - `weight_ih`: Concatenated weights mapping from input to hidden units
    ``\{ \mathbf{W}_{ih}^{f}, \mathbf{W}_{ih}^{c} \}``
    The functions provided in `init_weight` are applied in order:
    the first function initializes $\mathbf{W}_{ih}^{f}$, the second
    initializes $\mathbf{W}_{ih}^{c}$.
  - `weight_hh`: Concatenated weights mapping from hidden state to hidden units
    ``\{ \mathbf{W}_{hh}^{f}, \mathbf{W}_{hh}^{c} \}``
    The functions provided in `init_recurrent_weight` are applied in order:
    the first function initializes $\mathbf{W}_{hh}^{f}$, the second
    initializes $\mathbf{W}_{hh}^{c}$.
  - `bias_ih`: Concatenated input-to-hidden bias vectors (if `use_bias=true`)
    ``\{ \mathbf{b}_{ih}^{f}, \mathbf{b}_{ih}^{c} \}``
    The functions provided in `init_bias` are applied in order:
    the first function initializes $\mathbf{b}_{ih}^{f}$, the second
    initializes $\mathbf{b}_{ih}^{c}$.
  - `bias_hh`: Concatenated hidden-to-hidden bias vectors (if `use_bias=true`)
    ``\{ \mathbf{b}_{hh}^{f}, \mathbf{b}_{hh}^{c} \}``
    The functions provided in `init_recurrent_bias` are applied in order:
    the first function initializes $\mathbf{b}_{hh}^{f}$, the second initializes
    $\mathbf{b}_{hh}^{c}$.
  - `bias_mi`: Bias vector for the integration connection (not present if `use_integration_bias=false`)
    $\mathbf{b}_{mi}$
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)
  - `memory`: Initial memory vector (not present if `train_memory=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct JANETCell{TS <: StaticBool, TM <: StaticBool} <:
                 AbstractDoubleRecurrentCell{TS, TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_integration_bias
    init_weight
    init_recurrent_weight
    init_state
    init_memory
    use_bias <: StaticBool
    use_recurrent_bias <: StaticBool
    beta
    integration_mode
end

function JANETCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), use_recurrent_bias::BoolType=True(), use_integration_bias::BoolType=False(),
        train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_recurrent_bias=nothing, init_integration_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing, init_state=zeros32,
        init_memory=zeros32, beta::Number=1.0f0, integration_mode=AdditiveIntegration())
    init_weight isa NTuple{2} || (init_weight = ntuple(Returns(init_weight), 2))
    init_recurrent_weight isa NTuple{2} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 2))
    init_bias isa NTuple{2} || (init_bias = ntuple(Returns(init_bias), 2))
    init_recurrent_bias isa NTuple{2} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 2))
    init_integration_bias isa NTuple{2} ||
        (init_integration_bias = ntuple(Returns(init_integration_bias), 2))
    return JANETCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_recurrent_bias, init_integration_bias, init_weight, init_recurrent_weight, init_state,
        init_memory, static(use_bias), static(use_recurrent_bias), beta, integration_mode)
end

initialparameters(rng::AbstractRNG, janet::JANETCell) = multi_initialparameters(rng, janet)

function (janet::JANETCell)(
        (inp,
            (state, c_state))::Tuple{
            <:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}},
        ps, st::NamedTuple)
    matched_inp, matched_state, matched_cstate = match_eltype(
        janet, ps, st, inp, state, c_state)
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    bias_mi = safe_getproperty(ps, Val(:bias_mi))
    full_gs = recurrence_double_bias(janet.integration_mode, ps.weight_ih, ps.weight_hh,
        matched_inp, matched_state, bias_ih, bias_hh, bias_mi)
    gs = multigate(full_gs, Val(2))
    candidate_state = tanh_fast.(gs[2])
    ones_vec = one(eltype(candidate_state))
    new_cstate = @. sigmoid_fast(gs[1]) * c_state +
                    (ones_vec - sigmoid_fast(gs[1] - janet.beta)) *
                    candidate_state
    new_state = new_cstate
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, janet::JANETCell)
    print(io, "LSTMCell($(janet.in_dims) => $(janet.out_dims)")
    has_bias(janet) || print(io, ", use_bias=false")
    has_train_state(janet) && print(io, ", train_state=true")
    known(janet.train_memory) && print(io, ", train_memory=true")
    return print(io, ")")
end
