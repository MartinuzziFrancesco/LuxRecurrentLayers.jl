#https://proceedings.mlr.press/v37/jozefowicz15.pdf
@doc raw"""
    MUT1Cell(in_dims => out_dims;
        use_bias=true, train_state=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)

[Mutated unit 1 cell](https://proceedings.mlr.press/v37/jozefowicz15.pdf).

## Equations
```math
\begin{aligned}
    \mathbf{z}(t) &= \sigma\left(\mathbf{W}_{ih}^{z} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{z}\right), \\
    \mathbf{r}(t) &= \sigma\left(\mathbf{W}_{ih}^{r} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{r} + \mathbf{W}_{hh}^{r} \mathbf{h}(t) +
        \mathbf{b}_{hh}^{r}\right), \\
    \mathbf{h}(t+1) &= \left[ \tanh\left( \mathbf{W}_{hh}^{h} \left(
        \mathbf{r}(t) \circ \mathbf{h}(t) + \mathbf{b}_{hh}^{h} \right) + \tanh(\mathbf{W}_{ih}^{h}
        \mathbf{x}(t)) + \mathbf{b}_{ih}^{h} \right) \right] \circ \mathbf{z}(t)
        + \mathbf{h}(t) \circ (1 - \mathbf{z}(t)).
\end{aligned}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension

## Keyword Arguments

  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.
    Default set to `false`.
  - `init_bias`: Initializer for input-to-hidden biases
    $\mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{r}, \mathbf{b}_{ih}^{h}$.
    Must be a tuple containing 3 functions. If a single value is passed, it is copied into a 3-element tuple.
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`.
    The functions are applied in order:
    the first initializes $\mathbf{b}_{ih}^{z}$, the second $\mathbf{b}_{ih}^{r}$,
    and the third $\mathbf{b}_{ih}^{h}$.
  - `init_recurrent_bias`: Initializer for hidden-to-hidden biases
    $\mathbf{b}_{hh}^{r}, \mathbf{b}_{hh}^{h}$.
    Must be a tuple containing 2 functions. If a single value is passed, it is copied into a 2-element tuple.
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`.
    The functions are applied in order:
    the first initializes $\mathbf{b}_{hh}^{r}$, and the second $\mathbf{b}_{hh}^{h}$.
  - `init_weight`: Initializer for input-to-hidden weights
    $\mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{r}, \mathbf{W}_{ih}^{h}$.
    Must be a tuple containing 3 functions. If a single value is passed, it is copied into a 3-element tuple.
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`.
    The functions are applied in order:
    the first initializes $\mathbf{W}_{ih}^{z}$, the second $\mathbf{W}_{ih}^{r}$,
    and the third $\mathbf{W}_{ih}^{h}$.
  - `init_recurrent_weight`: Initializer for hidden-to-hidden weights
    $\mathbf{W}_{hh}^{r}, \mathbf{W}_{hh}^{h}$.
    Must be a tuple containing 2 functions. If a single value is passed, it is copied into a 2-element tuple.
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`.
    The functions are applied in order:
    the first initializes $\mathbf{W}_{hh}^{r}$, and the second $\mathbf{W}_{hh}^{h}$.
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

  - `weight_ih`: Input-to-hidden weights
    ``\{ \mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{r}, \mathbf{W}_{ih}^{h} \}``
  - `weight_hh`: Hidden-to-hidden weights
    ``\{ \mathbf{W}_{hh}^{r}, \mathbf{W}_{hh}^{h} \}``
  - `bias_ih`: Input-to-hidden biases (if `use_bias=true`)
    ``\{ \mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{r}, \mathbf{b}_{ih}^{h} \}``
  - `bias_hh`: Hidden-to-hidden biases (if `use_bias=true`)
    ``\{ \mathbf{b}_{hh}^{r}, \mathbf{b}_{hh}^{h} \}``
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct MUT1Cell{TS<:StaticBool} <: AbstractSingleRecurrentCell{TS}
    train_state::TS
    in_dims <: IntegerType
    out_dims <: IntegerType
    activation
    init_bias
    init_recurrent_bias
    init_weight
    init_recurrent_weight
    init_state
    use_bias <: StaticBool
end

function MUT1Cell(
    (in_dims, out_dims)::Pair{<:IntegerType,<:IntegerType}, activation=tanh_fast;
    use_bias::BoolType=True(), train_state::BoolType=False(), init_bias=nothing,
    init_recurrent_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
    init_state=zeros32)
    init_weight isa NTuple{3} || (init_weight = ntuple(Returns(init_weight), 3))
    init_recurrent_weight isa NTuple{2} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 2))
    init_bias isa NTuple{3} || (init_bias = ntuple(Returns(init_bias), 3))
    init_recurrent_bias isa NTuple{2} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 2))
    return MUT1Cell(
        static(train_state), in_dims, out_dims, activation, init_bias, init_recurrent_bias,
        init_weight, init_recurrent_weight, init_state, static(use_bias))
end

initialparameters(rng::AbstractRNG, mut::MUT1Cell) = multi_initialparameters(rng, mut)

initialstates(rng::AbstractRNG, ::MUT1Cell) = (rng=Utils.sample_replicate(rng),)

function parameterlength(mut::MUT1Cell)
    return mut.in_dims * mut.out_dims * 3 + mut.out_dims * mut.out_dims * 2 +
           mut.out_dims * 5
end

function (mut::MUT1Cell)(
    (inp, (state,))::Tuple{<:AbstractMatrix,Tuple{<:AbstractMatrix}},
    ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(mut, ps, st, inp, state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #computation
    t_ones = one(eltype(matched_inp))
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    gxs = multigate(full_gxs, Val(3))
    whs = multigate(ps.weight_hh, Val(2))
    bhs = multigate(bias_hh, Val(2))

    forget_gate = sigmoid_fast.(gxs[1])
    reset_gate = sigmoid_fast.(gxs[2] .+ whs[1] * matched_state .+ bhs[1])
    candidate_state = tanh_fast.(
        whs[2] * (reset_gate .* matched_state) .+ bhs[2] .+ tanh_fast(gxs[3])
    ) #in the paper is tanh(x_t) but dimensionally it cannot work
    new_state = candidate_state .* forget_gate .+ matched_state .* (t_ones .- forget_gate)
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, mut::MUT1Cell)
    print(io, "MUT1Cell($(mut.in_dims) => $(mut.out_dims)")
    has_bias(mut) || print(io, ", use_bias=false")
    has_train_state(mut) && print(io, ", train_state=true")
    print(io, ")")
end

@doc raw"""
    MUT2Cell(in_dims => out_dims;
        use_bias=true, train_state=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)

[Mutated unit 2 cell](https://proceedings.mlr.press/v37/jozefowicz15.pdf).

## Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \sigma\left( \mathbf{W}_{ih}^{z} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{z} + \mathbf{W}_{hh}^{z} \mathbf{h}(t) +
        \mathbf{b}_{hh}^{z} \right), \\
    \mathbf{r}(t) &= \sigma\left( \mathbf{W}_{ih}^{r} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{r} + \mathbf{W}_{hh}^{r} \mathbf{h}(t) +
        \mathbf{b}_{hh}^{r} \right), \\
    \mathbf{h}(t+1) &= \tanh\left( \mathbf{W}_{hh}^{h} \left( \mathbf{r}(t)
        \circ \mathbf{h}(t) + \mathbf{b}_{hh}^{h} \right) +
        \mathbf{W}_{ih}^{h} \mathbf{x}(t) + \mathbf{b}_{ih}^{h} \right) \circ
        \mathbf{z}(t) \\
    &\quad + \mathbf{h}(t) \circ \left( 1 - \mathbf{z}(t) \right)
\end{aligned}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension

## Keyword Arguments

  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.
    Default set to `false`.
  - `init_bias`: Initializer for input-to-hidden biases
    $\mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{r}, \mathbf{b}_{ih}^{h}$.
    Must be a tuple containing 3 functions. If a single value is passed, it is copied into a 3-element tuple.
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`.
    The functions are applied in order:
    the first initializes $\mathbf{b}_{ih}^{z}$, the second $\mathbf{b}_{ih}^{r}$,
    and the third $\mathbf{b}_{ih}^{h}$.
  - `init_recurrent_bias`: Initializer for hidden-to-hidden biases
    $\mathbf{b}_{hh}^{z}, \mathbf{b}_{hh}^{r}, \mathbf{b}_{hh}^{h}$.
    Must be a tuple containing 3 functions. If a single value is passed, it is copied into a 3-element tuple.
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`.
    The functions are applied in order:
    the first initializes $\mathbf{b}_{hh}^{z}$, the second $\mathbf{b}_{hh}^{r}$,
    and the third $\mathbf{b}_{hh}^{h}$.
  - `init_weight`: Initializer for input-to-hidden weights
    $\mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{r}, \mathbf{W}_{ih}^{h}$.
    Must be a tuple containing 3 functions. If a single value is passed, it is copied into a 3-element tuple.
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`.
    The functions are applied in order:
    the first initializes $\mathbf{W}_{ih}^{z}$, the second $\mathbf{W}_{ih}^{r}$,
    and the third $\mathbf{W}_{ih}^{h}$.
  - `init_recurrent_weight`: Initializer for hidden-to-hidden weights
    $\mathbf{W}_{hh}^{z}, \mathbf{W}_{hh}^{r}, \mathbf{W}_{hh}^{h}$.
    Must be a tuple containing 3 functions. If a single value is passed, it is copied into a 3-element tuple.
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`.
    The functions are applied in order:
    the first initializes $\mathbf{W}_{hh}^{z}$, the second $\mathbf{W}_{hh}^{r}$,
    and the third $\mathbf{W}_{hh}^{h}$.
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

  - `weight_ih`: Input-to-hidden weights
    ``\{ \mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{r}, \mathbf{W}_{ih}^{h} \}``
  - `weight_hh`: Hidden-to-hidden weights
    ``\{ \mathbf{W}_{hh}^{z}, \mathbf{W}_{hh}^{r}, \mathbf{W}_{hh}^{h} \}``
  - `bias_ih`: Input-to-hidden biases (if `use_bias=true`)
    ``\{ \mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{r}, \mathbf{b}_{ih}^{h} \}``
  - `bias_hh`: Hidden-to-hidden biases (if `use_bias=true`)
    ``\{ \mathbf{b}_{hh}^{z}, \mathbf{b}_{hh}^{r}, \mathbf{b}_{hh}^{h} \}``
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct MUT2Cell{TS<:StaticBool} <: AbstractSingleRecurrentCell{TS}
    train_state::TS
    in_dims <: IntegerType
    out_dims <: IntegerType
    activation
    init_bias
    init_recurrent_bias
    init_weight
    init_recurrent_weight
    init_state
    use_bias <: StaticBool
end

function MUT2Cell(
    (in_dims, out_dims)::Pair{<:IntegerType,<:IntegerType}, activation=tanh_fast;
    use_bias::BoolType=True(), train_state::BoolType=False(), init_bias=nothing,
    init_recurrent_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
    init_state=zeros32)
    init_weight isa NTuple{3} || (init_weight = ntuple(Returns(init_weight), 3))
    init_recurrent_weight isa NTuple{3} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 3))
    init_bias isa NTuple{3} || (init_bias = ntuple(Returns(init_bias), 3))
    init_recurrent_bias isa NTuple{3} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 3))
    return MUT2Cell(
        static(train_state), in_dims, out_dims, activation, init_bias, init_recurrent_bias,
        init_weight, init_recurrent_weight, init_state, static(use_bias))
end

initialparameters(rng::AbstractRNG, mut::MUT2Cell) = multi_initialparameters(rng, mut)

initialstates(rng::AbstractRNG, ::MUT2Cell) = (rng=Utils.sample_replicate(rng),)

function parameterlength(mut::MUT2Cell)
    return mut.in_dims * mut.out_dims * 3 + mut.out_dims * mut.out_dims * 3 +
           mut.out_dims * 6
end

function (mut::MUT2Cell)(
    (inp, (state,))::Tuple{<:AbstractMatrix,Tuple{<:AbstractMatrix}},
    ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(mut, ps, st, inp, state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #computation
    t_ones = one(eltype(matched_inp))
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    gxs = multigate(full_gxs, Val(3))
    whs = multigate(ps.weight_hh, Val(3))
    bhs = multigate(bias_hh, Val(3))

    forget_gate = sigmoid_fast.(gxs[1] .+ whs[1] * matched_state .+ bhs[1])
    # the dimensionlity alos does not work here like the paper describes it
    reset_gate = sigmoid_fast.(gxs[2] .+ whs[2] * matched_state .+ bhs[2])
    candidate_state = tanh_fast.(whs[3] * (reset_gate .* matched_state) .+ bhs[3] .+ gxs[3])
    new_state = candidate_state .* forget_gate .+ matched_state .* (t_ones .- forget_gate)
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, mut::MUT2Cell)
    print(io, "MUT2Cell($(mut.in_dims) => $(mut.out_dims)")
    has_bias(mut) || print(io, ", use_bias=false")
    has_train_state(mut) && print(io, ", train_state=true")
    print(io, ")")
end

@doc raw"""
    MUT3Cell(in_dims => out_dims;
        use_bias=true, train_state=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)

[Mutated unit 3 cell](https://proceedings.mlr.press/v37/jozefowicz15.pdf).

## Equations
```math
\begin{aligned}
    \mathbf{z}(t) &= \sigma\left( \mathbf{W}_{ih}^{z} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{z} + \mathbf{W}_{hh}^{z} \mathbf{h}(t) +
        \mathbf{b}_{hh}^{z} \right), \\
    \mathbf{r}(t) &= \sigma\left( \mathbf{x}(t) + \mathbf{W}_{hh}^{r}
        \mathbf{h}(t) + \mathbf{b}_{hh}^{r} \right), \\
    \mathbf{h}(t+1) &= \left[ \tanh\left( \mathbf{W}_{hh}^{h} \left(
        \mathbf{r}(t) \circ \mathbf{h}(t) + \mathbf{b}_{hh}^{h} \right) +
        \mathbf{W}_{ih}^{h} \mathbf{x}(t) + \mathbf{b}_{ih}^{h} \right) \right]
        \circ \mathbf{z}(t) \\
    &\quad + \mathbf{h}(t) \circ \left( 1 - \mathbf{z}(t) \right)
\end{aligned}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension

## Keyword Arguments

  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.
    Default set to `false`.
  - `init_bias`: Initializer for input-to-hidden biases
    $\mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{h}$.
    Must be a tuple containing 2 functions. If a single value is passed, it is copied into a 2-element tuple.
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`.
    The functions are applied in order:
    the first initializes $\mathbf{b}_{ih}^{z}$, and the second $\mathbf{b}_{ih}^{h}$.
  - `init_recurrent_bias`: Initializer for hidden-to-hidden biases
    $\mathbf{b}_{hh}^{z}, \mathbf{b}_{hh}^{r}, \mathbf{b}_{hh}^{h}$.
    Must be a tuple containing 3 functions. If a single value is passed, it is copied into a 3-element tuple.
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`.
    The functions are applied in order:
    the first initializes $\mathbf{b}_{hh}^{z}$, the second $\mathbf{b}_{hh}^{r}$,
    and the third $\mathbf{b}_{hh}^{h}$.
  - `init_weight`: Initializer for input-to-hidden weights
    $\mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{h}$.
    Must be a tuple containing 2 functions. If a single value is passed, it is copied into a 2-element tuple.
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`.
    The functions are applied in order:
    the first initializes $\mathbf{W}_{ih}^{z}$, and the second $\mathbf{W}_{ih}^{h}$.
  - `init_recurrent_weight`: Initializer for hidden-to-hidden weights
    $\mathbf{W}_{hh}^{z}, \mathbf{W}_{hh}^{r}, \mathbf{W}_{hh}^{h}$.
    Must be a tuple containing 3 functions. If a single value is passed, it is copied into a 3-element tuple.
    If set to `nothing`, weights are initialized from a uniform distribution within `[-bound, bound]`
    where `bound = inv(sqrt(out_dims))`.
    The functions are applied in order:
    the first initializes $\mathbf{W}_{hh}^{z}$, the second $\mathbf{W}_{hh}^{r}$,
    and the third $\mathbf{W}_{hh}^{h}$.
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

  - `weight_ih`: Input-to-hidden weights
    ``\{ \mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{h} \}``
  - `weight_hh`: Hidden-to-hidden weights
    ``\{ \mathbf{W}_{hh}^{z}, \mathbf{W}_{hh}^{r}, \mathbf{W}_{hh}^{h} \}``
  - `bias_ih`: Input-to-hidden biases (if `use_bias=true`)
    ``\{ \mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{h} \}``
  - `bias_hh`: Hidden-to-hidden biases (if `use_bias=true`)
    ``\{ \mathbf{b}_{hh}^{z}, \mathbf{b}_{hh}^{r}, \mathbf{b}_{hh}^{h} \}``
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct MUT3Cell{TS<:StaticBool} <: AbstractSingleRecurrentCell{TS}
    train_state::TS
    in_dims <: IntegerType
    out_dims <: IntegerType
    activation
    init_bias
    init_recurrent_bias
    init_weight
    init_recurrent_weight
    init_state
    use_bias <: StaticBool
end

function MUT3Cell(
    (in_dims, out_dims)::Pair{<:IntegerType,<:IntegerType}, activation=tanh_fast;
    use_bias::BoolType=True(), train_state::BoolType=False(), init_bias=nothing,
    init_recurrent_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
    init_state=zeros32)
    init_weight isa NTuple{3} || (init_weight = ntuple(Returns(init_weight), 3))
    init_recurrent_weight isa NTuple{3} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 3))
    init_bias isa NTuple{3} || (init_bias = ntuple(Returns(init_bias), 3))
    init_recurrent_bias isa NTuple{3} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 3))
    return MUT3Cell(
        static(train_state), in_dims, out_dims, activation, init_bias, init_recurrent_bias,
        init_weight, init_recurrent_weight, init_state, static(use_bias))
end

initialparameters(rng::AbstractRNG, mut::MUT3Cell) = multi_initialparameters(rng, mut)

initialstates(rng::AbstractRNG, ::MUT3Cell) = (rng=Utils.sample_replicate(rng),)

function parameterlength(mut::MUT3Cell)
    return mut.in_dims * mut.out_dims * 3 + mut.out_dims * mut.out_dims * 2 +
           mut.out_dims * 5
end

function (mut::MUT3Cell)(
    (inp, (state,))::Tuple{<:AbstractMatrix,Tuple{<:AbstractMatrix}},
    ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(mut, ps, st, inp, state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #computation
    t_ones = eltype(bias_ih)(1.0f0)
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    gxs = multigate(full_gxs, Val(3))
    whs = multigate(ps.weight_hh, Val(3))
    bhs = multigate(bias_hh, Val(3))

    forget_gate = sigmoid_fast.(gxs[1] .+ whs[1] * tanh_fast(matched_state) .+ bhs[1])
    reset_gate = sigmoid_fast.(gxs[2] .+ whs[2] * matched_state .+ bhs[2])
    candidate_state = tanh_fast.(whs[3] * (reset_gate .* matched_state) .+ gxs[3] .+ bhs[3])
    new_state = candidate_state .* forget_gate .+ matched_state .* (t_ones .- forget_gate)
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, mut::MUT3Cell)
    print(io, "MUT3Cell($(mut.in_dims) => $(mut.out_dims)")
    has_bias(mut) || print(io, ", use_bias=false")
    has_train_state(mut) && print(io, ", train_state=true")
    print(io, ")")
end
