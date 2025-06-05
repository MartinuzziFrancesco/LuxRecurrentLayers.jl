#https://arxiv.org/abs/1810.12546
@doc raw"""
    ATRCell(in_dims => out_dims;
        use_bias=true, train_state=false, init_bias=nothing,
        init_recurrent_bias=nothing, init_weight=nothing,
        init_recurrent_weight=nothing, init_state=zeros32)

[Addition-subtraction twin-gated recurrent cell](https://arxiv.org/abs/1810.12546).

## Equations

```math
\begin{aligned}
    \mathbf{p}(t) &= \mathbf{W}_{ih} \mathbf{x}(t) + \mathbf{b}_{ih}, \\
    \mathbf{q}(t) &= \mathbf{W}_{ih} \mathbf{h}(t-1) + \mathbf{b}_{hh}, \\
    \mathbf{i}(t) &= \sigma(\mathbf{p}(t) + \mathbf{q}(t)), \\
    \mathbf{f}(t) &= \sigma(\mathbf{p}(t) - \mathbf{q}(t)), \\
    \mathbf{h}(t) &= \mathbf{i}(t) \circ \mathbf{p}(t) + \mathbf{f}(t)
        \circ \mathbf{h}(t-1).
\end{aligned}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension

# Keyword arguments


  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.
    Default set to `false`.
  - `train_memory`: Flag to set the initial memory state as trainable.
    Default set to `false`.
  - `init_bias`: Initializer for input to hidden bias $\mathbf{b}_{ih}$. If set to
    `nothing`, weights are initialized from a uniform distribution within
    `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
    Default is `nothing`.
  - `init_recurrent_bias`: Initializer for hidden to hidden bias $\mathbf{b}_{hh}$. If set
    to `nothing`, weights are initialized from a uniform distribution within
    `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
    Default is `nothing`.
  - `init_weight`: Initializer for input to hidden weights $\mathbf{W}_{ih}$. If set to
    `nothing`, weights are initialized from a uniform distribution within
    `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
    Default is `nothing`.
  - `init_recurrent_weight`: Initializer for recurrent weight $\mathbf{W}_{hh}$. If set to
    `nothing`, weights are initialized from a uniform distribution within
    `[-bound, bound]` where `bound = inv(sqrt(out_dims))`.
    Default is `nothing`.
  - `init_state`: Initializer for hidden state. Default set to `zeros32`.
  - `init_memory`: Initializer for memory. Default set to `zeros32`.

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

  - `weight_ih`: Concatenated weights to map from input to the hidden state $\mathbf{W}_{ih}$.
  - `weight_hh`: Concatenated weights to map from hidden to the hidden state $\mathbf{W}_{hh}$.
  - `bias_ih`: Bias vector for the input-hidden connection (not present if
      `use_bias=false`) $\mathbf{b}_{ih}$.
  - `bias_hh`: Bias vector for the hidden-hidden connection (not present if
      `use_bias=false`) $\mathbf{b}_{hh}$.
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct ATRCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
    train_state::TS
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_weight
    init_recurrent_weight
    init_state
    use_bias <: StaticBool
end

function ATRCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(),
        init_bias=nothing, init_recurrent_bias=nothing, init_weight=nothing,
        init_recurrent_weight=nothing, init_state=zeros32)
    return ATRCell(static(train_state), in_dims, out_dims,
        init_bias, init_recurrent_bias, init_weight, init_recurrent_weight, init_state,
        static(use_bias))
end

function initialparameters(rng::AbstractRNG, atr::ATRCell)
    return single_initialparameters(rng, atr)
end

initialstates(rng::AbstractRNG, ::ATRCell) = (rng=Utils.sample_replicate(rng),)

function parameterlength(atr::ATRCell)
    return atr.in_dims * atr.out_dims + atr.out_dims * atr.out_dims +
           atr.out_dims * 2
end

statelength(::ATRCell) = 1

function (atr::ATRCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(atr, ps, st, inp, state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #computation
    pt = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    qt = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)
    it = @. sigmoid_fast(pt + qt)
    ft = @. sigmoid_fast(pt + qt)
    new_state = @. it * pt + ft * state
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, r::ATRCell)
    print(io, "ATRCell($(r.in_dims) => $(r.out_dims)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end
