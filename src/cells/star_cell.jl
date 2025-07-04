#https://arxiv.org/abs/1911.11033
@doc raw"""
    STARCell(in_dims => out_dims;
        use_bias=true, train_state=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)

[Stackable recurrent cell](https://arxiv.org/abs/1911.11033).

## Equations

```math
\begin{aligned}
    \mathbf{z}(t) &= \tanh\left(\mathbf{W}_{ih}^{z} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{z}\right), \\
    \mathbf{k}(t) &= \sigma\left(\mathbf{W}_{ih}^{k} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{k} + \mathbf{W}_{hh}^{k} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{k}\right), \\
    \mathbf{h}(t) &= \tanh\left((1 - \mathbf{k}(t)) \circ \mathbf{h}(t-1) +
        \mathbf{k}(t) \circ \mathbf{z}(t)\right).
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
  - `init_bias`: Initializer for input-to-hidden biases  
    $\mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{k}$.  
    Must be a tuple containing 2 functions. If a single value is passed, it is
    copied into a 2-element tuple. If set to `nothing`, biases are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = inv(sqrt(out_dims))`.  
    The functions are applied in order: the first initializes  
    $\mathbf{b}_{ih}^{z}$, the second $\mathbf{b}_{ih}^{k}$.  
    Default set to `nothing`.
  - `init_recurrent_bias`: Initializer for hidden-to-hidden bias  
    $\mathbf{b}_{hh}^{k}$.  
    Must be a single function. If set to `nothing`, bias is initialized  
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = inv(sqrt(out_dims))`.  
    Default set to `nothing`.
  - `init_weight`: Initializer for input-to-hidden weights  
    $\mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{k}$.  
    Must be a tuple containing 2 functions. If a single value is passed, it is
    copied into a 2-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = inv(sqrt(out_dims))`.  
    The functions are applied in order: the first initializes  
    $\mathbf{W}_{ih}^{z}$, the second $\mathbf{W}_{ih}^{k}$.  
    Default set to `nothing`.
  - `init_recurrent_weight`: Initializer for hidden-to-hidden weight  
    $\mathbf{W}_{hh}^{k}$.  
    Must be a single function. If set to `nothing`, weight is initialized  
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = inv(sqrt(out_dims))`.  
    Default set to `nothing`.
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

  - `weight_ih`: Input-to-hidden weights  
    ``\{ \mathbf{W}_{ih}^{z}, \mathbf{W}_{ih}^{k} \}``
  - `weight_hh`: Hidden-to-hidden weights  
    ``\{ \mathbf{W}_{hh}^{k} \}``
  - `bias_ih`: Input-to-hidden biases (not present if `use_bias=false`)  
    ``\{ \mathbf{b}_{ih}^{z}, \mathbf{b}_{ih}^{k} \}``
  - `bias_hh`: Hidden-to-hidden bias (not present if `use_bias=false`)  
    ``\{ \mathbf{b}_{hh}^{k} \}``
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct STARCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
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

function STARCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(),
        init_bias=nothing, init_recurrent_bias=nothing, init_weight=nothing,
        init_recurrent_weight=nothing, init_state=zeros32)
    init_weight isa NTuple{2} || (init_weight = ntuple(Returns(init_weight), 2))
    init_bias isa NTuple{2} || (init_bias = ntuple(Returns(init_bias), 2))
    return STARCell(static(train_state), in_dims, out_dims,
        init_bias, init_recurrent_bias, init_weight, init_recurrent_weight, init_state,
        static(use_bias))
end

function initialparameters(rng::AbstractRNG, star::STARCell)
    weight_ih = multi_inits(
        rng, star.init_weight, star.out_dims, (star.out_dims, star.in_dims))
    weight_hh = init_rnn_weight(rng, star.init_recurrent_weight, star.out_dims,
        (star.out_dims, star.out_dims))
    ps = (; weight_ih, weight_hh)
    if has_bias(star)
        bias_ih = multi_bias(rng, star.init_bias, star.out_dims, star.out_dims)
        bias_hh = init_rnn_bias(rng, star.init_recurrent_bias, star.out_dims, star.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    has_train_state(star) &&
        (ps = merge(ps, (hidden_state=star.init_state(rng, star.out_dims),)))
    return ps
end

initialstates(rng::AbstractRNG, ::STARCell) = (rng=Utils.sample_replicate(rng),)

function parameterlength(star::STARCell)
    return star.in_dims * star.out_dims * 2 + star.out_dims * star.out_dims +
           star.out_dims * 2
end

statelength(::STARCell) = 1

function (star::STARCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(star, ps, st, inp, state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #computation
    full_xs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    hs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)
    xs = multigate(full_xs, Val(2))

    input_gate = tanh_fast.(xs[1])
    forget_gate = @. sigmoid_fast(xs[2] + hs)
    new_state = @. tanh_fast((1 - forget_gate) * state + forget_gate * input_gate)
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, r::STARCell)
    print(io, "STARCell($(r.in_dims) => $(r.out_dims)")
    has_bias(r) || print(io, ", use_bias=false")
    has_train_state(r) && print(io, ", train_state=true")
    print(io, ")")
end
