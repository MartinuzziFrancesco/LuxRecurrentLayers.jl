#https://doi.org/10.1049/gtd2.12056
@doc raw"""
    SGRNCell(in_dims => out_dims;
        use_bias=true, train_state=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)
    
[Simple gated recurrent network](https://doi.org/10.1049/gtd2.12056).

## Equations
```math
\begin{aligned}
    \mathbf{f}(t) &= \sigma\left(
        \mathbf{W}_{ih} \mathbf{x}(t) + \mathbf{b}_{ih} +
        \mathbf{W}_{hh} \mathbf{h}(t-1) + \mathbf{b}_{hh} \right), \\
    \mathbf{i}(t) &= 1 - \mathbf{f}(t), \\
    \mathbf{h}(t) &= \tanh\left(
        \mathbf{i}(t) \circ \left( \mathbf{W}_{ih} \mathbf{x}(t) + \mathbf{b}_{ih} \right) +
        \mathbf{f}(t) \circ \mathbf{h}(t-1) \right)
\end{aligned}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension

## Keyword Arguments

  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.  
    Default set to `false`.
  - `init_bias`: Initializer for input-to-hidden bias  
    $\mathbf{b}_{ih}$.  
    Must be a single function. If set to `nothing`, bias is initialized  
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = inv(sqrt(out_dims))`.  
    Default set to `nothing`.
  - `init_recurrent_bias`: Initializer for hidden-to-hidden bias  
    $\mathbf{b}_{hh}$.  
    Must be a single function. If set to `nothing`, bias is initialized  
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = inv(sqrt(out_dims))`.  
    Default set to `nothing`.
  - `init_weight`: Initializer for input-to-hidden weight  
    $\mathbf{W}_{ih}$.  
    Must be a single function. If set to `nothing`, weight is initialized  
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = inv(sqrt(out_dims))`.  
    Default set to `nothing`.
  - `init_recurrent_weight`: Initializer for hidden-to-hidden weight  
    $\mathbf{W}_{hh}$.  
    Must be a single function. If set to `nothing`, weight is initialized  
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = inv(sqrt(out_dims))`.  
    Default set to `nothing`.
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

  - `weight_ih`: Input-to-hidden weight  
    ``\{ \mathbf{W} \}``
  - `weight_hh`: Hidden-to-hidden weight  
    ``\{ \mathbf{U} \}``
  - `bias_ih`: Input-to-hidden bias (not present if `use_bias=false`)  
    ``\{ \mathbf{b} \}``
  - `bias_hh`: Hidden-to-hidden bias (not present if `use_bias=false`)  
    ``\{ \mathbf{b}_{hh} \}``
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct SGRNCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
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

function SGRNCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), init_bias=nothing,
        init_recurrent_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)
    return SGRNCell(static(train_state), in_dims, out_dims, init_bias, init_recurrent_bias,
        init_weight, init_recurrent_weight, init_state, static(use_bias))
end

initialparameters(rng::AbstractRNG, sgrn::SGRNCell) = single_initialparameters(rng, sgrn)

function parameterlength(sgrn::SGRNCell)
    return sgrn.in_dims * sgrn.out_dims + sgrn.out_dims * sgrn.out_dims +
           sgrn.out_dims * 2
end

function (sgrn::SGRNCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(sgrn, ps, st, inp, state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #computation
    t_ones = eltype(bias_ih)(1.0f0)
    xs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    hs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)

    forget_gate = @. sigmoid_fast(xs + hs)
    input_gate = @. t_ones - forget_gate
    new_state = @. tanh_fast(input_gate * xs + forget_gate * matched_state)
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, sgrn::SGRNCell)
    print(io, "SGRNCell($(sgrn.in_dims) => $(sgrn.out_dims)")
    has_bias(sgrn) || print(io, ", use_bias=false")
    has_train_state(sgrn) && print(io, ", train_state=true")
    print(io, ")")
end
