#https://arxiv.org/pdf/1603.09420
@doc raw"""
    MGUCell(in_dims => out_dims;
        use_bias=true, train_state=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)
    
[Minimal gated unit](https://arxiv.org/pdf/1603.09420).

## Equations
```math
\begin{aligned}
    \mathbf{f}(t) &= \sigma\left( 
        \mathbf{W}_{ih}^{f} \mathbf{x}(t) + \mathbf{b}_{ih}^{f} + 
        \mathbf{W}_{hh}^{f} \mathbf{h}(t-1) + \mathbf{b}_{hh}^{f} \right), \\
    \tilde{\mathbf{h}}(t) &= \tanh\left( 
        \mathbf{W}_{ih}^{h} \mathbf{x}(t) + \mathbf{b}_{ih}^{h} + 
        \mathbf{W}_{hh}^{h} \left( \mathbf{f}(t) \circ \mathbf{h}(t-1) \right) +
        \mathbf{b}_{hh}^{h} \right), \\
    \mathbf{h}(t) &= \left(1 - \mathbf{f}(t)\right) \circ \mathbf{h}(t-1) +
        \mathbf{f}(t) \circ \tilde{\mathbf{h}}(t)
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
    $\{ \mathbf{b}_{ih}^{f}, \mathbf{b}_{ih}^{h} \}$.  
    Must be a tuple of 2 functions. If a single function is provided, it is
    expanded to 2 copies. If set to `nothing`, biases are initialized from a
    uniform distribution in `[-bound, bound]` where
    `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_recurrent_bias`: Initializer for hidden-to-hidden biases  
    $\{ \mathbf{b}_{hh}^{f}, \mathbf{b}_{hh}^{h} \}``.  
    Must be a tuple of 2 functions. If a single function is provided,
    it is expanded to 2 copies. If set to `nothing`, biases are initialized
    from a uniform distribution in `[-bound, bound]` where
    `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_weight`: Initializer for input-to-hidden weights  
    $\{ \mathbf{W}_{ih}^{f}, \mathbf{W}_{ih}^{h} \}$.  
    Must be a tuple of 2 functions. If a single function is provided, it is
    expanded to 2 copies. If set to `nothing`, weights are initialized from a
    uniform distribution in `[-bound, bound]` where
    `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_recurrent_weight`: Initializer for hidden-to-hidden weights  
    $\{ \mathbf{W}_{hh}^{f}, \mathbf{W}_{hh}^{h} \}$.  
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

  - `weight_ih`: Input-to-hidden weights  
    ``\{ \mathbf{W}_{ih}^{f}, \mathbf{W}_{ih}^{h} \}``  
    The functions from `init_weight` are applied in order:  
    the first initializes $\mathbf{W}_{ih}^{f}$, the second $\mathbf{W}_{ih}^{h}$.
  - `weight_hh`: Hidden-to-hidden weights  
    ``\{ \mathbf{W}_{hh}^{f}, \mathbf{W}_{hh}^{h} \}``  
    The functions from `init_recurrent_weight` are applied in order:  
    the first initializes $\mathbf{W}_{hh}^{f}$, the second $\mathbf{W}_{hh}^{h}$.
  - `bias_ih`: Input-to-hidden biases (if `use_bias=true`)  
    ``\{ \mathbf{b}_{ih}^{f}, \mathbf{b}_{ih}^{h} \}``  
    The functions from `init_bias` are applied in order:  
    the first initializes $\mathbf{b}_{ih}^{f}$, the second $\mathbf{b}_{ih}^{h}$.
  - `bias_hh`: Hidden-to-hidden biases (if `use_bias=true`)  
    ``\{ \mathbf{b}_{hh}^{f}, \mathbf{b}_{hh}^{h} \}``  
    The functions from `init_recurrent_bias` are applied in order:  
    the first initializes $\mathbf{b}_{hh}^{f}$, the second $\mathbf{b}_{hh}^{h}$.
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct MGUCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
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

function MGUCell(
        (in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType}, activation=tanh_fast;
        use_bias::BoolType=True(), train_state::BoolType=False(), init_bias=nothing,
        init_recurrent_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)
    init_weight isa NTuple{2} || (init_weight = ntuple(Returns(init_weight), 2))
    init_recurrent_weight isa NTuple{2} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 2))
    init_bias isa NTuple{2} || (init_bias = ntuple(Returns(init_bias), 2))
    init_recurrent_bias isa NTuple{2} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 2))
    return MGUCell(
        static(train_state), in_dims, out_dims, activation, init_bias, init_recurrent_bias,
        init_weight, init_recurrent_weight, init_state, static(use_bias))
end

initialparameters(rng::AbstractRNG, mgu::MGUCell) = multi_initialparameters(rng, mgu)

function parameterlength(mgu::MGUCell)
    return mgu.in_dims * mgu.out_dims * 2 + mgu.out_dims * mgu.out_dims * 2 +
           mgu.out_dims * 4
end

function (mgu::MGUCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(mgu, ps, st, inp, state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #computation
    t_ones = eltype(bias_ih)(1.0f0)
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    gxs = multigate(full_gxs, Val(2))
    whs = multigate(ps.weight_hh, Val(2))
    bhs = multigate(bias_hh, Val(2))

    forget_gate = sigmoid_fast.(gxs[1] .+ whs[1] * matched_state .+ bhs[1])
    candidate_state = tanh_fast.(gxs[2] .+ whs[2] * (forget_gate .* matched_state) .+
                                 bhs[2])
    new_state = @. forget_gate * matched_state + (t_ones - forget_gate) * candidate_state
    return (new_state, (new_state,)), st
end

function Base.show(io::IO, mgu::MGUCell)
    print(io, "MGUCell($(mgu.in_dims) => $(mgu.out_dims)")
    has_bias(mgu) || print(io, ", use_bias=false")
    has_train_state(mgu) && print(io, ", train_state=true")
    print(io, ")")
end
