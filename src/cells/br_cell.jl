#https://doi.org/10.1371/journal.pone.0252676
@doc raw"""
    BRCell(in_dims => out_dims;
        use_bias=true, train_state=false, init_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)
    
[Bistable recurrent cell](https://doi.org/10.1371/journal.pone.0252676).

## Equations
```math
\begin{aligned}
    \mathbf{h}_t &= \mathbf{c}_t \circ \mathbf{h}_{t-1} + (1 - \mathbf{c}_t)
        \circ \tanh\left(\mathbf{U}_x \mathbf{x}_t + \mathbf{a}_t \circ
        \mathbf{h}_{t-1}\right), \\
    \mathbf{a}_t &= 1 + \tanh\left(\mathbf{U}_a \mathbf{x}_t +
        \mathbf{w}_a \circ \mathbf{h}_{t-1}\right), \\
    \mathbf{c}_t &= \sigma\left(\mathbf{U}_c \mathbf{x}_t + \mathbf{w}_c \circ
        \mathbf{h}_{t-1}\right).
\end{aligned}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension

## Keyword Arguments

  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.
    Default set to `false`.
  - `init_bias`: Initializer for bias. Must be a tuple containing 2 functions. If a single
    value is passed, it is copied into a 2 element tuple. If `nothing`, then we use
    uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(out_dims))`. Default set to `nothing`.
  - `init_weight`: Initializer for weight. Must be a tuple containing 2 functions. If a
    single value is passed, it is copied into a 2 element tuple. If `nothing`, then we use
    uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(out_dims))`. Default set to `nothing`.
  - `init_recurrent_weight`: Initializer for recurrent weight. Must be a tuple containing 2 functions. If a
    single value is passed, it is copied into a 2 element tuple. If `nothing`, then we use
    uniform distribution with bounds `-bound` and `bound` where
    `bound = inv(sqrt(out_dims))`. Default set to `nothing`.
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

  -  `weight_ih`: Concatenated Weights to map from input space
                 ``\{ U_x, U_a, U_c \}``.
  - `weight_hh`: Concatenated Weights to map from hidden space
                 ``\{ a, w_a, w_c \}``
  - `bias_ih`: Bias vector for the input-hidden connection (not present if `use_bias=false`)
  - `bias_hh`: Bias vector for the hidden-hidden connection (not present if `use_bias=false`)
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct BRCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
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

function BRCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), init_bias=nothing,
        init_recurrent_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32)
    init_weight isa NTuple{3} || (init_weight = ntuple(Returns(init_weight), 3))
    init_recurrent_weight isa NTuple{3} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 3))
    init_bias isa NTuple{3} || (init_bias = ntuple(Returns(init_bias), 3))
    return BRCell(static(train_state), in_dims, out_dims, init_bias, init_recurrent_bias,
        init_weight, init_recurrent_weight, init_state, static(use_bias))
end

function initialparameters(rng::AbstractRNG, br::BRCell)
    weight_ih = multi_inits(
        rng, br.init_weight, br.out_dims, (br.out_dims, br.in_dims))
    weight_hh = vec(multi_inits(rng, br.init_recurrent_weight, br.out_dims,
        (br.out_dims, 1)))
    ps = (; weight_ih, weight_hh)
    if has_bias(br)
        bias_ih = multi_bias(rng, br.init_bias, br.out_dims, br.out_dims)
        bias_hh = init_rnn_bias(rng, br.init_recurrent_bias, br.out_dims, br.out_dims)
        ps = merge(ps, (; bias_ih, bias_hh))
    end
    has_train_state(br) &&
        (ps = merge(ps, (hidden_state=br.init_state(rng, br.out_dims),)))
    return ps
end

initialstates(rng::AbstractRNG, ::BRCell) = (rng=Utils.sample_replicate(rng),)

function parameterlength(br::BRCell)
    return br.in_dims * br.out_dims * 3 + br.out_dims * 3 +
           br.out_dims * 3 * 2
end

function (br::BRCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state = match_eltype(br, ps, st, inp, state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #computation
    t_ones = eltype(bias_ih)(1.0)
    full_xs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    xs = multigate(full_xs, Val(3))
    ws = multigate(ps.weight_hh, Val(3))
    bhs = multigate(bias_hh, Val(3))
    modulation_gate = @. t_ones + tanh_fast(xs[1] + ws[1] * state + bhs[1])
    candidate_state = @. sigmoid_fast(xs[2] + ws[2] * state + bhs[2])
    new_state = @. candidate_state * state +
                   (t_ones - candidate_state) * tanh_fast(xs[3] + ws[3] * state + bhs[3])

    return (new_state, (new_state,)), st
end

function Base.show(io::IO, br::BRCell)
    print(io, "BRCell($(br.in_dims) => $(br.out_dims)")
    has_bias(br) || print(io, ", use_bias=false")
    has_train_state(br) && print(io, ", train_state=true")
    print(io, ")")
end
