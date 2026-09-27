#https://arxiv.org/pdf/1607.03474
#https://github.com/jzilly/RecurrentHighwayNetworks/blob/master/rhn.py#L138C1-L180C60
@doc raw"""
    RHNCell(in_dims => out_dims, [depth];
        couple_carry=true, use_bias=true, train_state=false,
        init_bias=nothing, init_weight=nothing, init_state=zeros32)

[Recurrent highway network cell](https://arxiv.org/pdf/1607.03474).

## Equations

At each time step the running state $\mathbf{s}_0(t) = \mathbf{s}(t-1)$ is
passed through `depth` highway micro-layers, and $\mathbf{s}(t) =
\mathbf{s}_{depth}(t)$ becomes the new state:

```math
\begin{aligned}
    \mathbf{h}_{\ell}(t) &= \tanh\left( \mathbf{W}^{h}_{\ell}
        \big[\mathbf{x}(t)\, \mathbb{I}_{\ell = 1}; \mathbf{s}_{\ell-1}(t)\big] +
        \mathbf{b}^{h}_{\ell} \right), \\
    \mathbf{t}_{\ell}(t) &= \sigma\left( \mathbf{W}^{t}_{\ell}
        \big[\mathbf{x}(t)\, \mathbb{I}_{\ell = 1}; \mathbf{s}_{\ell-1}(t)\big] +
        \mathbf{b}^{t}_{\ell} \right), \\
    \mathbf{c}_{\ell}(t) &= \sigma\left( \mathbf{W}^{c}_{\ell}
        \big[\mathbf{x}(t)\, \mathbb{I}_{\ell = 1}; \mathbf{s}_{\ell-1}(t)\big] +
        \mathbf{b}^{c}_{\ell} \right), \\
    \mathbf{s}_{\ell}(t) &= \mathbf{h}_{\ell}(t) \circ \mathbf{t}_{\ell}(t) +
        \mathbf{s}_{\ell-1}(t) \circ \mathbf{c}_{\ell}(t)
\end{aligned}
```

where $[\mathbf{a}; \mathbf{b}]$ denotes concatenation, and $\mathbb{I}_{\ell = 1}$
is only `1` for the first micro-layer, which is the only one to see the input
$\mathbf{x}(t)$. When `couple_carry=true` (the default) the carry gate is tied
to the transform gate, $\mathbf{c}_{\ell}(t) = 1 - \mathbf{t}_{\ell}(t)$, and
only $\mathbf{h}_{\ell}$ and $\mathbf{t}_{\ell}$ are computed.

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State) Dimension
  - `depth`: Number of highway micro-layers applied per time step. Default is 3.

## Keyword Arguments

  - `couple_carry`: Couples the carry gate to the transform gate. Default `true`.
  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable. Default set to `false`.
  - `init_bias`: Initializer for the biases of each of the `depth` micro-layers. If set to
    `nothing`, each bias is initialized from a uniform distribution within
    `[-bound, bound]`, where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
  - `init_weight`: Initializer for the weights of each of the `depth` micro-layers. If set to
    `nothing`, each weight is initialized from a uniform distribution within
    `[-bound, bound]`, where `bound = inv(sqrt(out_dims))`. Default is `nothing`.
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

  - `weights`: Tuple of `depth` weight matrices, one per micro-layer. The first maps from
    `in_dims + out_dims` (input concatenated with the running state), the remaining
    `depth - 1` map from `out_dims` alone.
  - `biases`: Tuple of `depth` bias vectors (if `use_bias=true`)
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct RHNCell{TS <: StaticBool} <: AbstractSingleRecurrentCell{TS}
    train_state::TS
    in_dims <: IntegerType
    out_dims <: IntegerType
    depth <: IntegerType
    num_gates <: IntegerType
    couple_carry::Bool
    init_bias
    init_weight
    init_state
    use_bias <: StaticBool
end

function RHNCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType},
        depth::IntegerType=3;
        couple_carry::Bool=true, use_bias::BoolType=True(),
        train_state::BoolType=False(), init_bias=nothing, init_weight=nothing,
        init_state=zeros32)
    depth > 0 || throw(ArgumentError("depth must be a positive integer; got $depth"))
    num_gates = couple_carry ? 2 : 3
    return RHNCell(static(train_state), in_dims, out_dims, depth, num_gates,
        couple_carry, init_bias, init_weight, init_state, static(use_bias))
end

function initialparameters(rng::AbstractRNG, rhn::RHNCell)
    out_dims, num_gates = rhn.out_dims, rhn.num_gates
    weights = ntuple(rhn.depth) do l
        in_l = l == 1 ? rhn.in_dims + out_dims : out_dims
        init_rnn_weight(rng, rhn.init_weight, out_dims, (num_gates * out_dims, in_l))
    end
    ps = (; weights)
    if has_bias(rhn)
        biases = ntuple(
            _ -> init_rnn_bias(rng, rhn.init_bias, out_dims, num_gates * out_dims),
            rhn.depth)
        ps = merge(ps, (; biases))
    end
    has_train_state(rhn) &&
        (ps = merge(ps, (hidden_state=rhn.init_state(rng, out_dims),)))
    return ps
end

initialstates(rng::AbstractRNG, ::RHNCell) = (rng=Utils.sample_replicate(rng),)

function parameterlength(rhn::RHNCell)
    out_dims, num_gates = rhn.out_dims, rhn.num_gates
    weight_len = sum(1:(rhn.depth)) do l
        in_l = l == 1 ? rhn.in_dims + out_dims : out_dims
        num_gates * out_dims * in_l
    end
    bias_len = has_bias(rhn) ? rhn.depth * num_gates * out_dims : 0
    return weight_len + bias_len
end

statelength(::RHNCell) = 1

function (rhn::RHNCell)(
        (inp, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple)
    matched_inp, matched_state = match_eltype(rhn, ps, st, inp, state)
    biases = safe_getproperty(ps, Val(:biases))
    out_dims = rhn.out_dims
    current_state = matched_state
    for l in 1:(rhn.depth)
        layer_inp = l == 1 ? vcat(matched_inp, current_state) : current_state
        bias_l = biases === nothing ? nothing : biases[l]
        pre = fused_dense_bias_activation(identity, ps.weights[l], layer_inp, bias_l)
        hidden_gate = tanh_fast.(view(pre, 1:out_dims, :))
        transform_gate = sigmoid_fast.(view(pre, (out_dims + 1):(2out_dims), :))
        current_state = if rhn.couple_carry
            @. (hidden_gate - current_state) * transform_gate + current_state
        else
            carry_gate = sigmoid_fast.(view(pre, (2out_dims + 1):(3out_dims), :))
            @. hidden_gate * transform_gate + current_state * carry_gate
        end
    end
    return (current_state, (current_state,)), st
end

function Base.show(io::IO, rhn::RHNCell)
    print(io, "RHNCell($(rhn.in_dims) => $(rhn.out_dims)")
    rhn.depth == 3 || print(io, ", $(rhn.depth)")
    has_bias(rhn) || print(io, ", use_bias=false")
    rhn.couple_carry || print(io, ", couple_carry=false")
    has_train_state(rhn) && print(io, ", train_state=true")
    return print(io, ")")
end
