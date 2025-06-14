#https://arxiv.org/pdf/1611.01578
@doc raw"""
    NASCell(in_dims => out_dims;
        use_bias=true, train_state=false, train_memory=false,
        init_bias=nothing, init_recurrent_bias=nothing,
        init_weight=nothing, init_recurrent_weight=nothing,
        init_state=zeros32, init_memory=zeros32)

[Neural Architecture Search unit](https://arxiv.org/pdf/1611.01578).

## Equations
```math
\begin{aligned}
    \mathbf{o}_1(t) &= \sigma\left( \mathbf{W}_{ih}^{(1)} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{(1)} + \mathbf{W}_{hh}^{(1)} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{(1)} \right), \\
    \mathbf{o}_2(t) &= \text{ReLU}\left( \mathbf{W}_{ih}^{(2)} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{(2)} + \mathbf{W}_{hh}^{(2)} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{(2)} \right), \\
    \mathbf{o}_3(t) &= \sigma\left( \mathbf{W}_{ih}^{(3)} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{(3)} + \mathbf{W}_{hh}^{(3)} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{(3)} \right), \\
    \mathbf{o}_4(t) &= \text{ReLU}\left( \left( \mathbf{W}_{ih}^{(4)}
        \mathbf{x}(t) + \mathbf{b}_{ih}^{(4)} \right) \circ \left(
        \mathbf{W}_{hh}^{(4)} \mathbf{h}(t-1) + \mathbf{b}_{hh}^{(4)} \right)
        \right), \\
    \mathbf{o}_5(t) &= \tanh\left( \mathbf{W}_{ih}^{(5)} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{(5)} + \mathbf{W}_{hh}^{(5)} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{(5)} \right), \\
    \mathbf{o}_6(t) &= \sigma\left( \mathbf{W}_{ih}^{(6)} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{(6)} + \mathbf{W}_{hh}^{(6)} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{(6)} \right), \\
    \mathbf{o}_7(t) &= \tanh\left( \mathbf{W}_{ih}^{(7)} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{(7)} + \mathbf{W}_{hh}^{(7)} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{(7)} \right), \\
    \mathbf{o}_8(t) &= \sigma\left( \mathbf{W}_{ih}^{(8)} \mathbf{x}(t) +
        \mathbf{b}_{ih}^{(8)} + \mathbf{W}_{hh}^{(8)} \mathbf{h}(t-1) +
        \mathbf{b}_{hh}^{(8)} \right), \\
    \mathbf{l}_1(t) &= \tanh\left( \mathbf{o}_1(t) \circ \mathbf{o}_2(t)
        \right), \\
    \mathbf{l}_2(t) &= \tanh\left( \mathbf{o}_3(t) + \mathbf{o}_4(t) \right), \\
    \mathbf{l}_3(t) &= \tanh\left( \mathbf{o}_5(t) \circ \mathbf{o}_6(t) \right), \\
    \mathbf{l}_4(t) &= \sigma\left( \mathbf{o}_7(t) + \mathbf{o}_8(t) \right), \\
    \mathbf{l}_1(t) &= \tanh\left( \mathbf{l}_1(t) + \mathbf{c}(t-1) \right), \\
    \mathbf{c}(t) &= \mathbf{l}_1(t) \circ \mathbf{l}_2(t), \\
    \mathbf{l}_5(t) &= \tanh\left( \mathbf{l}_3(t) + \mathbf{l}_4(t) \right), \\
    \mathbf{h}(t) &= \tanh\left( \mathbf{c}(t) \circ \mathbf{l}_5(t) \right)
\end{aligned}
```

## Arguments

  - `in_dims`: Input Dimension
  - `out_dims`: Output (Hidden State & Memory) Dimension

## Keyword Arguments

  - `use_bias`: Flag to use bias in the computation. Default set to `true`.
  - `train_state`: Flag to set the initial hidden state as trainable.  
    Default set to `false`.
  - `train_memory`: Flag to set the initial memory state as trainable.  
    Default set to `false`.
  - `init_bias`: Initializer for input-to-hidden biases  
    $\{ \mathbf{b}_{ih}^{(1)}, \mathbf{b}_{ih}^{(2)}, \dots, \mathbf{b}_{ih}^{(8)} \}$.  
    Must be a tuple containing 8 functions. If a single value is passed, it is
    copied into an 8-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`, where
    `bound = inv(sqrt(out_dims))`. The functions are applied in order to
    initialize $\mathbf{b}_{ih}^{(1)}$ through $\mathbf{b}_{ih}^{(8)}$.
    Default set to `nothing`.
  - `init_recurrent_bias`: Initializer for hidden-to-hidden biases  
    $\{ \mathbf{b}_{hh}^{(1)}, \mathbf{b}_{hh}^{(2)}, \dots, \mathbf{b}_{hh}^{(8)} \}$.  
    Must be a tuple containing 8 functions. If a single value is passed, it is
    copied into an 8-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = inv(sqrt(out_dims))`. The functions are applied in order to
    initialize $\mathbf{b}_{hh}^{(1)}$ through $\mathbf{b}_{hh}^{(8)}$.
    Default set to `nothing`.
  - `init_weight`: Initializer for input-to-hidden weights  
    $\{ \mathbf{W}_{ih}^{(1)}, \mathbf{W}_{ih}^{(2)}, \dots, \mathbf{W}_{ih}^{(8)} \}$.  
    Must be a tuple containing 8 functions. If a single value is passed, it is
    copied into an 8-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = inv(sqrt(out_dims))`. The functions are applied in order to
    initialize $\mathbf{W}_{ih}^{(1)}$ through $\mathbf{W}_{ih}^{(8)}$.
    Default set to `nothing`.
  - `init_recurrent_weight`: Initializer for hidden-to-hidden weights  
    $\{ \mathbf{W}_{hh}^{(1)}, \mathbf{W}_{hh}^{(2)}, \dots, \mathbf{W}_{hh}^{(8)} \}$.  
    Must be a tuple containing 8 functions. If a single value is passed, it is
    copied into an 8-element tuple. If set to `nothing`, weights are initialized
    from a uniform distribution within `[-bound, bound]`,  
    where `bound = inv(sqrt(out_dims))`. The functions are applied in order to
    initialize $\mathbf{W}_{hh}^{(1)}$ through $\mathbf{W}_{hh}^{(8)}$.
    Default set to `nothing`.
  - `init_state`: Initializer for hidden state. Default set to `zeros32`.
  - `init_memory`: Initializer for memory. Default set to `zeros32`.

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

  - `weight_ih`: Input-to-hidden weights  
    ``\{ \mathbf{W}_{ih}^{(1)}, \mathbf{W}_{ih}^{(2)}, \dots, \mathbf{W}_{ih}^{(8)} \}``  
  - `weight_hh`: Hidden-to-hidden weights  
    ``\{ \mathbf{W}_{hh}^{(1)}, \mathbf{W}_{hh}^{(2)}, \dots, \mathbf{W}_{hh}^{(8)} \}``  
  - `bias_ih`: Input-to-hidden biases (if `use_bias=true`)  
    ``\{ \mathbf{b}_{ih}^{(1)}, \mathbf{b}_{ih}^{(2)}, \dots, \mathbf{b}_{ih}^{(8)} \}``  
  - `bias_hh`: Hidden-to-hidden biases (if `use_bias=true`)  
    ``\{ \mathbf{b}_{hh}^{(1)}, \mathbf{b}_{hh}^{(2)}, \dots, \mathbf{b}_{hh}^{(8)} \}``  
  - `hidden_state`: Initial hidden state vector (not present if `train_state=false`)  
  - `memory`: Initial memory vector (not present if `train_memory=false`)

## States

  - `rng`: Controls the randomness (if any) in the initial state generation

"""
@concrete struct NASCell{TS <: StaticBool, TM <: StaticBool} <:
                 AbstractDoubleRecurrentCell{TS, TM}
    train_state::TS
    train_memory::TM
    in_dims <: IntegerType
    out_dims <: IntegerType
    init_bias
    init_recurrent_bias
    init_weight
    init_recurrent_weight
    init_state
    init_memory
    use_bias <: StaticBool
end

function NASCell((in_dims, out_dims)::Pair{<:IntegerType, <:IntegerType};
        use_bias::BoolType=True(), train_state::BoolType=False(), train_memory::BoolType=False(),
        init_bias=nothing, init_weight=nothing, init_recurrent_weight=nothing,
        init_recurrent_bias=nothing,
        init_state=zeros32, init_memory=zeros32)
    init_weight isa NTuple{8} || (init_weight = ntuple(Returns(init_weight), 8))
    init_recurrent_weight isa NTuple{8} ||
        (init_recurrent_weight = ntuple(Returns(init_recurrent_weight), 8))
    init_bias isa NTuple{8} || (init_bias = ntuple(Returns(init_bias), 8))
    init_recurrent_bias isa NTuple{8} ||
        (init_recurrent_bias = ntuple(Returns(init_recurrent_bias), 8))
    return NASCell(static(train_state), static(train_memory), in_dims, out_dims,
        init_bias, init_recurrent_bias, init_weight, init_recurrent_weight,
        init_state, init_memory, static(use_bias))
end

function initialparameters(rng::AbstractRNG, nas::NASCell)
    return multi_initialparameters(rng, nas)
end

function parameterlength(nas::NASCell)
    return nas.in_dims * nas.out_dims * 8 + nas.out_dims * nas.out_dims * 8 +
           nas.out_dims * 16
end

function (nas::NASCell)(
        (inp,
            (state, c_state))::Tuple{
            <:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}},
        ps, st::NamedTuple)
    #type match
    matched_inp, matched_state, matched_cstate = match_eltype(
        nas, ps, st, inp, state, c_state)
    #get bias
    bias_ih = safe_getproperty(ps, Val(:bias_ih))
    bias_hh = safe_getproperty(ps, Val(:bias_hh))
    #gates
    full_gxs = fused_dense_bias_activation(identity, ps.weight_ih, matched_inp, bias_ih)
    full_ghs = fused_dense_bias_activation(identity, ps.weight_hh, matched_state, bias_hh)
    gates = full_gxs .+ full_ghs
    split_gates = multigate(gates, Val(8))
    #computation
    #first layer
    layer1_1 = sigmoid_fast.(split_gates[1])
    layer1_2 = relu.(split_gates[2])
    layer1_3 = sigmoid_fast.(split_gates[3])
    layer1_4 = relu.(split_gates[4])
    layer1_5 = tanh_fast.(split_gates[5])
    layer1_6 = sigmoid_fast.(split_gates[6])
    layer1_7 = tanh_fast.(split_gates[7])
    layer1_8 = sigmoid_fast.(split_gates[8])

    #second layer
    l2_1 = @. tanh_fast(layer1_1 * layer1_2)
    l2_2 = @. tanh_fast(layer1_3 + layer1_4)
    l2_3 = @. tanh_fast(layer1_5 * layer1_6)
    l2_4 = @. sigmoid_fast(layer1_7 + layer1_8)

    #inject cell
    l2_1 = @. tanh_fast(l2_1 + matched_cstate)

    # Third layer
    new_cstate = l2_1 .* l2_2
    l3_2 = @. tanh_fast(l2_3 + l2_4)

    new_state = @. tanh_fast(new_cstate * l3_2)
    return (new_state, (new_state, new_cstate)), st
end

function Base.show(io::IO, nas::NASCell)
    print(io, "NASCell($(nas.in_dims) => $(nas.out_dims)")
    has_bias(nas) || print(io, ", use_bias=false")
    has_train_state(nas) && print(io, ", train_state=true")
    known(nas.train_memory) && print(io, ", train_memory=true")
    print(io, ")")
end
