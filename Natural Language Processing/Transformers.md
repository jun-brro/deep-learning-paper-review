# Introduction

**Transformer:** entirely based on attention mechanism (no use of recurrence or convolution → Therefore, less training time and calculations)

**SOTA:** State of the Art (The highest result)

### Limit of Recurrent Model

- Parallelization is impossible (all sequences should be processed sequently)
- Extremely low ability when long sequence length. (The longer, the worse)

→ “The fundamental constraint of sequential computation” remains in case of recurrent model.

→ Transformers model is proposed, based on attention mechanism.

# Model Architecture

<img width="724" alt="Untitled" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/81e6ddca-ceca-4905-ac7a-dfa0d595e461">

### Encoder

- **Structure**: Consists of 6 identical layers, each with two sub-layers: a multi-head self-attention mechanism and a positionwise fully connected feed-forward network.
- **Residual Connections and Normalization**: Each sub-layer's output is added to its input (residual connection) and normalized, enhancing training stability.
- **Dimensionality**: Outputs from sub-layers and embedding layers are dimensionally consistent at *d*model=512.

### Decoder

- **Structure**: Mirrors the encoder with 6 identical layers, but includes an additional third sub-layer for multi-head attention over the encoder's output.
- **Masked Self-Attention**: The first sub-layer uses masked self-attention to ensure predictions for a position depend only on earlier positions.
- **Consistency and Adaptations**: Applies residual connections and layer normalization like the encoder, maintaining output dimensionality consistency.

### Attention (Scaled Dot-Product Attention)

<img width="495" alt="Untitled 1" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/801463d1-7795-4fe9-8c01-8b8b8a105f3c">

An attention mechanism maps a query along with a collection of key-value pairs to an output, with the query, keys, values, and the output all being represented as vectors.

Attention = Mapping Query & Key-Value pair to output (Output is calculated as weighted sum of values)

<img width="351" alt="Untitled 2" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/80646f25-8912-431f-86c4-f83e9ddd4691">

**Additive Attention:** Calculate the compatibility function using a feed-forward layer network with a single hidden layer

**Dot-Product Attention:** The Attention method used in this study

### Multi-Head Attention

<img width="462" alt="Untitled 3" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/d4a77ddc-95c9-49a8-85fb-927c7e4afc33">

→ Since the dimension is reduced at each head, total calculation cost is similar to that of single-head attention.

It was found out that linearly projecting to dk, dv, using differently trained (h-times trained) linear projections. Performing attention in parallel on these projections yields *dv*-dimensional outputs, addressing the scale issue in dot products between queries and keys.

### Applications of Attentions (Multi-head attention)

- In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder.
- The encoder contains self-attention layers.
- Self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position.

### Position-wise Feedforward Networks

Although the linear transformations are same for different positions, the parameters are different at each layers.

<img width="580" alt="Untitled 4" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/46db30d9-fb54-4826-8ece-74dba40e75ec">

### Embeddings and Softmax

- Using trained embeddings, in order to convert input and ouput tokens to dmodel vector.
- In the Transformer, to convert the decoder output into probabilities for the next token prediction, a linear transformation followed by a softmax is used. The model shares the same weight matrix across the two embedding layers and the pre-softmax linear transformation, with the embedding layers' weights being scaled by dmodel.

![Untitled 5](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/815a5759-166a-4552-9f4c-494eae2e02fe)

### [Additional] Embeddings vs Encodings

- **Embeddings**:
    - **Learnable**: Embeddings are vectors learned from data during the model training process. They can be optimized for specific tasks to learn semantic representations.
    - **Semantic Relationship Learning**: Embedding vectors are trained such that words or entities with similar meanings are located close to each other in the vector space.
    - **High-Dimensional Dense Vectors**: Embeddings typically represent data in dense vectors of several hundred to thousands of dimensions. Each dimension can represent specific semantic attributes.
- **Encodings**:
    - **Fixed Representations**: Encodings often transform data into vectors using predefined methods. They tend to remain unchanged during the training process (e.g., one-hot encoding).
    - **Structural/Positional Information**: Encodings can provide structural or positional information to the model.
    - **Sparse or Dense Vectors**: Encodings can take the form of sparse vectors (e.g., one-hot encoding) or dense vectors (e.g., positional encodings), depending on the encoding method used.

[Understanding Differences Between Encoding and Embedding](https://www.linkedin.com/pulse/understanding-differences-between-encoding-embedding-mba-ms-phd/)

### Positional Encoding

- Since Transformers does not use any of recurrent process or convolution, it is necessary to input the ‘positional data’ to train the order of sequences.

![Untitled 6](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/1f65ac3b-e113-49ec-88e7-4888914b87f4)

**Question: Why using sin, cos functions for positional encoding?**

- The position vector must not explode, to prevent data deterioration. Sin&cos function value is from -1 to 1, therefore satisfying the condition.
- Sine and cosine functions are periodic functions. For the sigmoid function, with long sequences of sentences, the difference in position vector values can become negligible. However, for trigonometric functions (sine & cosine), since they oscillate periodically between -1 and 1, even with long sequences of sentences, the difference in position vector values can be maintained significantly.

![Untitled 7](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/b58e91fa-ac46-4e5e-810d-1262d3e2fe38)

- If there is only a single sine or cosine function, the position vector value might be equal, for each different token. To prevent this case, sine and cosine functions are used with various periods. Therefore, the word vector can have different position vector for each dimensions.

![Untitled 8](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/45f062d0-17f7-4478-98f9-441b86f28cb8)

[트랜스포머(Transformer) 파헤치기—1. Positional Encoding](https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding)

# Why Self-Attention?

1. Total Computational complexity per layer
    - Self-Attention Layers lower total computational complexity per layer. This happens especially when sequence length is less than the dimensionality of the representations.
2. Amount of computation that can be parallelized
    - Self-attention allows for greater amount of computations that can be parallelized, minimizing the number of required sequential operations and reducing training & inference time.
3. The path length between long-range dependencies in the network
    - Self-attention mechanism shorten the path length, facilitating easier, more efficient learning of dependencies.

# Training / Result

### Training Setting

- Optimizer: Adam

<img width="699" alt="Untitled 9" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/f267f567-6357-48ad-98b3-1304ec558284">

- Three types of Regularization
    - Residual Dropout
        - Applying Dropout to outputs of each sublayers
        - Applying Dropout to the sum of embeddings and positional encoding
    - Label Smoothing

<img width="657" alt="Untitled 10" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/5aa0389a-e172-44e7-86ae-078fcb95fac4">

### Result

<img width="653" alt="Untitled 11" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/c47159de-d168-4e12-b208-4165d10ee2e7">

<img width="647" alt="Untitled 12" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/914a0020-0bc7-4401-814e-cdaa2dfca022">
