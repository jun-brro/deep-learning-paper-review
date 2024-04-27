# YAI 봄 1주차

# Skip-Gram Model

The fat cat sat on the table → the central word ‘sat’ → predicting surrounding words.

How specify surroundings? → ‘window’

<img width="448" alt="Untitled" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/b64dd686-bc0e-499e-836f-f8b490b6856c">

The fat cat sat on the table → ‘sat’ is the central word, and predicting surrounding words using the central words. Then, how can we define ‘surrounding’? → “Window Size”

CBOW vs Skip-gram: CBOW only predict/train one word. Skip-gram understands the meaning of several words, and getting several outputs

Input: One-hot Vector

One hidden layer → ‘Shallow NN’

Skip-gram Formulation (using softmax)

<img width="594" alt="Untitled 1" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/8cca01c3-3095-44e0-9271-b21613e4fe14">

# Hierarchical Softmax

Using Softmax above is inefficient → Using the structure of binary Tree!

<img width="873" alt="Untitled 2" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/5f6cc637-36a1-4fdc-957a-429284253f6a">

<img width="683" alt="Untitled 3" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/24c2e14c-eb40-499f-8ded-dbb70420b82b">

- Increases computational efficiency by using a binary tree representation of the vocabulary for word prediction, instead of the entire vocabulary.
- Words are represented as leaf nodes of a binary tree, with each word being identified by the path from the root to the leaf.
- To predict a word, the model calculates the probability along the path from the root to the leaf node (target word), which is done in logarithmic time complexity.
- By calculating probabilities for nodes along the path instead of the entire vocabulary, it significantly reduces computational costs.
- Particularly useful in datasets with a very large vocabulary size, greatly reducing training time.

# Negative Sampling

<img width="343" alt="Untitled 4" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/8371e738-0303-4b54-bd57-43db97ca3ae0">

- Reduces computational costs by updating the model using a small number of "negative" samples (words not in the target context) instead of the entire output layer for every word in the vocabulary.
- **Selection Mechanism**: Selects one "positive" example (the target word) and a small number of "negative" examples (non-target words) for each training sample.
- The model is trained to increase the probability for the positive example and decrease it for the negative examples.
- Negative examples are randomly selected from the vocabulary, but the sampling probability can be adjusted based on the frequency of the words.
- Negative Sampling improves training speed significantly, especially on large datasets, and provides good performance even on smaller datasets.

### [Additional] NCE vs NEG

**Noise Contrastive Estimation (NCE)**

- **Goal**: Estimates model parameters by distinguishing target data from noise data (samples not from the target distribution).
- **Approach**: Introduces noise distribution and models the probability of data being from the true distribution rather than noise. Requires estimation of both the noise and true data distribution parameters.
- **Usage**: Beyond embedding models, NCE is applicable in various machine learning tasks where learning a complex, multi-class distribution is necessary.
- **Complexity**: Involves more complex calculations than NEG, as it estimates additional parameters related to the noise distribution.

**Negative Sampling (NEG)**

- **Goal**: Simplifies the training of embeddings by focusing on a small subset of "negative" samples rather than the full vocabulary for each training step.
- **Approach**: For each positive training sample, a small number of negative samples are randomly selected. The model learns by distinguishing positive samples from these negative examples.
- **Usage**: Primarily used in training word embeddings like Word2Vec, making it highly specific and efficient for such tasks.
- **Complexity**: Simpler than NCE as it directly modifies the objective function to focus only on a small subset of negative examples without estimating the noise distribution parameters.

# SubSampling of Frequent Words

‘a’, ‘the’, ‘in’, ‘of’ … → no big effect to the meaning! → Subsampling! (Not using the word during training)

<img width="170" alt="Untitled 5" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/9a6c21d9-3bb5-4125-a814-6b5e04838282">

where f(wi) is the frequency of word wi and t is a chosen threshold, typically around 10^−5.

# Validation

<img width="745" alt="Untitled 6" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/22e2c845-b88c-4ec3-8948-e80fcdd91c52">

Grammarly / Textual Similarity

![Untitled 7](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/373af9a7-1667-4d73-9576-a66e89edc0b0)

Result: Negative Sampling was way much better than hierarchical softmax, slightly better than nois contrastive estimation. If using subsampling, the training time dramatically reduces.

# Learning ‘Phrases’

‘Phrase’ = ‘One Token’

using ‘unigram’ and ‘bigram’ to find out which part works as ‘phrase’

<img width="511" alt="Untitled 8" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/824b44f7-b7ed-4385-9085-7c5a643d2759">

<img width="256" alt="Untitled 9" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/da087259-4efc-4108-a550-824d5768ae4b">

δ: Discounting Coefficient (Prevent constructing phrases with low-frequency words)

The authors repeated the training data 2 to 4 times to reduce the threshold value, δ, so that longer phrases could be formed. The idioms created in this way were also learned using the five categories of learning data below.

![Untitled 10](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/01cc7758-1d10-46b6-a0e6-31cf0e4ccca6)

Phrase skip-gram result: Hierarchical softmax had lower result when not using subsampling. → Subsampling can make faster, more accurate result

![Untitled 11](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/7f7283ba-ab1e-4712-b567-b1a86ef564e7)

The study found that increasing the training data to approximately 33 billion words and using hierarchical softmax with 1000 dimensions led to the best phrase analogy task results, achieving a 72% accuracy rate. This outcome underscores the significance of large datasets in improving model accuracy. 

# Additive Compositionality

![Untitled 12](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/16538042-36d1-45bb-8388-df2e273a034a)

This paper demonstrates that representations of words or phrases, when utilizing the Skip-gram model, can exhibit accurate performance in analogical reasoning despite their simplistic vector structure, proving that these are linear representations.

# Conclusion

- Trained distributed word and phrase representations with Skip-gram, enabling analogical reasoning.
- Techniques applicable to both Skip-gram and CBoW models.
- Improved representation quality through large-scale data training.
- Subsampling and Negative sampling enhance learning speed and accuracy.
- Key factors: model architecture, vector size, subsampling, and window size.
- Simplified text representation through vector addition and minimal complexity.
