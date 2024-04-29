# Introduction: Limitation of traditional DNN

DNN-based researches achieved steady success in terms of voice recognition and visual detection. However, there was a limitation that the sequential problem could not be properly solved because **the input size was fixed**. Due to the nature of language, which **inevitably has variable length**, it was difficult to solve this problem. Seq2Seq aims to solve this problem that the length of the same sentence varies depending on the language.

# Seq2Seq Model Structure

**Sequent usage of LSTM!**

<img width="593" alt="Untitled" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/4df5a6cb-dd25-4794-a9de-7ff72fa06552">

A model consisting of two LSTM-based structures, an Encoder and a Decoder, first reads the language through the Encoder and creates a **fixed-length Context Vector**. The end of each sentences are detected by using EOS(End of Sentence) token. Therefore, the input and output length can vary. When reading sentences through an encoder, they are read in reverse order. (It was way better than reading in forward order, proved by experimental result) The decoder produces an output that maximizes the conditional probability by considering the input, latent variable H, and context C.

### Question: How the length of context vector is fixed?

Answer: Since the context vector, which is the output of Encoder, also works as input of Decoder, the size should be fixed when building a model. Often use 256, 512, 1024, etc.

[Attention 메커니즘의 이해](https://gaussian37.github.io/dl-concept-attention/)

There was some trials to use RNN to handle sequence data.

<img width="294" alt="Untitled 1" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/ee71c941-4c14-4116-ad12-235cc949f27d">

RNNs work well when they know how inputs and outputs match up, especially if they are the same length. But, when inputs and outputs are different lengths and their connection isn't straightforward, it's hard to use RNNs. Furthermore, RNN is vulnerable to long term dependency, because Vanishing Gradient damages long-term dependency.

Therefore, LSTM is used to handle long-term dependency and get higher ability of the model.

<img width="410" alt="Untitled 2" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/aa8c98ce-fac8-44f2-b364-199fe7e1d6f2">

The LSTM achieves this by first generating a fixed-dimensional representation (v) of the input sequence (x_1 … x_T), represented by the final hidden state of the LSTM. Following that, it uses (v) as the initial hidden state for a standard LSTM-based language model (LSTM-LM) to calculate the probability of the output sequence (y_1 … y_T')

# Experiments

<img width="369" alt="Untitled 3" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/f8722669-4238-46b7-ac40-c421f665d780">

<img width="318" alt="Untitled 4" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/fbada0bb-7e93-420a-acb4-41fb8d8f5e66">

The target function we aim to optimize is the maximization of the log likelihood, which translates into maximizing conditional probability. This maximization involves dividing the sum of these probabilities by 1/∣*S*∣, where ∣*S*∣ represents the size of the entire training set. This approach indicates that the objective is to develop a model that performs better on average, rather than one that fits a few specific sentences too closely.

### Beam Search

When creating target sentence using decoder, beam search method is used.

![Untitled 5](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/677cbf2a-f62d-44c7-b6e2-4b4e021f5622)

Beam search is a search algorithm used for finding sequences with the highest probabilities. It maintains a fixed number of best options, called the "beam width," at each step, allowing it to explore more possibilities than a simple greedy search, which only keeps the single best option.

When creating the sentence, the seq2seq decoder select nodes with highest probability in descending order. Here, All probabilities considered in beam search are cumulative probabilities. Even if some child nodes have the same probability, the cumulative probability varies depending on which beam they come from.

### GPU Parallel Calculation

The calculation speed of LSTM is way slow to get meaningful result. Therefore, the study adapts parallelizing 8 GPU machines, to reduce total calculation time. *(Each layer of the LSTM was executed on a different GPU and communicated its activations to the next GPU / layer as soon as they were computed.)*

### BLEU Score**(Bilingual Evaluation Understudy Score)**

[14-03 BLEU Score(Bilingual Evaluation Understudy Score)](https://wikidocs.net/31695)

BLEU measures the correspondence between a machine's output and that of a human reference translation, based on the precision of n-grams (contiguous sequences of *n* items from a given sample of text) in the translated text compared to the reference.

<img width="654" alt="Untitled 6" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/a4772dc3-33ed-4a36-9ecb-7e75ebb4a100">

### Experiment Results (Reverse Reading)

<img width="777" alt="Untitled 7" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/195041e1-0c80-4eab-a37c-be7fb08edd8b">

Notable thing is that the model's performance is better when the words in the sentence are entered in reverse order. The paper assumes that reversing the word improves backpropagation. Reversing the source sentence doesn't change the average distance between corresponding words but significantly reduces this time lag for the initial words. This reduction makes it easier for backpropagation to link the source and target sentences, leading to notably better performance.

### Experiment Results (Long-term Dependency)

<img width="705" alt="Untitled 8" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/f03cfd79-912e-41d4-9f67-a76df8fc41cf">

<img width="564" alt="Untitled 9" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/5c534d7f-7ab0-4749-a838-963b6c9191fd">
