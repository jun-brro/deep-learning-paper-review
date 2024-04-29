# Introduction

- **Importance of Bidirectional Pre-training**
    - **Contextual Background**: Unlike previous models which were largely unidirectional, BERT leverages context from both directions, significantly enhancing language comprehension.
    - **MLM (Masked Language Model)**: Randomly masks some words and predicts these masked words based solely on their context, enabling effective bidirectional training.
- **Novel Pre-training Objectives**
    - **Next Sentence Prediction (NSP)**: Predicts whether two sentences logically follow each other, enhancing the model's understanding of sentence relationships. This is particularly beneficial for tasks like Question Answering (QA) and Natural Language Inference (NLI).
- **Simplification and Enhancement of Fine-tuning**
    - BERT provides a universal language model that can be finely adjusted for a wide range of NLP tasks, achieving high performance without the need for many task-specific architectures.

# Related Work

<img width="599" alt="Untitled" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/6d596cd3-bce5-44a3-95c9-de4007f2d2bb">

## Unsupervised Feature-based Approaches

- **Evolution from Non-neural to Neural Methods**: The journey from early non-neural approaches to advanced neural embeddings marks a significant advancement in creating effective word representations.
- **Pre-trained Embeddings' Impact**: Pre-trained word embeddings, developed through various pre-training objectives, have become essential, significantly enhancing NLP system performances over scratch-learned embeddings.
- **Extension to Sentence and Paragraph Embeddings**: The field has expanded to include embeddings for larger text units like sentences and paragraphs, employing innovative training objectives to capture broader context.

### **Embeddings from Language Model (**ELMO)

<img width="708" alt="Untitled 1" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/195d4ba1-38eb-41dd-b6cc-20bbc89073c1">

If words with the same notation can be word-embedded differently depending on the context, the performance of natural language processing can be improved. The idea of taking context into account when embedding words is called Contextualized Word Embedding.

- **Bi-LSTM**: A bidirectional long-term short-term memory network that reads text forward and backward to understand context in both directions, improving comprehension and prediction accuracy in tasks such as text analysis.
- **BiLM**: A bidirectional language model that improves word representation by considering the context of the surrounding text, and is the basis for ELMo by providing rich and contextualized word embeddings.

## Unsupervised Fine-tuning Approaches

Initial research in feature-based methods concentrated on pre-training word embeddings using unlabeled text. Recently, the focus has shifted to pre-training encoders for generating contextual representations of tokens within sentences or documents, followed by fine-tuning for specific tasks. This method is advantageous because **it minimizes the number of parameters needed to be learned from scratch.**

## Transfer Learning from Supervised Data

Studies have shown the effectiveness of transfer learning from large-scale datasets in tasks like machine translation and natural language inference (NLI). This approach has been emphasized not only in NLP but also in CV, where models pre-trained on datasets like ImageNet are known to perform well when fine-tuned.

# Bidirectional Encoder Representations from Transformers (BERT)

## Model Architecture

- BERT utilizes a multi-layer bidirectional Transformer encoder, following the design outlined by Vaswani et al. (2017). Its implementation closely mirrors the original, as described in the tensor2tensor library.
- Due to the widespread adoption of Transformers and the similarity of BERT's implementation to the original, a detailed description of the architecture is omitted, with references recommended to Vaswani et al. (2017) and resources like “The Annotated Transformer.”
- **Configuration Details**:
    - **BERTBASE**:
        - Layers (L): 12
        - Hidden Size (H): 768
        - Self-Attention Heads (A): 12
        - Total Parameters: 110M
    - **BERTLARGE**:
        - Layers (L): 24
        - Hidden Size (H): 1024
        - Self-Attention Heads (A): 16
        - Total Parameters: 340M
- **Model Size Parity**: BERTBASE's design mirrors that of OpenAI GPT to facilitate direct comparisons.
- **Self-Attention Mechanism Difference**: BERT uses bidirectional self-attention, enabling tokens to attend to the entire input sequence, contrasting with GPT’s unidirectional approach that limits attention to preceding tokens.

## Input/Output Representations

<img width="609" alt="Untitled 2" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/149df631-08a9-45bb-8664-6b968ae05b3d">

- **[CLS] Token**: Every input sequence begins with a [CLS] token. The final hidden state corresponding to this token aggregates sequence representations for classification tasks.
- **Sentence Pairs**: Input sequences can comprise a pair of sentences, separated by a [SEP] token. This structure is used for tasks involving sentence relationships.
- **Segment Embeddings**: To distinguish between the two sentences in a pair, segment embeddings are applied, labeling sentences as either "A" or "B."
- **Token Embeddings**: BERT employs WordPiece embeddings to represent individual tokens within the input.
- **Position Embeddings**: Follows the same approach as used in the Transformer model to account for the position of each token within the sequence.
- **Input Representation**: The final input representation is a combination of the corresponding token, segment, and position embeddings for each token in the input.

## Pre-training BERT

### Task #1 : Masked LM

![Untitled 3](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/d60468b8-066a-4bc8-a78d-a2b6c4dd7462)

Unlike Denoising Auto Encoders, which reconstruct the entire input, BERT focuses solely on predicting [MASK] tokens.

- **Mismatch Issue**: A mismatch between pre-training and fine-tuning arises because [MASK] tokens, used during pre-training, don't appear in the fine-tuning phase.

To mitigate this, additional strategies are applied to the 15% of tokens selected for masking:

- **80% of the time**: The token is replaced with a [MASK] token. For example, "my dog is hairy" becomes "my dog is [MASK]."
- **10% of the time**: The token is replaced with a random word. For example, "my dog is hairy" might turn into "my dog is apple."
- **10% of the time**: The token is left unchanged. For example, "my dog is hairy" stays as "my dog is hairy."

### Task #2 : Next Sentence Prediction (NSP)

![Untitled 4](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/2faf8540-48b6-4f66-8b71-329d29c6a0e4)

BERT uses Next Sentence Prediction (NSP) to learn sentence relationships, crucial for NLP tasks. It trains on sentence pairs, labeled as "IsNext" if they sequentially follow each other, or "NotNext" if unrelated. 

- **IsNext**: Input: "[CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] of milk [SEP]", Label: IsNext.
- **NotNext**: Input: "[CLS] the man [MASK] to the store [SEP] penguin [MASK] are flightless birds [SEP]", Label: NotNext.

This approach enhances BERT's understanding of sentence context and relationships.

### Pre-training Data

*For the pre-training corpus we use the BooksCorpus (800M words) (Zhu et al., 2015) and English Wikipedia (2,500M words).* 

## Fine-tuning BERT

- **Fine-tuning Flexibility**: BERT adapts to various NLP tasks by adjusting inputs and outputs, efficiently handling both single texts and text pairs.
- **Unified Encoding**: Leverages self-attention to process text pairs, offering integrated bidirectional cross attention, simplifying the encoding process.
- **Task Adaptation**: Directly integrates task-specific inputs and outputs, supporting a wide range of NLP tasks through end-to-end fine-tuning.
- **Efficient Process**: Achieves results within an hour on a Cloud TPU or a few hours on a GPU, demonstrating fine-tuning's cost-effectiveness.
- **Comprehensive Application**: Detailed task-specific applications and fine-tuning processes are documented, ensuring replicability and clarity.

# Experiments

## General Language Understanding Evaluation (GLUE)

<img width="601" alt="Untitled 5" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/1ff7afd6-95ac-4a22-80d6-6beb73f373a5">

## Stanford Question Answering Dataset (SQuAD)

<img width="471" alt="Untitled 6" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/302b7ea4-4632-47d1-8d1b-ac9c4fed81ae">

<img width="447" alt="Untitled 7" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/c9b917f5-8256-4175-b5c0-31f770f1e481">

## Situations With Adversarial Generations (SWAG)

<img width="304" alt="Untitled 8" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/17453d5b-ae3f-475a-88c0-6c295b762ff4">

# Ablation Studies

## Effect of Pre-training Tasks

### No NSP

- Removing NSP from BERT's pre-training tasks showed a decrease in performance on downstream tasks, indicating NSP's role in improving understanding of sentence relationships.

### LTR & No NSP

- Adopting a left-to-right (LTR) model without the Next Sentence Prediction (NSP) task significantly diminishes performance on key NLP benchmarks, underscoring the vital role of NSP in understanding sentence relationships.
- Bidirectional models far outstrip LTR models in performance, particularly in tasks that require comprehensive context understanding, such as question answering.
- Attempts to improve LTR models with a BiLSTM layer yield some benefits but fail to reach the effectiveness of bidirectional models, illustrating the inherent limitations of LTR approaches.
- Despite considering separate LTR and RTL models to emulate bidirectionality, this strategy is deemed inefficient and less effective than a unified bidirectional model, highlighting BERT's optimized approach for superior performance across a range of NLP tasks.

## Effect of Model Size

- **Model size variation**: The paper also explores how changing the model size, including the number of layers (depth), the size of hidden layers, and the number of attention heads, impacts performance.
- **Larger models perform better**: It was found that larger models generally achieve better results on a range of NLP tasks, underscoring the trade-off between computational resources and performance. BERTLARGE, with more layers and a greater number of parameters, significantly outperformed smaller versions.

## Feature-based Approach with BERT

<img width="295" alt="Untitled 9" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/ca8c3a7d-1a63-4486-9ac3-c735b84df216">

<img width="309" alt="Untitled 10" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/92f047e5-73fa-42ac-af9b-ecd3c3e1a537">
