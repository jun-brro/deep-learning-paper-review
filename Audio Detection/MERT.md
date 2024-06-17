# MERT - ACOUSTIC MUSIC UNDERSTANDING MODELWITH LARGE-SCALE SELF-SUPERVISED TRAINING

# Base: Masked Language Modeling

## **How is MLM different from Word2Vec?**

Similar to masked language modeling and CLM, Word2Vec is an approach used in NLP where the vectors capture the semantics of the words and the relationships between them by using a neural network to learn the vector representations.

However, Word2Vec differs from self-supervised training models such as masked language modeling in the following ways:

- Word2Vec is an unsupervised learning algorithm that's used to generate word embeddings.
- It captures the syntactic and semantic links between words by representing them as dense vectors in a continuous vector space.
- Word2Vec acquires word embeddings by training on large corpora and predicting the context of words within a designated text window, encompassing either the target word itself or the surrounding words.
- It can be trained using two different algorithms -- [Continuous Bag of Words](https://medium.com/@codethulo/understanding-the-continuous-bag-of-words-cbow-model-architecture-working-mechanism-and-math-78c7284a8d5a) and Skip-Gram.
- Word2Vec embeddings are often used to measure word similarity or as input features for downstream natural language processing tasks.

[What are Masked Language Models (MLMs)? | Definition from TechTarget](https://www.techtarget.com/searchenterpriseai/definition/masked-language-models-MLMs)

# Introduction

## Self-supervised Learning

*Pre-trained language models (PLMs) can learn generalisable representations of data without human annotated labels in a self-supervised learning (SSL) style, leading to remarkable performance improvement in natural language processing and related fields (Brown et al., 2020; Fang et al., 2022; Chen et al., 2021a). →* PLM (Pre-trained Language Model) 기반 유사성 판단은 음악 시퀀스 적용에 유망하다.

*First, PLMs can potentially pave the way to unify the modelling of a wide range of music understanding, or the so-called Music Information Retrieval (MIR) tasks, including but not limited to music tagging, beat tracking, music transcription, and source separation, so that different tasks no longer need task-specific models or features.*

*Unfortunately, we are yet to see a general-purpose and cost-effective open-source PLM on acoustic music understanding. Most existing studies are designed to solely address music tagging problems (Pons and Serra, 2019; Spijkervet and Burgoyne, 2021; McCallum et al., 2022; Huang et al., 2022; Zhu et al., 2021; Zhao and Guo, 2021), and **many of them do not provide open-source code bases or checkpoints for further evaluation.***

![Untitled](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/4ebb6c73-adac-48fd-9eaa-a34d824fd1c3)

*MERT inherits a speech SSL paradigm, employing teacher models to generate pseudo targets for sequential audio clips. Specifically, to capture the distinctive pitched and tonal characteristics in music, MERT incorporates a multi-task paradigm to balance the acoustic and musical representation learning as demonstrated in Fig. 1.*

## RVQ-VAE (Residual Vector Quantization Variational AutoEncoder)

[What is Residual Vector Quantization?](https://www.assemblyai.com/blog/what-is-residual-vector-quantization/)

[[논문리뷰] Autoregressive Image Generation using Residual Quantization (RQ-VAE-Transformer)](https://kimjy99.github.io/논문리뷰/rq/)

## CQT (Constant-Q Transformation)

- *CQT is a type of frequency transform that is widely used in various MIR tasks, such as pitch detection, chord recognition, and music transcriptions*

[Constant-Q transform](https://en.wikipedia.org/wiki/Constant-Q_transform)

[[DL] 딥러닝 음성 이해 - Introduction to sound data analysis](https://heeya-stupidbutstudying.tistory.com/entry/DL-딥러닝-음성-이해-Introduction-to-sound-data-analysis)

*To summarise, our contributions are:*

- *proposing a multi-task style predictive acoustic self-supervised learning paradigm, which achieves SOTA performance on various MIR tasks, including important yet unexplored tasks for pre-training such as pitch detection, beat tracking and source separation applications;*
- *conducting a broad range of analysis based on ablation studies of the proposed MERT pretraining paradigm;*
- *exploring robust and stable strategies for acoustic music model training to overcome training instability and frequent crashes when scaling up the pre-training on model size;*
- *providing an open-source, generalisable and computationally affordable acoustic music pretrained model, which addresses the needs of both industry and research communities.*

---

# Related Work

## PLMs for Acoustic Music

*Existing acoustic music pre-trained models primarily focus on tagging tasks and rely on supervised tagging labels for pre-training (Pons and Serra, 2019; Spijkervet and Burgoyne, 2021; McCallum et al., 2022; Huang et al., 2022).*

*they face limitations in training data and model size, hampering the performance improvements (Choi et al., 2017; Li et al., 2022). Additionally, several models trained on inaccessible datasets or without publicly available codes and model weights make it difficult to reproduce or extend their approaches (McCallum et al., 2022; Castellon et al., 2021; Li et al., 2022; Zhu et al., 2021; Zhao and Guo, 2021).*

## Self-Supervised Speech Processing

*both acoustic music and speech processing models need to deal with the cocktail party problem (Brown and Bidelman, 2022; Petermann et al., 2022) since good source separation capabilities help both separating noises and background sounds with speech and processing polyphonic music audio.*

### Cocktail Party Problem

[NCBI - WWW Error Blocked Diagnostic](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2692487/)

코텔 파티 문제(Cocktail Party Problem)는 소리 신호 처리 및 인공지능 분야에서 중요한 문제로, 사람들이 여러 대화가 겹치는 소음 속에서 원하는 대화를 식별하는 상황을 다루고 있다. 이 문제는 실제 코텔 파티에서 많은 사람들이 동시에 대화하는 상황을 상상해 보면 이해하기 쉽다. 이 상황에서 우리는 한 명의 목소리에 집중하여 다른 모든 소음을 배제하고자 한다.

컴퓨터 과학 및 신호 처리에서는 이러한 문제를 "블라인드 소스 분리(Blind Source Separation)"라고도 부르며, 한 가지 음성 신호를 여러 소스에서 분리하고 원하는 신호만을 추출하는 작업을 의미한다. 이 문제는 인간의 청각 능력에 비해 기계로 해결하는 데 있어 상당한 어려움이 있다.

해결 방법으로는 독립 성분 분석(Independent Component Analysis, ICA), 파동 변환, 딥러닝 기반 모델 등 다양한 기술이 활용된다. ICA는 서로 독립적인 신호 성분을 분리하는 방법이고, 딥러닝 기반 모델은 대규모 데이터셋을 학습시켜 패턴 인식 및 음성 분리에 사용된다.

## Audio Representation with Language Modelling

*Mask strategy-based large-scale language models have been applied to a wide range of domains (Lample and Charton, 2019; Chen et al., 2021a;b; Fang et al., 2022), but **still remain under-explored in acoustic music understanding.***

*Baevski and Mohamed (2020) introduce a pre-trained VQ-VAE (Baevski et al., 2019) to provide prediction targets to conduct speech representation learning with MLM. While introducing K-means to provide discrete token codebooks and pre-training the model to detect sound units, **Hsu et al. (2021) claim that a better teacher model in SSL could lead to better downstream task performance.***

*the recently released **RVQ-VAEs (Zeghidour et al., 2021; D´ efossez et al., 2022), achieving good results in music reconstruction, could be adopted as teacher models for music understanding pre-training and provide acoustic information guidance.***

---

# Methodology

## Pre-Training with MLM

![Untitled 1](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/2d05f4c4-573b-43ed-8772-003218055edd)

![Untitled 2](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/9d889361-88d5-4d84-9180-4e841fd66e0d)

### HuBERT

![Untitled 3](https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/3d347f1a-85d9-49b7-aa4d-3989e879d849)

[[논문리뷰 | Speech] HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units (2021) Summary](https://velog.io/@9e0na/논문리뷰-Speech-HuBERT-Self-Supervised-Speech-RepresentationLearning-by-Masked-Prediction-of-Hidden-Units-2021-Summary)

## Modelling Acoustic Information

Mel-Frequency Cepstral Coefficients (MFCCs) are **only capable at** modeling acoustic and single-pitch signals

[MFCC(Mel-Frequency Cepstral Coefficient) 이해하기](https://brightwon.tistory.com/11)

### Method 1. Using K-means on the log-Mel spectrum & Chroma Features

- In case of music, each frame contain more informations than that of speech. → Larger number of classes is needed!
- *The complexity of the k-means algorithm is linear with the number of centroids, leading to a time-consuming k-means for the music feature.* → 300-means for log-Mel spectrum with dimension 229, and 200-means for Chroma features with dimension 264 → **Computational complexity remains comparable to that of HuBERT**
- Disadvantage
    - difficult to scale up to a larger number of classes and larger datasets
    - results are sensitive to initialization

### Method 2. EnCodec (8-layer residual VQ-VAE)

- Each acoustic features are denoted as 2-dimensional auditory code matrix with L (length of the recording)
- *Converts 24kHz input waveforms to 8 different embeddings at 75Hz with a 320-fold reduction, and the quantizer has 1024 dimensions → Decoder of Encodec can reconstruct the waveform at 24kHz with authentic information in timbre.*

[High Fidelity Neural Audio Compression (EnCodec)](https://ostin.tistory.com/206)

[[논문 리뷰] VQ-VAE: Neural Discrete Representation Learning](https://velog.io/@dien-eaststar/논문-리뷰-SSL-Neural-Discrete-Representation-Learning)

## Modeling Musical Information

- CQT Spectrogram is used solely for pitch-level information, regardless of acoustic information. (Similar to Fourier Transformation)
- Bin widths are proportional to frequency → **giving each octave the same number of bins**

![Uploading Untitled 4.png…]()

---

- **다운스트림 작업:** 14개의 다운스트림 작업을 대상으로 평가하며, 음악 태깅, 키 탐지, 장르 분류, 감정 점수 회귀, 악기 분류, 피치 분류, 보컬 기술 탐지, 가수 식별과 같은 프레임 수준의 작업과 박자 추적, 소스 분리와 같은 순차 작업을 포함한다.
- **프로빙 프로토콜:** 백본 모델을 고정한 채 간단한 다운스트림 구조만 훈련하여 평가하며, 하이퍼파라미터 검색 공간도 제한한다.
- Baseline Model:
    - **음악:** MusiCNN, CLMR, MULE, Jukebox 및 JukeMIR
    - **스피치:** HuBERT, data2vec

---

# Appendix

## DownStream Tasks

We evaluate the models on 14 downstream tasks to provide a comprehensive view of our method and the comparison between baselines. The full descriptions of the datasets and tasks are given as follows.

- Music Tagging involves determining which of a set of fixed tags apply to a particular song. Tag categories may include genre, instrumentation, mood, tempo (e.g. fast) or other tags. We used two large datasets: MagnaTagATune (MTT) (Law et al., 2009) and MTG-Jamendo (Bogdanov et al., 2019). For both datasets, we limit the tag vocabulary according to official instructions. We use all clips in MTTandMTG-Jamendo. Since many of the audio recordings among 5.5k MTG-Jamendo excerpts are longer than the 30s, we averaged the multiple embeddings computed with a sliding window as the overall embedding. The window length is set to the same default length as in every system. For MERT series, the window length is typically set to 30s. The metrics are the macro-average of ROC-AUCs and the average precision (AP) / PR-AUC among all top-50 tags.
- Key detection predicts the tonal scale and dominant pitch level of a song. We use Giantsteps (Knees et al., 2015) as test set and a commonly-used subset of Giantsteps-MTG-keys dataset (Korzeniowski and Widmer, 2017) as the training and validation set. The splitting is the same as in (Castellon et al., 2021). The metric is a refined accuracy with error tolerance, giving partial credit to reasonable errors (Raffel et al., 2014).
- Genre classification estimates the most appropriate genre for each given song. We report the accuracy of the GTZAN (Tzanetakis and Cook, 2002) dataset along with ROC and AP on MTG-Genre, since the former task is a multi-class classification and the latter is multi-label. We used the standard ”fail-filtered” split (Kereliuk et al., 2015) for GTZAN.
- Emotion score regression. The Emomusic dataset (Soleymani et al., 2013) contains 744 music clips of 45 seconds in length, each reported on a two-dimensional valence-arousal plane after listening, where valence indicates positive and negative emotional responses, and arousal indicates emotional intensity. We use the same dataset split as (Castellon et al., 2021). The official evaluation metric is the determination coefficient (r2) between the model regression results and human annotations of arousal (EmoA) and valence (EmoV) (Soleymani et al., 2013). For inference, we split the 45-second clip into a 5-second sliding window and averaged the prediction.
- Instrument classification is the process of identifying which instruments are included in a given sound. We use the Nsynth (Engel et al., 2017) and MTG-instrument datasets. The former is a monophonic note-level multi-class task with 306k audio samples in 11 instrument classes with accuracy as an indicator. The latter is a subset of MTG-Jamendo, containing 25k polyphonic audio tracks and 41 instrument tags; each track can contain multiple instruments and is evaluated on ROC and AP.
- Pitch classification estimates which of the 128 pitch categories the given audio segment belongs to. Weuse the NSynth dataset for this task. Given these segments are short monophonic audio, this task is multi-class, and the accuracy is used as an evaluation metric.
- Vocal technique detection involves identifying what singing techniques are contained in a given audio clip. We use the VocalSet dataset (Wilkins et al., 2018), which is the only publicly available dataset for the study of singing techniques. The dataset contains the vocals of 20 different professional singers (9 female and 11 male) who perform 17 different singing techniques in various contexts for a total of 10.1 hours. As the audio clips are divided into 3 seconds, the task only requires a judgement on the type of technique and not on the start and end of the technique. We used the same 10 different singing techniques as in Yamamoto et al. (2022) as a subset and used the same 15 singers as the training and validation sets and 5 singers as the test set. Since there is no accepted division between training and validation sets, we selected 9 singers as the training set and 6 singers as the validation set. All the 3-second segments that originate from the same recording are allocated to the same part of the split (e.g. all are in the training set). The evaluation metric is accuracy. 16Published as a conference paper at ICLR 2024
- Singer identification identifies the vocal performer from a given recording. We use the VocalSet dataset for this task. We randomly divided the dataset into a training set, validation set and testing set based on a ratio of 12:8:5, all containing the same 20 singers.
- Beat tracking is the process of determining whether there is a beat in each frame of a given piece of music. We use an offline approach to the binary classification, i.e. the model can use information following each frame to help with inference. The model needs to output frame-by-frame predictions at a certain frequency and post-process them using a dynamic Bayesian network (DBN) (B¨ ock et al., 2016b) to obtain the final result. The DBN is implemented using madmom (B¨ ock et al., 2016a). The dataset we use is GTZAN Rhythm (Marchand and Peeters, 2015). We also label the two adjacent frames of each label as beat, which is a common way of label smoothing in beat tracking to improve the performance of the model and to compare the SSL model fairly with the spin model. The model is evaluated using the f measure implemented in mir eval (Raffel et al., 2014), and the prediction is considered correct if the difference between the predicted event and the ground truth does not exceed 20ms. In this task, some models were trained on other datasets, and the full GTZAN set was used as the test set.
- Source separation. Source separation aims to demix the music recording into its constituent parts, e.g., vocals, drums, bass, and others. We adopt MUSDB18 (Rafii et al., 2017), a widely used benchmark dataset in music source separation. MUSDB18 contains 150 full-length music tracks (˜ 10 hours), along with multiple isolated stems. We use 86 tracks for training, 14 tracks for validation, and 50 tracks for evaluation following the official setting in MUSDB18. During training, we randomly sample 6-second segments and apply random track mixing for augmentation. Due to the difficulty of this task, we adopt the baseline architecture in the Music Demixing Challenge (MDX) 2021 (Mitsufuji et al., 2022), which consists of three linear layers and three bi-directional LSTM layers. We directly compute the l2-loss between predicted and ground-truth spectrograms for optimisation. The metric for this task is the Source-to-Distortion Ratio (SDR) defined by MDX 2021 (Mitsufuji et al., 2022), which is the mean across the SDR scores of all songs.

이 글은 음악 관련 인공지능 모델의 성능을 평가하기 위한 14개의 다운스트림 작업에 대해 설명합니다. 각각의 작업에 사용된 데이터셋, 평가 지표 및 방법을 설명하고 있습니다. 주요 작업은 다음과 같습니다:

1. **음악 태깅:** 특정 곡에 적용할 수 있는 장르, 악기, 분위기, 템포 등의 태그를 결정하는 작업입니다. 데이터셋으로는 MagnaTagATune(MTT)와 MTG-Jamendo를 사용하며, 평가 지표는 ROC-AUC와 PR-AUC의 매크로 평균입니다.
2. **키 감지:** 노래의 조성과 음계 레벨을 예측하는 작업입니다. Giantsteps 데이터셋과 Giantsteps-MTG-keys 서브셋을 사용합니다. 평가 지표는 정확도와 오차 허용치에 따른 수정된 정확도입니다.
3. **장르 분류:** 노래의 장르를 예측하는 작업으로, GTZAN 데이터셋의 정확도 및 MTG-Genre 데이터셋의 ROC 및 AP 지표를 보고합니다.
4. **감정 점수 회귀:** Emomusic 데이터셋을 사용하여 노래의 감정을 두 차원인 정서와 각성도로 평가합니다. r^2 값을 사용해 모델 결과와 인간 평가자 간의 결정 계수를 측정합니다.
5. **악기 분류:** Nsynth와 MTG-instrument 데이터셋을 사용하여 음악에 포함된 악기를 분류합니다. Nsynth는 단일 악기의 음을 분류하는 다중 클래스 작업이며, MTG-instrument는 다중 레이블 태스크로 ROC 및 AP 지표를 사용합니다.
6. **음정 분류:** Nsynth 데이터셋을 사용하여 128개의 음정 중 해당하는 음정을 예측하는 작업입니다.
7. **보컬 기법 감지:** VocalSet 데이터셋을 사용하여 음악에 포함된 보컬 기법을 감지합니다. 17개의 기법 중 10개를 선택하여 정확도 지표로 평가합니다.
8. **가수 식별:** VocalSet 데이터셋을 사용하여 노래하는 가수를 식별합니다.
9. **박자 추적:** GTZAN Rhythm 데이터셋을 사용하여 각 프레임에 박자가 있는지 확인하는 작업입니다. 모델은 프레임별로 예측하며, 평가 지표는 f-measure를 사용합니다.
10. **음원 분리:** MUSDB18 데이터셋을 사용하여 음악을 구성 요소별로 분리합니다. 최종 결과는 SDR(Source-to-Distortion Ratio)로 평가합니다.
