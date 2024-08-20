## What is MAE?

verb - highly semantic
If removing verb, the sentence is challenging to understand. However, if masking just little part of high resolution image -> we can still understand image

text: high semantic elements

First idea - > Masking 75% (remaining 25%)
ViT -> Patches

- -->> same effect for shuffling all the patches and pick first 25% --> Different order --> Encoder output also different order --> unshuffling --> decoder(+ positional embed) --> **Predict the missing shape**

<img width="731" alt="Untitled" src="https://github.com/user-attachments/assets/ae339c63-f168-4aab-8fe6-cee3c83da2ef">

<img width="722" alt="Untitled 1" src="https://github.com/user-attachments/assets/c93df76a-e1cd-462b-af64-1f77656093a2">

Linear Layer -> Classification

<img width="418" alt="Untitled 2" src="https://github.com/user-attachments/assets/e2f00c2a-9202-4107-812a-39bc3818f470">

## VideoMAE

Adapting MAE Concept to Video  -> Adding temporal axis

<img width="691" alt="Untitled 3" src="https://github.com/user-attachments/assets/83189cda-a366-4af8-b137-ffb14776cb4c">

VideoMAE는 비디오 처리 모델을 학습하기 위한 하나의 방법으로, 쉽게 말해서 MAE의 개념을 비디오에 가지고 온 것이다. MAE는 자연어처리에서의 마스킹과 다른 비전에서의 특성을 반영하기 위해 만들어진 개념이라면, VideoMAE는 비전 대비 비디오의 특성을 반영하기 위해 만들어진 개념이라고 볼 수 있다. 여기서 비전 태스크 대비 비디오 모델의 특성이라 함은

1. 시간에 따른 불필요함 (Temporal Redundancy)
2. 시간에 따른 프레임별 상호 연관성 (Temporal Correlation)

비전 태스크에서는 고해상도 이미지에서 일어날 수 있는 상호 추론 가능성을 최대한 삭제하기 위해 ViT 입력 패치들을 모두 shuffle 한 후에, 특정 비율만 뽑아서 학습에 사용하면서 masking 효과를 내고, 데이터셋 자체에서 발생할 수 있는 추론 가능성을 최대한 배제한다.
비디오 모델에서는 비전 태스크 대비 새로운 개념들이 추가된다. 바로 '시간' 개념의 도입이다. 비디오 recognition에서는 비전 태스크에서의 처리 방법에, 시간에 따른 특성 또한 추론에 사용될 수 있다. 이러한 추론 가능성을 배제하기 위해 VideoMAE에서는 새로운 방안을 제시한다.

<img width="690" alt="Untitled 4" src="https://github.com/user-attachments/assets/eb9fd221-8b57-4ccf-a9ec-edd952b501e3">

### Temporal DownSampling

앞서 언급한 바와 같이, 비디오는 연속적인 프레임의 특성에서 오는 데이터셋에서의 추론 가능성이 존재한다. 이것을 배제하기 위해 구간 downsampling을 진행한다. Temporal Stride라는 개념을 도입하여 T개의 샘플에 대해서 compressed된 샘플링을 진행한다. 보통 stride는 Kinetics 데이터셋에 대해서는 4로, Something-Something에 대해서는 2로 설정된다.

### Cube Embedding

Video는 Vison Task에 Temporal 개념 또한 도입해야 하기 때문에, 이를 'Cube'라 칭할 수 있다. (ViViT에서도 같은 개념 차용) 이 때 차원은 T/2 * H/12 * W/16 3D 토큰 형태로 사용된다. 이렇게 공간 및 시간 차원을 감소시키면 불필요한 정보를 해결할 수 있다. (Alleviate the spatiotemporal redundancy in videos)

### Tube Masking with extremely high ratios

비디오에서는 프레임별로 랜덤하게 마스킹하게 되면, 오히려 마스킹되지 않은 패치들 끼리의 개연성이 떨어져서 추론에 불리하다. 그렇기 때문에 특정한 패치 위치에 대해서, 모든 시간 동안 마스킹을 하는 Tube Masking 기법을 사용한다. Cube를 모든 Time axis에 적용하면 Tube가 된다고 볼 수 있겠다. 여기서 high ratio를 적용하는 이유는, 비디오의 경우에는 정보의 밀도가 이미지에 비해 매우 작기 때문에 high ratio를 적용했을 때 reconstruction difficulty 를 높일 수 있기 때문이다.

### Evaluation & Result

<img width="678" alt="Untitled 5" src="https://github.com/user-attachments/assets/72042152-d225-4918-b603-bbba7b7188ea">

### Masking, Reconstruction

VideoMAE는 높은 마스킹 비율일 때 더 높은 성능을 나타낸다. 마스킹 비율을 75%에서 90%로 증가시켰을 때 SSV2 데이터셋의 성능이 68.0%에서 69.6%로 향상되었다. Kinetics-400 (K400) 데이터셋에서도 유사한 결과가 나타났으나, SSV2에 비해 성능 차이는 적었다. central frame만 재구성 목표로 사용할 경우 성능이 크게 감소하였으며, 작은 샘플링 간격 또한 성능을 떨어뜨리는 원인이 되었다.

### Pretrain / dataset

ImageNet-21K 데이터셋에서 사전 학습을 수행한 ViT-B는 SSV2와 K400에서 각각 32.6%에서 61.8%, 68.8%에서 78.9%로 성능이 향상되었다. ImageNet-1K로 1600 epoch 동안 사전 학습한 후, 비디오 데이터셋에서 파인튜닝을 수행한 모델은 처음부터 학습한 모델보다 더 나은 성능을 보였다. 하지만, 사전 학습된 ImageNet 데이터셋과 다른 비디오 데이터셋으로 전이 학습을 수행할 경우, 도메인 차이로 인해 성능이 다소 감소한 것으로 확인되었다.

### Transfer Learning / Downstream

VideoMAE는 K400에서 SSV2, UCF101, HMDB51 데이터셋으로의 전이 학습에서는 우수한 성능을 보였다. 특히, UCF101과 HMDB51 데이터셋에서는 pre trained된 모델보다 전이 학습된 모델의 성능이 더 우수했다. K400에서 학습된 VideoMAE를 AVA dataset에 전이한 결과, 평균 정확도(mAP)가 26.7에서 31.8로 증가했다.
