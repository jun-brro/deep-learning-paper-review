## Overview of ViT

ViT는 이미지 처리를 위해 기존 자연어처리에서 사용된 트랜스포머의 개념을 차용해 온 것이다. ViT를 되짚어보면, overlape되지 않은 h * w 패치로 분리한 뒤에, linear projection + positional embedding을 사용하여 정보를 인코딩한다고 이해할 수 있다.

<img width="458" alt="Untitled" src="https://github.com/user-attachments/assets/d0ec859c-652a-4f70-8a80-f58644c773bf">

z는 토큰, E는 projection, MSA는 Multi-Headed Self Attention, LN은 Layer Normalization이다. MLP의 경우, non-linearity, token-dimensionality로 구분된 two linear projections으로 구성되는데, 이들은 모든 레이어를 통과하면서 고정된 상태를 유지한다. 이러한 연산 과정에서, 3차원의 개념을 도입하면 세 가지 연산 배열이 가능하다.

## Embedding Video Clips

- Uniform frame Sampling
    
<img width="413" alt="Untitled 1" src="https://github.com/user-attachments/assets/29eb9f27-1b1b-4117-a35d-a9cd0cf34c79">
    
- Tubelet Embedding
Uniform frame Sampling은 2D 프레임을 나누어 샘플링한다면, 비디오는 시간 축을 도입한 샘플링 기법을 사용한다.
Tubelet dimension이 작을 수록 토큰 수가 많이 필요하게 되고, 계산량이 많아지게 된다. 이 임베딩 방법은 위 Uniform frame sampling과 다르게 시공간 연관 정보를 얻을 수 있도록 도와준다.
    
<img width="392" alt="Untitled 2" src="https://github.com/user-attachments/assets/a1fcca23-195d-42bc-9e06-a7d377baa012">
    
<img width="337" alt="Untitled 3" src="https://github.com/user-attachments/assets/39f24ab2-e1b3-4d37-ae29-d9225e42bffc">
    

## Model Variation

<img width="696" alt="Untitled 4" src="https://github.com/user-attachments/assets/5f1c0413-b374-4590-9e87-f683d4f0f820">

### Spatio-temporal attention

비디오로부터 추출한 모든 spatio-temporal tokens을 transformer encoder에 입력한다. 각 transformer layer는 모든 spatio-temporal tokens 간의 쌍 상호작용을 모델링한다.(CNN에서 receptive field가 선형적으로 증가하는 현상에 대한 해결) 하지만 MSA는 토큰의 개수에 따라 복잡도가 제곱으로 증가하게 된다는 단점이 존재한다.

### Factorised encoder

<img width="420" alt="Untitled 5" src="https://github.com/user-attachments/assets/46b9d0f4-8eef-4518-9aac-06b108f316e3">

각 프레임별 공간 정보 인코딩 -> temporal 에다가 태움 -> 다 모아두고 Temporal 인코딩 처리 (two separate encoder)

### Factorised self-attention

<img width="408" alt="Untitled 6" src="https://github.com/user-attachments/assets/e01ad885-667b-4a8d-b8d7-af511c43af49">

위에서 나온 수식의 연장선이다. (temporal)

<img width="317" alt="Untitled 7" src="https://github.com/user-attachments/assets/a7033697-6be7-4e22-ab3d-55c9f637179d">

spatial self -attention -> temporal self attention (same encoder number as spatio-temporal attention)

공간과 시간 처리의 순서는 성능에 영향을 미치지 않는다. 이 모델의 경우 spatio-temporal attention과 달리, 추가적인 attention layer가 존재하기 때문에 cls토큰을 필요로 하지 않는다.

### Factorised dot-product attention

<img width="399" alt="Untitled 8" src="https://github.com/user-attachments/assets/46095cc1-d7e7-4559-85b1-b0bf0e279caf">

일반적인 multi-head 상황에서, 절반은 spatial, 나머지는 temporal 으로 K, V를 분리한다. 이후 concat를 한다.

<img width="363" alt="Untitled 9" src="https://github.com/user-attachments/assets/beb80f16-dac4-45ea-bd4f-b948d33d5c26">

## Initialization by leveraging pre-trained models

CNN과 달리, ViT는 큰 데이터셋에서만 효과적이고, inductive bias가 부족하다는 특징이 있다. 여기에 더해, 대부분의 비디오 데이터셋은 이미지 데이터셋에 비해 라벨링된 데이터가 훨씬 적다. 때문에 큰 모델을 처음부터 높은 정확도로 학습시키는 것은 매우 어렵고, 이를 위해 pretrained 이미지 모델을 도입할 수 있다. 위치 임베딩은 비디오모델이 이미지 모델에 비해 훨씬 많은 토큰을 가지고 있어, 위치 임베딩을 시간적으로 반복하여 초기화를 하게 된다.

## Result

### Dataset

- Kinetics - 유뷰트에서 샘플링된 10초 길이의 비디오 클립이다. Kinetics 400과 600이 있다.
- Epic Kitchens-100 - 주방 활동을 촬영한 100시간 분량의 비디오로, 각 비디오는 'verb'와 'noun'로 라벨링된다.
- Moments in Time: 다양한 동물, 사물, 사람, 자연 현상을 담은 800,000개의 3초 길이 유튜브 클립이다.
- Something-Something v2 (SSv2): 220,000개의 2~6초 길이 비디오로, 물체와 배경이 일관된 비디오로 세밀한 동작 패턴 인식이 중요하게 작용하는 데이터셋이다.

### Evaluations

<img width="428" alt="Untitled 10" src="https://github.com/user-attachments/assets/b2f7ddfb-2b9b-4a4c-ac80-eb02bad7ee5d">

ViViT-B 모델과 spatio-temporal attention을 사용하는 경우 Kinetics 400 데이터셋에서 centeral frame initialisation을 사용한 tublet embedding이 79.2%의 Top-1 정확도를 기록해 가장 우수한 성능을 보였다. 이는 filter inflation initialisation보다 1.6% 높고, uniform frame sampling보다 0.7% 높다.

Model Variants에서 ViViT-B를 기본 모델로 사용했으며, Kinetics 400과 Epic Kitchens를 사용하여 Evaluation을 진행했다. Kinetics 400에서는 unfactorized 모델(Model 1)이 80.0%의 Top-1 정확도로 최고 성능을 기록했지만, Epic Kitchens에서는 Factorized Encoder 모델(Model 2)이 43.7%의 action accuracy로 가장 우수했다. Model 2는 avg pooling을 사용하지 않고 temporal transformer를 사용한 결과, Kinetics 400에서 3%, Epic Kitchens에서 4.9%의 정확도 향상을 보였다.
