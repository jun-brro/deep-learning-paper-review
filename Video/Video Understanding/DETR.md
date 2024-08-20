# DETR

# End-to-End Object Detection with Transformers (DETR)

**Citations**
[Youtube - DSBA Lab](https://youtu.be/q1wSykClIMk?si=fFxHKqCrNkuOcW9F)
[HerbWood Tistory](https://herbwood.tistory.com/26)

Object detection은 이미지 내의 물체를 검출하고, 해당 물체의 클래스와 위치를 예측하는 태스크이다. 기존의 object detection 모델은 크게 one-stage와 two-stage 디텍터로 나뉜다. One-stage 디텍터는 대표적으로 YOLO(You Only Look Once)로, 속도가 빠르고 실시간 응용에 적합하다. 반면, two-stage 디텍터는 Faster R-CNN과 같은 모델로, 일반적으로 높은 정확도를 보이지만 속도가 느리다.

트랜스포머 모델은 NLP를 위해 고안되었으나, DETR에서는 이를 vision task에 적용하였다.  인코더-디코더 구조를 기반으로 하며, Attention 메커니즘을 사용하여 입력 데이터의 Global 정보를 학습할 수 있는 트랜스포머의 특징을 활용한다. 하지만 여기서 ViT와 다른 점은, ViT는 이미지의 패치를 그대로 인코더에 넣는다면 DETR에서는 CNN을 이용하여 나온 image feature를 사용한다는 것이다.

### DETR의 네트워크 구조

![Untitled](https://github.com/user-attachments/assets/ce5f48c6-73d6-45d6-9c0d-ecd0ec88dd47)

<img width="754" alt="Untitled 1" src="https://github.com/user-attachments/assets/192780ef-86dd-46e8-8340-868ab95e7a5f">


*첫 번째로 DETR는 encoder에서 이미지 feature map을 입력받는 반면, **Transformer는 문장에 대한 embedding을 입력받는다.** Transformer는 sequence 정보를 입력받기에 적합하기 때문에 DETR은 CNN backbone에서 feature map을 추출한 후, 1x1 convolution layer를 거쳐 차원을 줄이고, spatial dimension을 flatten하여 encoder에 입력한다. 예를 들어, 높이 h, 너비 w, 채널 수 C의 feature map을 d * hw 로 변환하여 입력한다. 여기서 d는 C보다 작은 채널 수이다.*

*두 번째로 positional encoding에서 차이가 있다. Transformer는 입력 embedding의 순서와 상관없이 동일한 값을 출력하는 permutation invariant한 성질을 가지므로 positional encoding을 더해준다. **DETR은 x, y axis가 있는 2D 크기의 feature map을 입력받기 때문에 기존의 positional encoding을 2D 차원으로 일반화하여 spatial positional encoding을 수행한다.** 입력값의 차원이 d일 때, x, y 차원에 대해 row-wise, column-wise로 2d 크기의 sine, cosine 함수를 적용한다. 이후 channel-wise하게 concat하여 d 채널의 spatial positional encoding을 얻은 후 입력값에 더해준다.*

*세 번째로 Transformer는 decoder에 target embedding을 입력하는 반면, **DETR은 object queries를 입력한다. object queries는 길이가 N인 학습 가능한 embedding이다.***

*네 번째로 Transformer는 decoder에서 첫 번째 attention 연산 시 masked multi-head attention을 수행하는 반면, **DETR은 multi-head self-attention을 수행한다.** Transformer는 auto-regressive하게 다음 token을 예측하기 때문에 attention 연산 시 후속 token에 대한 정보를 masking하여 활용하지 못하게 한다. 이 과정은 attention 연산에서 softmax 함수 입력에 후속 token 위치에 -inf를 입력하는 방식으로 수행된다. 그러나 DETR은 입력된 이미지에 동시에 모든 객체의 위치를 예측하기 때문에 별도의 masking 과정이 필요 없다.*

*마지막으로 Transformer는 decoder 이후 하나의 head를 가지는 반면, **DETR은 두 개의 head를 가진다.** Transformer는 다음 token에 대한 class probability를 예측하기 때문에 하나의 linear layer를 가지지만, DETR은 이미지 내 객체의 bounding box와 class probability를 예측하기 위해 각각을 예측하는 두 개의 linear layer를 가진다.*

<img width="815" alt="Untitled 2" src="https://github.com/user-attachments/assets/df9e71cc-d966-4a45-a083-3b2af76cf2eb">

### Encoder

이미지는 CNN을 통해 feature map으로 변환되어 트랜스포머 인코더의 인풋으로 들어간다 (하지만 기존 CNN과 다르게 global informations 확보).

![Untitled 3](https://github.com/user-attachments/assets/7031c6c2-b347-4211-9fb2-f2d39af866c8)

- 특정 픽셀에 있는 모든 피처를 갖고옴
- 2d fixed sine positional encoding

![Untitled 4](https://github.com/user-attachments/assets/1df7e4f1-0a3d-4012-9607-0e3651b387ff)

![Untitled 5](https://github.com/user-attachments/assets/7d2be02d-f5b9-4183-aa13-2f017f5acf9a)

### Decoder

![Untitled 6](https://github.com/user-attachments/assets/5ebb6651-946c-4481-af71-a98fce6fed53)

- 트랜스포머 디코더는 object query를 입력으로 받아, 해당 쿼리가 이미지의 특정 부분을 디텍션하도록 학습한다.
- 디코더에서는 Positional embedding은 의미가 없음 (Object queries는 병렬적, 독립적이라서 위치 정보가 의미가 없다) → Object query에 대한 output Positional encoding

<Object Queries> - 정보를 담기위한 그릇 (slot) → attention을 이용한 학습

![Untitled 7](https://github.com/user-attachments/assets/7ddd7b31-f831-4120-9613-6c2512d8e47d)

![Untitled 8](https://github.com/user-attachments/assets/761b8b48-78b5-4ef8-8dca-ea12d0f438ba)

DETR는 Bipartite Matching이라는 개념을 도입해서 예측값과 ground truth를 매칭하는 알고리즘을 가지고 있다. 이를 통해 각 object query와 실제 객체를 매칭시킨다. 이 과정은 Hungarian알고리즘을 통해 최적화되며, 클래스 예측 비용과 박스 예측 비용을 기반으로 수행된다.

<img width="836" alt="Untitled 9" src="https://github.com/user-attachments/assets/ff3e72db-4684-46ae-9d2b-8a0ffae1c482">

<img width="734" alt="Untitled 10" src="https://github.com/user-attachments/assets/892d5b73-3089-4cea-b170-96a9e9bda122">

### Loss function

1. 각 object query가 예측한 클래스 확률과 실제 클래스 간의 차이를 나타낸다. 이는 일반적인 crossentropy loss를 사용하여 계산된다. **(classification loss)**
2. 예측한 박스 좌표와 실제 박스 좌표 간의 차이를 측정한다. 여기에는 L1 loss와 Generalized IoU loss가 포함된다. L1 loss는 예측한 박스와 실제 박스 간의 절대적인 차이를 계산하며, Generalized IoU loss는 박스 간의 중첩 영역을 기반으로 차이를 측정한다. 여기서 IoU는 두 바운딩 박스 사이 겹치는 영역에 대한 비율이다. 하지만 이 경우 박스가 전혀 겹치지 않는 경우에 대해서는 위치의 차이에 관계없이 0이 되기 때문에 적절한 수치가 될 수 없다. 때문에 추가적인 항을 도입하여 GIoU라는 개념을 도입해서 사용한다. **(BBOX Loss)**

<img width="750" alt="Untitled 11" src="https://github.com/user-attachments/assets/a13435a8-7719-43c5-b4c4-69e7358f0707">

### 성능 평가 및 실험 결과

DETR의 성능 평가와 실험 결과는 기존 object detection 모델과의 비교를 통해 이루어졌다. 특히 여기서는 Object detection model evaluation에 널리 쓰이는 COCO 데이터셋을 사용하여 성능을 평가하였다.

평가지표로는 Average Precision (AP)과 Average Recall (AR)을 사용하였다. AP는 Precision-Recall 곡선 아래의 면적을 나타내며, 높은 AP 값은 모델이 높은 정확도와 재현율을 동시에 달성했음을 의미한다. AR은 주어진 임계값에서 모델이 얼마나 많은 객체를 올바르게 탐지했는지를 나타낸다.

- DETR은 큰 물체에 대한 detection에 강점을 보였다. 이는 트랜스포머의 전역적 Attention 메커니즘이 큰 물체의 전체적인 형태와 특징을 잘 학습할 수 있기 때문으로 해석된다. COCO 데이터셋에서 큰 물체에 대한 AP는 기존의 Faster R-CNN보다 높은 수치를 기록하였다.
- 작은 물체에 대한 성능은 상대적으로 낮았다. 이는 CNN의 특성상 작은 물체에 대한 특징을 잘 학습하지 못하기 때문이다. 특히 작은 물체에 대한 AP는 다른 모델에 비해 다소 낮은 결과를 보였다. 이는 Multiscaling 을 효과적으로 학습하지 못한 트랜스포머의 한계로 분석된다.
- DETR의 학습 시간은 기존의 Faster R-CNN에 비해 더 오래 걸렸다. 트랜스포머의 Attention 메커니즘은 초기 학습 단계에서 classification하기까지 속도가 오래 걸렸다. 다만 추론 단계에서는 비교적 시간이 적게 소요되었다.
- Bipartite 매칭을 통해 중복 없는 일대일 매칭이 가능해짐에 따라, 오탐지(false positive)와 미탐지(false negative)의 비율이 감소하였다. 이를 통해 모델의 전반적인 정확도를 향상시킬 수 있었다.

<img width="905" alt="Untitled 12" src="https://github.com/user-attachments/assets/16e83512-6ec4-492c-827a-5e3b01cdcb88">
