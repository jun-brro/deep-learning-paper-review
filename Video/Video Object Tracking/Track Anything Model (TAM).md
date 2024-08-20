# Introduction

VOT → VOS

Segmentation → massive human labor force, needing specific object mask groundtruth

‘SAM’ → Strong image segmentation ability, high interactivity with different kinds of prompts

→ 유동적인 (generalized) 프롬프트, 실시간 마스킹 및 생성 가능

‘Video Task’ → 타겟 정형화, 모션 blur, camera motion등 많은 요소를 추가적으로 고려할 필요가 있음

Suggesting ‘TAM’ : SAM + XMem (mask prediction of the object in the next frame)

# TAM - Methodology

![image](https://github.com/user-attachments/assets/15d29adb-8d4e-4290-a69c-fe8cbdc83ac1)

## Preliminaries

1. **SAM**
    - ViT trained by SA-1B (only for image segmentation)

vision task with prompt (Vision Segmentation + Prompt Processing)

<img width="1292" alt="image 1" src="https://github.com/user-attachments/assets/72174772-6758-4941-b488-ea876b890e36">

2. **XMem (XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model)**
    - 첫 프레임에서 mask description을 주면, 바로 따라오는 프레임에서 object를 추적하고 이에 적합한 마스크를 생성할 수 있다.
    - Atkinson-Shiffrin Memory Mode에서 영감을 받은 모델로, 인간의 기억 체계를 감각 기억, 단기 기억, 장기 기억의 세 단계로 나누어서 구현한다. 이를 영상 객체 분할에 적용한 XMem은 영상 내에서 객체 정보를 추적하기 위해 unified feature memory stores를 사용한다.
    - 단기 메모리는 가까운 프레임에서 객체 정보를 추적 및 저장하여, 객체가 프레임 간 이동하거나 변형될 때 필요한 정보를 저장하는 곳이다.
    - 장기 메모리는 장기적인 범위 내에서 프레임에서의 객체 정보를 저장하여, 객체가 일시적으로 프레임에서 사라지거나 배경과 혼동될 수 있는 상황에서도 올바른 추적을 진행할 수 있게 한다.

<img width="952" alt="image 2" src="https://github.com/user-attachments/assets/0b893d44-0039-4beb-a4ee-56c2e5d48379">

<img width="931" alt="image 3" src="https://github.com/user-attachments/assets/b1b1bf58-e761-4dbb-b1ae-b5e73f00d626">

Segmentation 기법이라 보기는 좀 애매하고, Segmentation된 정보를 비디오 전체에서 더 효율적으로 저장하기 위한 ‘Memory structure 개선’이라고 보는 것이 적절한 해석으로 보인다.

3. **Interactive Video Object Segmentation**
    - User interaction을 입력으로 받고, 원하는 입력이 나올 때 까지 Segmentation result를 수정할 수 있다.
    - SAM에서 prompt랑 다른 점은, SAM에서는 프롬프트를 가지고 이미지 오브젝트를 찾아내는 태스크라고 하면, Interactive VOS에서는 직접 object mask?를 선택할 수 있는 상황이다

## Implementation

### Step 1: Initialization with SAM

SAM 은 weak prompt (points, bounding boxes) 를 사용하여 segmentation을 할 수 있도록 하는 모델이다. 이를 사용하여 target object에 대한 Mask를 초기에 생성하게 된다. 

### Step 2: Tracking with XMem

위에서 생성된 마스크를 활용하여, 뒤에 오게되는 frame에서 SS VOS를 수행하게 된다. 만약 step 1에서 생성된 마스크가 의미있는 마스크가 아닐 경우, XMem 예측치와 함께 Probe, affinity등 중간자 파라미터를 저장한다.

### Step 3: Refinement with SAM

Inference 비디오 모델에 대해서는 일관적인 정확한 Mask를 예측하는 것이 굉장히 어려운 일이다. → XMem에서 예측한 품질이 낮다? → SAM을 사용하여 정제 (XMem에서 저장한 Probe, Affinity 등)을 사용하여 SAM 프롬프트로 변환하여 새로운 마스크를 생성하게 된다.

### Step 4: Correction with human participation

만약 이러한 과정을 거침에도 불구하고 의미있는 이해가 이루어지지 않는다면 인간의 개입을 활용하여 TAM이 비디오의 내용을 잘 이해할 수 있도록 할 수 있다.

# Evaluations

![image 4](https://github.com/user-attachments/assets/e1e63b93-4c51-42c7-bdc3-bf563094612b)

![image 5](https://github.com/user-attachments/assets/d298ff4f-5de3-4b1b-a4d2-917d71497aea)

## Quantitative Results (정량적 지표)

DAVIS-2016 val, DAVIS 2017 test set 활용 → J & F Score

![image 6](https://github.com/user-attachments/assets/623cb62e-bc64-44e3-97ea-fcb02f08219c)

![image 7](https://github.com/user-attachments/assets/321f806e-928e-42c6-a872-ad74939eb62a)

Jaccard 지수 (IoU)는 predicted segmentations과 ground truth 간의 겹치는 영역을 측정한 지표. → 공간적인 정확도를 평가할 수 있다

F-measure는 precision와 recall의 조화 평균으로, segmentation의 완전성(recall)과 정확성(precision)을 평가하는 데 사용된다

## Qualitative Results (정성적 지표)

![image 8](https://github.com/user-attachments/assets/75859b74-0463-4e34-9224-f282d390076e)

실제로 정성적으로 보았을때에도, 다양한 challenging 상황을 보여주었을 때에도 효과적으로 segmentation하고 있다

## Failed Case

![image 9](https://github.com/user-attachments/assets/6fadb6cc-1099-49e1-88ef-2ea6e8bdf18b)

1. Long term memory problem

대부분의 VOS 모델이 짧은 비디오에 최적화되어있기 떄문에 긴 비디오에서는 마스크의 정확도 및 크기가 줄어드는 상황이 벌어졌다. SAM Refinement가 이를 개선하는 방향성인지도 확인해보았으나 실질적인 영향은 미미하다.

1. Complex object structure

자전거 바퀴와 같이 굉장히 복잡한 구조를 가진 경우, 마스크가 초기에 정밀하게 설정되지 않으면 이 결과가 이후 프레임에도 이어서 나타나게 된다.
