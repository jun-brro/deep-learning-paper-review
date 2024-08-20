## MOT 기존 접근 방식

MOT는 비디오 시퀀스 내에서 객체의 궤적을 추적하면서 객체의 Identity를 유지하는 것을 목표로 한다. 기존 대부분의 접근법은 개별 프레임에서 객체를 검출한 후, 프레임 간 검출 결과를 correlate 하는 과정을 진행하였다. *(Traditional tracking-by-detection methods associate detections via temporally sparse [23,26] or dense [19,22] graph optimization, or apply convolutional neural networks to predict matching scores between detections [8,24].)*

## TrackFormer

![Untitled](https://github.com/user-attachments/assets/8a99bb2f-ab35-4688-8007-15c4e60dcf1e)

### MOT as a set prediction problem

TrackFormer는 Set prediction 으로써 MOT Task를 진행한다. 비디오 시퀀스에서 K개의 spearate object identity를 가지는 경우, MOT는 bbox, track identity k를 가진 순서 있는 트랙 T_k = (b_t1, b_t2, ...)를 생성하게 된다. 이때, 총 프레임 T의 부분집합 (t1,t2,...)은 object가 장면에 들어오고 나가는 time range를 나타낸다. 이때 가려지는 Object 상황에 대해서도 포함한다. MOT를 set prediction으로 바꾸어서 처리하기 위해, Enc-Dec transformer를 활용한다. (DETR의 개념과 유사하다고 이해할 수 있음)

1. Frame-level Feature Extraction - 일반적인 CNN backbone 사용
2. Frame Feature Encoding - 추출된 feature을 Encoder self-attention 통해서 encode
3. Query Decoding - Decoder self-attention & encoder-decoder attention
4. Mapping of Queries to Box and Class Predictions - Query → BBOX + Class Mapping

### Tracking-by-attention with queries

TrackFormer는 tracking-by-attention을 통해 Object Tracking, Detection을 동시에 진행할 수 있게 하는 것이 가장 핵심적인 개념이다. Static Object Queries는 비디오의 어느 프레임에서도 트랙을 초기화할 수 있게 한다. Autoregressive Track Queries는 프레임 간 object tracking을 수행한다. Object Query와 Track Query를 동시에 decode함으로써 모델은 detection & tracking을 통합하여 수행할 수 있게된다

**[Track Initialization]**

새롭게 등장하는 object는 일정 수의 N_object 출력 임베딩에 의해 감지된다. 각 임베딩은 pretrained static object queries를 이용하여 initialize할 수 있다. 각 Object query는 static object를 예측할 수 있도록 하긋ㅂ되게 된다. Decoder self-attention은 중복 검출을 피하고 spatial & categorical object를 추론하는 데 사용된다.

**[Track Queries]**

DETR에서의 Object Query는 특정 사진에 대한 identity를 갖고 있기 때문에, 비디오 처리에서 Object Query를 적용하기 위해서는 temporal한 축을 대입해야하는 상황이다. Trackformer에서는 이러한 프레임 간 트랙 생성을 위해 Decoder에 ‘Track Object Query’ 개념을 도입한다. Track Query는 Sequence 개념이 적용된 Object Query를 통해 Object Tracking을 진행하며, Object Identity를 유지하면서 autoregressive하게 location tracking을 진행한다. 새로운 Object Detection이 발생할 때마다 이전 프레임의 Output Embedding과 대응되는 Track Query를 초기화하게 된다.

**[Track Query Re-identification]**

Track Query를 randomly decode 할 수 있는 능력은 attention 기반의 short-time re-detection 과정을 가능하게 한다. 이전에 제거된 track query는 최대 T_track−reid 프레임 동안 decode된다. 이 기간 동안 track query는 deactivated, 분류 점수가 σ_track−reid를 초과할 때까지는 track에 변화를 일으키지 않는다. 각 track query에 포함된 공간 정보는 단기적인 관점에서의 track loss 를 해결한다.

### TrackFormer Training

TrackFormer가 Object query와 상호 작용하여 object를 다음 프레임으로 track하기 위해서는 frame tracking training이 필요하다. 이러한 구조적인 필요성을 해결하기 위해 two adjacent frames에서 트레이닝을 수행하고 전체 MOT에 최적화한다. class & bbox prediction에서 모든 출력 임베딩 N =ㅡN_object + N_track의 set prediction에 대한 loss를 모두 계산하여 프레임 t에 대한 Loss로 통합한다.

**[Bipartite matching]**

Ground truth object를 predicted object에 매핑하는 과정에서 track identity, bbox similarity, class accuracy를 기반으로 한 cost를 고려하게 된다. track identity의 경우, 프레임 t의 ground truth track identity K_t⊂K를 나타낸다. 각 검출은 프레임 t−1 track identity set K_t−1⊂K에서 해당하는 track identity k에 할당된다. 해당 출력 임베딩 (Track Query)는 identity를 다음 프레임으로 넘기게 된다 (Object Query의 개념을 Track Query로 옮김으로써 가능한 task임).

<img width="362" alt="Untitled 1" src="https://github.com/user-attachments/assets/f3a55acc-6c4a-40b3-96fa-6f46d886cb51">

**[Set Prediction Loss]**

<img width="306" alt="Untitled 2" src="https://github.com/user-attachments/assets/78fbdb26-2f19-44d2-ad7e-8dbbdacd1131">

<img width="445" alt="Untitled 3" src="https://github.com/user-attachments/assets/0a895f61-e9f3-475b-adae-3868d4946ec3">

Query에 대한 loss는 class loss, bbox loss를 모두 고려한다 (DETR과 동일, 헝가리안 알고리즘과 동일한 로스 계산 구조를 가지고 있음)

**[Track Augmentations]**

*The two-step loss computation, see (i) and (ii), for training track queries represents only a limited range of possible tracking scenarios.*

1. Frame Range Sampling - 프레임 t−1을 t 주변의 다양한 프레임에서 샘플링하여 객체가 이전 위치와는 아예 다른 별개의 위치에 존재하는 challenging frame pair를 생성한다. → 시간에 대한 종속성 문제를 해결 (특정 시간에 특정 위치에 있다고 잘못된 추론을 하게 되는 가능성을 배제하여 성능을 올릴 수 있다
2. False Negatives Sampling - 확률 p_FN로 Track query를 제거하여 false negatives 샘플링을 한다. 프레임 t의 해당 ground truth object는 object query와 매칭되어 new object detection을 유도하게 된다 → 비디오에서 출현하는 특정 물체들에 피팅되지 않고, generalize된 성능 개선을 유도한다?
3. False Positives Sampling: occlusion scenario에서 removal of track probelm을 개선하기 위해 false positives를 track query set에 추가한다. → 2번 증강 방식과 다른 개념의 attacking 개념

## Evaluation

TrackFormer는 MOT17, MOT20, MOTS20 벤치마크에서 높은 성능을 입증하였다. MOT17의 public/private detection processing에서 뛰어난 성능을 보였으며, MOT20에서도 추가적인 trajectory 데이터 없이 높은 성능을 달성하였다.

![Untitled 4](https://github.com/user-attachments/assets/9ce7a906-c9b7-4a45-b7dd-d4e01ffa9310)

![Untitled 5](https://github.com/user-attachments/assets/78b1ecce-636d-43a7-b985-b9428fc64048)

![Untitled 6](https://github.com/user-attachments/assets/e6de2f49-e342-4a85-b249-f8907fa334e5)
