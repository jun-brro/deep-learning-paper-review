# SlowFast

비디오 인식은 이미지 인식과 달리 시간 도메인을 고려해야 한다.(행동이라는 정보 자체가 과거에서부터 얼마나 변화되었는지를 확인해야하는 요소이기 때문) 비디오 데이터는 시간 축을 포함하여 3차원 공간에서 움직임을 포착해야 하므로, 단순히 2D 이미지 처리 기술을 확장하는 것만으로는 충분하지 않다. 때문에 비디오 recognition task를 위해서는 기존 비전 처리기법을 프레임단위로 도입하면서도, 과거의 정보를 잃지 않도록 하는 구조가 필요하다. Neural Network의 착안을 신경세포에서 한 것 처럼, SlowFast 논문도 뇌가 행동을 인식하는 구조를 차용해온다.
Citation: [Joochan Github Blog](https://joochann.github.io/posts/SlowFast-Networks-for-Video-Recognition/)

![](https://github.com/user-attachments/assets/9a060284-ba46-4e58-93b5-8b59c6f4dd2d)

## SlowFast 네트워크 구조

![Untitled 1](https://github.com/user-attachments/assets/134a8f22-2460-4157-a1db-048665baca1e)

![Untitled 2](https://github.com/user-attachments/assets/4cc3e5b3-d3e6-4223-b2a3-1befade068cf)

![Untitled 3](https://github.com/user-attachments/assets/e8a529c4-ed4f-401c-9f9b-422fce9306ae)

SlowFast 네트워크는 인간의 시각 시스템에서 (M cell, P cell) 영감을 받은 모델이다. Slow Pathway는 고해상도의 공간 정보(spatial, semantic)를 처리하고, Fast Pathway는 움직임과 같은 시간 정보를 처리한다. 이러한 구조를 모방하여 SlowFast 네트워크는 두 가지 경로로 구성된다.

### Slow Pathway

Slow Pathway는 낮은 frame rate로 입력 데이터를 처리하면서, 고해상도의 공간 정보를 캡처하는 데 중점을 둔다. (컬러, 공간정보 등) slow pathway는 ResNet을 기반으로 설계되었으며, 이러한 방식으로 중요한 공간적 정보를 잃지 않으면서도 계산 효율성을 높일 수 있다.

### Fast Pathway

fast pathway는 높은 frame rate로 입력 데이터를 처리하며 주로 움직임과 같은 시간 정보를 캡처하는 데 중점을 둔다. fast pathway 또한 ResNet을 기반으로 설계되었지만, 더 많은 프레임을 조밀하게 샘플링한다. 이 경로에서는 채널 수를 줄여 계산 자원을 절약하면서도 중요한 모션 정보를 효과적으로 캡처할 수 있다.

<img width="675" alt="Untitled 4" src="https://github.com/user-attachments/assets/0d00197a-8c57-4adb-9d8b-1bb8d48175db">

<img width="722" alt="Untitled 5" src="https://github.com/user-attachments/assets/19357a02-e70e-437d-a59b-5142f5686793">

slow pathway, fast pathway의 정보를 결합하기 위해 두 경로 간의 lateral connection을 통해 정보를 교환한다. 이 교환은 상보적인 효과를 나타내어 서로 학습되지 않은 내용에 대한 보충을 할 수 있다. slow pathway는 고해상도의 공간 정보를 제공하고, fast pathway는 시간적인 변화를 포착하여 종합적인 비디오 인식을 가능하게 한다.

<img width="404" alt="Untitled 6" src="https://github.com/user-attachments/assets/837c45cd-13d1-4fde-bf83-2454b5542149">

## 실험 및 검증

SlowFast 네트워크는 다양한 대형 비디오 데이터셋에서 성능을 평가받았다. 주요 데이터셋으로는 Kinetics-400, Kinetics-600, Charades, AVA 등이 있다.

### Kinetics-400 / 600

Kinetics-400은 유튜브에서 수집한 400개의 클래스에 대한 비디오 데이터셋이다. 이 데이터셋은 240,000개의 트레이닝 비디오 클립과 20,000개의 검증 클립으로 구성되어 있다. Kinetics-600은 Kinetics-400의 확장 버전으로, 600개의 클래스와 더 많은 비디오 클립을 포함하고 있다.

<img width="966" alt="Untitled 7" src="https://github.com/user-attachments/assets/91a84661-54c8-48fe-8fe1-967174f31620">

### AVA

AVA(Atomic Visual Actions) 데이터셋은 비디오에서의 인간 행동을 고해상도로 라벨링한 데이터셋이다. 각 프레임에는 사람의 행동을 나타내는 바운딩 박스와 행동 태그가 포함되어 있다.

<img width="666" alt="Untitled 8" src="https://github.com/user-attachments/assets/69c74eef-704e-45ab-9c02-ad44e9cd9199">

### 주요 실험 결과

Kinetics-400 및 Kinetics-600 데이터셋에서 SlowFast 네트워크는 기존의 최첨단 모델을 능가하는 성능을 보였다. 특히, 3D 컨볼루션 기반의 모델들과 비교하여 더 높은 정확도와 효율성을 나타냈다. Charades 데이터셋에서는 다양한 행동을 동시에 인식하는 능력을 입증하였다. AVA 데이터셋에서는 행동 탐지에서도 뛰어난 성능을 보였다.
