# **Audio Spectrogram Transformer (AST)**

## Structure

### 입력 및 특징 변환

**오디오 입력 처리**

- **오디오 웨이브폼**: 오디오 처리 모델의 입력은 일반적으로 시간에 따라 변화하는 오디오 신호(waveform)
- **길이**: 입력 오디오의 길이는 t초 → 최종 스펙트로그램의 크기 영향

**특징 추출**

- **멜 필터뱅크 (Log Mel Filterbank)**: 오디오 신호에서 주요 특징을 추출하기 위해 멜 스케일을 기반으로 한 필터뱅크를 사용. 이 필터뱅크는 인간의 귀가 특정 주파수 범위에서 소리를 인지하는 방식을 모방하였다.
    
<img width="654" alt="Untitled" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/42ff8ede-7bf2-4f62-a0ad-75a909eb5d3d">
    
<img width="470" alt="Untitled 1" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/e225e111-7ff4-4cbc-9256-7db968cb44ff">
    
<img width="499" alt="Untitled 2" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/a7ba6b2e-526c-45da-b1a1-6a767d79aca9">
    
[오디오 데이터 전처리 (4) Mel Filter Bank](https://hyunlee103.tistory.com/46)
    
- **차원**: 추출된 특징은 128차원으로, 각 차원은 특정 주파수 밴드의 에너지 레벨이다.

### 시간-주파수 변환

- **윈도우 및 스텝**: 25ms의 해밍 윈도우를 사용하여 10ms 간격으로 슬라이딩하면서 오디오 신호를 분석. 해밍 윈도우는 신호를 부드럽게 해서 스펙트럼의 누설(leakage)을 감소시키는 데 유용하다.

[FFT 연산을 수행할 때 어떤 윈도우를 적용해야 할까요?](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=lecroykorea&logNo=221549211257)

- **스펙트로그램 크기**: 결과적으로 생성되는 스펙트로그램은 128×100t의 크기를 가진다. (각 t초 동안 100개의 시간 스텝)
1. **스펙트로그램의 패치 분할**
    - 스펙트로그램을 시간과 주파수 차원에서 6의 중복을 가진 16×16 패치로 나눔.
    - 각 패치는 1차원 패치 임베딩(크기 768)으로 평탄화됨.
    - 트랜스포머가 입력 순서 정보를 포착하지 않으므로, 각 패치 임베딩에 훈련 가능한 위치 임베딩(크기 768) 추가.
2. **트랜스포머 구조**
    - [CLS] 토큰을 시퀀스 시작 부분에 추가. AST는 분류 작업용으로 인코더만 사용.
    - 사용된 트랜스포머 인코더는 768의 임베딩 차원, 12개의 레이어 및 12개의 헤드를 가짐.
3. **출력 및 분류**
    - 트랜스포머 인코더의 [CLS] 토큰 출력이 오디오 스펙트로그램의 표현으로 사용됨.
    - 시그모이드 활성화를 가진 선형 층이 오디오 스펙트로그램 표현을 레이블로 매핑하여 분류 수행. 
    *A linear layer with sigmoid activation maps the audio spectrogram representation to labels for classification.*
4. **이미지넷 사전 훈련의 활용**
    - 트랜스포머가 CNN보다 더 많은 데이터를 필요로 함.
    - 이미지와 오디오 스펙트로그램이 유사한 포맷을 가지므로, 비전 작업에서 오디오 작업으로의 교차 모달성 전이 학습을 적용.
    *audio datasets typically do not have such large amounts of data, which motivates us to apply cross-modality transfer learning to AST since images and audio spectrograms have similar formats. Transfer learning from vision tasks to audio tasks has been previously studied in [23, 24, 25, 8], but only for CNN-based models, where ImageNet-pretrained CNN weights are used as initial CNN weights for audio classification training.*
5. **ViT와의 아키텍처 차이점 및 적응**
    - ViT는 3채널 이미지 입력을 받는 반면, AST는 단일 채널 스펙트로그램 입력을 받음.
    *First, the input of ViT is a 3-channel image while the input to the AST is a single-channel spectrogram, we average the weights corresponding to each of the three input channels of the ViT patch embedding layer and use them as the weights of the AST patch embedding layer. This is equivalent to expanding a single-channel spectrogram to 3channels with the same content, but is computationally more efficient.*
    
<img width="864" alt="Untitled 3" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/a7de8e3a-f16c-4e37-91c2-b18004fea6c6">
    
[Vision Transformer](https://nuguziii.github.io/survey/S-007/)
    
- ViT의 위치 임베딩을 잘라내고 이차원 보간을 통해 AST에 맞게 조정.
        
  *While the Transformer naturally supports variable input length and can be directly transferred from ViT to AST, the positional embedding needs to be carefully processed because it learns to encode the spatial information during the ImageNet training. We propose a cut and bi-linear interpolate method for positional embedding adaptation.*
        
- ViT의 최종 분류 층을 제거하고 AST용으로 새로 초기화.
