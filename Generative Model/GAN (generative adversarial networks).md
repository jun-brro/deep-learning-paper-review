# Week 1. GAN
작성날짜: 2024년 3월 12일
작성자: Junhyeong Park

<img width="679" alt="Untitled" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/fd4ea1ed-38e2-4033-b34d-6c3a2659261d">

<img width="927" alt="Untitled 1" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/33650f2e-7a7c-4901-9cec-14cc6ece89d8">


### 참고 링크
[[Paper Review] Generative Adversarial Networks](https://youtu.be/jB1DxJMUlxY?si=4XdV9bbvJ8RnnCnd)

[초짜 대학원생 입장에서 이해하는 Generative Adversarial Nets (1)](https://jaejunyoo.blogspot.com/2017/01/generative-adversarial-nets-1.html)

# Overview of GAN

Competing versus generative model & discriminative model

Generative Model: Aims to fake discriminative model

Discriminative Model: tries to determine whether the generated result is fake or true.

<img width="440" alt="Untitled 2" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/a27af60a-ca75-4322-be29-294996320fe0">



### ‘위조 지폐범’ vs ‘경찰’

<img width="749" alt="Untitled 3" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/e55ff32b-8f55-465c-a89a-3411de70fbac">

위조 지폐범 (Generator) - providing fake
경찰 (Discriminator) - finding the fake data

# **Adversarial Nets**

<img width="767" alt="Untitled 4" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/b90753a3-fd60-4906-81d8-98b0a987a06a">
<img width="968" alt="Untitled 5" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/148510fe-f33c-4e7b-8f4e-58ba126b9449">
<img width="970" alt="Untitled 6" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/2ebbd768-f5ad-48d5-a43f-d6b42246e438">


 ‘경쟁’이다. Generator aims to minimize loss(D(fake)~1), when the discriminator aims to maximize loss(D(real)~1).

경쟁을 계속하게 되면, 결국 결국에는 pg=pdata가 되어 discriminator가 둘을 전혀 구별하지 못하는 즉, D(x)=1/2인 상태가 된다.

<img width="779" alt="Untitled 7" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/425faf81-306f-47aa-9dd5-b9ccd382f2fc">


# Algorithm

<img width="807" alt="Untitled 8" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/02aef6ba-e60d-4661-ad2e-3acf48b1c2db">


# Global Optimality

<img width="782" alt="Untitled 9" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/d8f2a84e-5ed7-4e62-9449-b1452fbfd1e0">


<img width="503" alt="Untitled 10" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/49eca05a-e6bb-4683-b4b3-e1dd18493206">


# Mathematical Proof (Theorem)

<img width="948" alt="Untitled 11" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/ff743385-fc74-4ddb-a0c8-309422964e01">


<img width="530" alt="Untitled 12" src="https://github.com/jun-brro/deep-learning-paper-review/assets/115399447/dc356736-b687-4a53-93bd-13ba73246f9b">


[Understanding KL Divergence](https://towardsdatascience.com/understanding-kl-divergence-f3ddc8dff254)

[KL divergence와 JSD의 개념 (feat. cross entropy)](https://ddongwon.tistory.com/118)

[Jensen–Shannon divergence](https://en.wikipedia.org/wiki/Jensen–Shannon_divergence)

[Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)

# Questions

1. Generator vs Discriminator 경쟁을 시켜서, 서로의 능력을 모두 향상시킬 수 있다는 개념. 실제로 데이터를 생성하는 쪽은 Generator인데, 해당 Generator의 성능 향상을 aim하기 위해 Discriminator를 도입한 것으로 이해함. 실제 모델의 성능 평가지표는 어떻게 구성되는가?
2. Pre-trained 된 discriminator를 새로운 generator를 train 하는 데에 사용할 수 있는가?
3. Generator의 목적을 정확히 정의할 수 있는가? (생성의 목적?)
4. 위에서 언급한 True data를 train하는 데에 사용한 데이터와 같다고 볼 수 있는지?
