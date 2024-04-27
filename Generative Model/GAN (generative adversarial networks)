# Week 1. GAN

Week: Week1
Topic: GAN
작성날짜: 2024년 3월 12일
작성자: Junhyeong Park

![Untitled](Week%201%20GAN%20f8341a07a948443b9105a8721d6bb585/Untitled.png)

[[Paper Review] Generative Adversarial Networks](https://youtu.be/jB1DxJMUlxY?si=4XdV9bbvJ8RnnCnd)

[초짜 대학원생 입장에서 이해하는 Generative Adversarial Nets (1)](https://jaejunyoo.blogspot.com/2017/01/generative-adversarial-nets-1.html)

# Overview of GAN

Competing versus generative model & discriminative model

Generative Model: Aims to fake discriminative model

Discriminative Model: tries to determine whether the generated result is fake or true.

![Untitled](Week%201%20GAN%20f8341a07a948443b9105a8721d6bb585/Untitled%201.png)

![Untitled](Week%201%20GAN%20f8341a07a948443b9105a8721d6bb585/Untitled%202.png)

### ‘위조 지폐범’ vs ‘경찰’

![Untitled](Week%201%20GAN%20f8341a07a948443b9105a8721d6bb585/Untitled%203.png)

위조 지폐범 (Generator) - providing fake

경찰 (Discriminator) - finding the fake data

# **Adversarial Nets**

![Untitled](Week%201%20GAN%20f8341a07a948443b9105a8721d6bb585/Untitled%204.png)

![Untitled](Week%201%20GAN%20f8341a07a948443b9105a8721d6bb585/Untitled%205.png)

![Untitled](Week%201%20GAN%20f8341a07a948443b9105a8721d6bb585/Untitled%206.png)

 ‘경쟁’이다. Generator aims to minimize loss(D(fake)~1), when the discriminator aims to maximize loss(D(real)~1).

경쟁을 계속하게 되면, 결국 결국에는 pg=pdata가 되어 discriminator가 둘을 전혀 구별하지 못하는 즉, D(x)=1/2인 상태가 된다.

![Untitled](Week%201%20GAN%20f8341a07a948443b9105a8721d6bb585/Untitled%207.png)

# Algorithm

![Untitled](Week%201%20GAN%20f8341a07a948443b9105a8721d6bb585/Untitled%208.png)

# Global Optimality

![Untitled](Week%201%20GAN%20f8341a07a948443b9105a8721d6bb585/Untitled%209.png)

![Untitled](Week%201%20GAN%20f8341a07a948443b9105a8721d6bb585/Untitled%2010.png)

# Mathematical Proof (Theorem)

![Untitled](Week%201%20GAN%20f8341a07a948443b9105a8721d6bb585/Untitled%2011.png)

![Untitled](Week%201%20GAN%20f8341a07a948443b9105a8721d6bb585/Untitled%2012.png)

[Understanding KL Divergence](https://towardsdatascience.com/understanding-kl-divergence-f3ddc8dff254)

[KL divergence와 JSD의 개념 (feat. cross entropy)](https://ddongwon.tistory.com/118)

[Jensen–Shannon divergence](https://en.wikipedia.org/wiki/Jensen–Shannon_divergence)

[Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)

# Questions

1. Generator vs Discriminator 경쟁을 시켜서, 서로의 능력을 모두 향상시킬 수 있다는 개념. 실제로 데이터를 생성하는 쪽은 Generator인데, 해당 Generator의 성능 향상을 aim하기 위해 Discriminator를 도입한 것으로 이해함. 실제 모델의 성능 평가지표는 어떻게 구성되는가?
2. Pre-trained 된 discriminator를 새로운 generator를 train 하는 데에 사용할 수 있는가?
3. Generator의 목적을 정확히 정의할 수 있는가? (생성의 목적?)
4. 위에서 언급한 True data를 train하는 데에 사용한 데이터와 같다고 볼 수 있는지?
