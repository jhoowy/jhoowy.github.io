---
title: "GA-DAN: Geometry-Aware Domain Adaptation Network for Scene Text Detection and Recognition"
published: true
use_math: true
categories: 
  - review
---

GA-DAN: Geometry-Aware Domain Adaptation Network for Scene Text Detection and Recognition (ICCV 2019)

외형(Appearance)적인 feature 뿐만 아니라 공간(Geometry)적인 feature를 target domain에 adaptation 해보자는 것이 목표  
+ appearance space: motion blurs 등
+ geometry space: perspective distortion 등

공간적인 feature shift를 담당하는 spatial modules($S_X$, $S_Y$), 그리고 빈 부분을 채워넣는 generator($G_X$, $G_Y$)로 이루어져 있음

Discriminator는 이미지의 진위여부를 판단하는 것에 더해, spatial transformation이 잘 됐는지도 판단함

논문에서는 appearance와 geometry space를 동시에 domain adaptation하는 최초의 네트워크라 하고 있다

## GA-DAN Architecture

<figure class="align-center">
  <img src="/assets/images/post/200205/architecture.png" alt="">
  <center><figcaption>GA-DAN Architecture</figcaption></center>
</figure> 

Spatial module, generator, discriminator 세가지로 구성되어 있는 사이클 구조

전체적인 CycleGAN 구조와 많이 비슷하다

크게 두 가지의 맵핑을 한다고 볼 수 있다
+ $X → Adapted\ X\ (X → Y)$
+ $Y → Adapted\ Y\ (Y → X)$

위에서 설명했듯, $S_X$는 $X$를 $Y$의 공간적인 스타일과 비슷하게 변환시키고, $G_X$는 변환된 $X$의 빈공간을 $Y$와 비슷하게 채워넣어준다

$D_Y$는 생성된 $Adapted\ X$와 $Y$가 비슷한지 판단해주는 역할을 함

### Spatial Module

Spatial Module은 **Localization Network**($LN$)와, **Transformation Module**($T$)로 이루어져 있음

모듈은 다음과 같이 동작

1. normal distribution에서 Spatial Code를 하나 샘플링함 (i.e Random vector)
2. Spatial Code를 $LN$을 통해 이미지에 맞는 spatial transformation matrix로 regress함
3. $T$와 transformation matrix를 통해 이미지 생성
(※ 한 cycle에서 $Adapted\ X$와 $Adapted\ Y$를 생성할 때 쓴 spatial code는 같아야 함)

이 때, output은 변환된 이미지와 binary map이 있다

binary map은 원본 이미지에서 transform된 pixel들의 위치는 1, 그 외에 검은 배경은 0으로 표시해주는 map

만약 여러 종류의 변환된 이미지를 원하면 spatial code를 여러개로 해서 병렬로 돌려주면 됨

### Generator

Generator는 $G_A$와 $G_B$로 나누어져 있음

$G_A$는 앞선 Spatial Module에서 생성된 binary map의 검은 배경 부분을 타겟 도메인에서 학습한 내용을 바탕으로 채워넣음

$G_B$는 $G_A$에서 완성된 이미지를 다시 한 번 타겟 도메인에 맞도록 수정해줌

만약 Generator를 두개가 아닌 하나만 쓴다면, 생성된 이미지가 blurry 해진다고 함

### Discriminator

$D_X$와 $D_Y$로만 학습을 진행하게 되면 학습이 불안정해지고 잘 수렴하지 않는다고 함

중간에 새로운 Discriminator $D_T$ 만들어줌

아이디어는 $X → Y$의 transformation matrix의 inverse와 $Y → X$의 transform matrix가 같아야 된다는 점에서 착안

$D_T$의 loss는 아래 SCL Loss 참고


## Disentangled Cycle-Consistency Loss


Spatial transformation 특성상 geometry space의 아주 작은 shift도 매우 큰 cycle-consistency loss를 발생시킬 수 있음

이를 방지하기 위해 ACL(Appearance Cycle-Consistency Loss)와 SCL(Spatial Cycle-consistency Loss)로 나눔

추가적으로 RML(Region Missing Loss)도 사용했다고 함

<figure class="align-center">
  <img src="/assets/images/post/200205/spatial_cycle_loss.png" alt="">
  <center><figcaption>Disentangled cycle-consistency loss</figcaption></center>
</figure> 

**Appearance Cycle-consistency Loss**

위 그림처럼 $H_{XY}^{-1}$을 통해 이미지를 다시 복원해줌

이 때 복원된 이미지는 단순히 inverse matrix를 사용했기 때문에 geometry space에서는 동일하겠지만, appearance space에서는 다르게 됨

따라서 복원된 이미지와 원본 이미지사이의 L1 loss를 appearance space 상에서의 차이 ACL로 사용

$$
ACL_X = E_{x\sim X}[\|x_{S_X^{-1}} - x\|]
$$

**Spatial Cycle-consistency Loss**

Geometry space에서의 차이는 이미지 사이의 L1 loss로 구하면 loss 차이가 너무 크게 나오기 때문에 안됨

대신 $H_{XY}^{-1}$와 $H_{YX}$를 직접 비교해준 것을 geometry space 상에서의 차이로 사용

$$
SCL = E_{x\sim X}[\|H_{XY}^{-1} - H_{S_Y}\|]
$$

**Region Missing Loss**

또한 도메인 $X$에서 도메인 $Y$로 adaptation할 때 도메인 $X$에 있던 이미지의 모든 정보가 보존되어야 한다고 함

위의 spatial module의 output인 binary map($m$)에 $H_{XY}^{-1}$을 적용하면 transformation을 하면서 잃어버린 정보들을 알 수 있음

$$
RML = E_{x\sim X}[\|m_{H_{XY}^{-1}} - m\|]
$$

최종 cycle-consistency loss는 다음과 같음

$$
L_{cyc} = \lambda_{acl}ACL + \lambda_{scl}SCL + RML
$$

## Evaluation

Text recognition은 아직 관심 없어서 detection만 볼 것임

### Dataset

ICDAR2013을 source domain, ICDAR2015 + MSRA-TD500을 target domain으로 사용

+ ICDAR 2013: Focused Scene Text dataset. 이름처럼 Text 있는 부분만 집중적으로 찍은 데이터셋
+ ICDAR 2015: Incidental Scene Text dataset. 위와는 다르게 Text가 아닌 부분이 많다. 길거리나 건물내부 등에 있는 text를 찾는게 목표
+ MSRA-TD500: 500장의 이미지. 전체적으로 ICDAR 2013과 2015를 섞은 느낌.

Text detector는 EAST 사용했다고 함

<figure class="align-center">
  <img src="/assets/images/post/200205/detection_eval.png" alt="">
</figure> 

10-AD-IC13은 spatial code를 10개로 한 모델

domain adaptation 안한 것보다는 확실히 좋은 성능을 냈다

SynthText가 ICDAR 2015와 도메인차이가 커서 ICDAR 2015에서는 자기들 모델이 더 좋은 성능을 냈고,  
MSRA-TD500에서는 SynthText와 도메인차이가 작아서 기존의 TextSnake, RRD 모델이 더 좋은 성능을 보여줬다고 한다.

SynthText가 80,000장인 만큼 상대적으로 훨씬 작은 데이터셋인 '10-AD-IC13' 으로 성능향상을 시켜서 detection 발전에 기여를 더 많이 했다고 주장함

[arXiv](https://arxiv.org/abs/1907.09653)