---
title: "YOLACT: Real-time Instance Segmentation"
published: true
use_math: true
categories: 
  - review
---

ICCV 2019 에서 발표된 Real-time Instance Segmentation 에 대한 논문입니다.

Object detection 분야에서 YOLO가 single stage 방식을 사용하여 속도를 끌어올린 것처럼, 이 논문도 기존에 state-of-the-art를 차지하던 two stage 방식이 아닌 single stage 방식으로 **30 fps 이상**을 달성했습니다. 그래서 이름도 **YOLACT(You Only Look At CoefficienTs)** 로 지은 듯 하네요.

[arXiv](https://arxiv.org/abs/1904.02689)  
[GitHub](https://github.com/dbolya/yolact)

## Instance Segmentation

<figure style="max-width: 750px" class="align-center">
  <img src="/assets/images/post/191203/Instance_segmentation.png" alt="">
  <center><figcaption>Computer Vision Tasks (출처: cs231n 강좌)</figcaption></center>
</figure> 

이미지 인식 분야에서 다뤄지는 문제들 중에는 크게 Object detection과 segmentation 문제가 있습니다. 

Object detection 문제는 위의 세 번째 그림처럼, 이미지 안에 어떤 클래스의 물체가 어느 위치에 존재하는지 박스형태로 찾아내는게 목표입니다.

Segmentation 문제도 물체의 클래스와 위치를 찾아내는 게 목표지만, 박스 형태가 아닌 pixel 단위로 물체의 클래스를 구분해내야 합니다.

이 중 semantic segmentation은 모든 픽셀을 클래스에 따라 분류만 하면 되지만, instance segmentation은 같은 클래스여도 서로 다른 물체면 다른 물체로 구분해내야 합니다. 이 때문에 instance segmentation은 object detection에 비해 더 어렵고, 속도도 느렸습니다.

<figure style="max-width: 550px" class="align-center">
  <img src="/assets/images/post/191203/mask_rcnn.png" alt="">
  <center><figcaption>Mask RCNN (출처: cs231n 강좌)</figcaption></center>
</figure> 

기존에 instance segmentation 작업에 주로 사용되던 Mask R-CNN은 먼저 RPN를 통해 물체가 있을만한 위치를 특정한 후(localization step), ROI Pooling 을 거쳐 클래스 라벨, bounding box 보정, 그리고 mask를 생성하는 두 단계로 이루어져 있습니다. 이러한 방식은 정확도는 높았으나 mask를 생성하는 단계가 앞선 localization 단계에 묶여있어 속도를 향상시키기가 매우 어려웠습니다.

## YOLACT

<figure class="align-center">
  <img src="/assets/images/post/191203/YOLACT_architecture.png" alt="">
  <center><figcaption>YOLACT Architecture</figcaption></center>
</figure> 

YOLACT 기본 구조는 object detection에 사용되던 Retinanet을 토대로 만들어졌습니다. YOLACT의 목표는 Mask R-CNN 처럼 기존의 object detection 모델에 mask를 추출하는 분기를 추가하는 것에 더불어, 별도의 feature localization 과정을 없애는 것입니다. 이를 위해 저자는 instance segmentation의 mask를 만들어내는 복잡한 과정을 간단하고, 병렬적으로 진행되는 두 작업으로 나누었습니다.

### Protonet

<figure style="max-width: 450px" class="align-center">
  <img src="/assets/images/post/191203/protonet.png" alt="">
  <center><figcaption>Protonet Architecture</figcaption></center>
</figure> 

첫 번째로 Protonet이라 불리는 부분으로, FCN(Fully Convolution Network)을 통해 이미지 전체에 대한 $k$ 개의 **prototype mask**들을 만들어냅니다. 이 $k$는 클래스의 개수와는 상관이 없는 hyperparameter 입니다. 

논문에 따르면, protonet은 backbone 네트워크에서 더 깊고 해상도가 높은 feature를 사용할수록 더 좋은 성능의 마스크를 만들어 낸다고 합니다. 그래서 인풋으로 FPN의 가장 마지막 output(그림상에서 P<sub>3</sub>)을 사용하고, upsample도 한 번 해줬다고 하네요.

### Mask Coefficients

<figure style="max-width: 250px" class="align-center">
  <img src="/assets/images/post/191203/head_architecture.png" alt="">
  <center><figcaption>Head Architecture</figcaption></center>
</figure> 

다른 하나는 기존 모델에서 클래스 라벨과 박스를 예측하던 object detection 분기에 **mask coefficient**를 예측하는 부분만 추가한 것입니다. 여기서 구한 k 개의 mask coefficient와 prototype mask들을 선형 결합 시키면, 각 detection에 대한 최종 마스크가 나오게 됩니다.

### Rationale

저자가 이러한 방식을 택한 이유는 마스크의 공간적 일관성(spartially coherence) 때문이라고 합니다. (인접한 픽셀들은 같은 객체에 속할 확률이 높은 것처럼.. 그냥 공간적인 특성이 있다는 뜻인 듯)

기존의 single stage 방식은 클래스와 박스 계수를 도출해내기 위해 fc 레이어를 사용했는데, 이 때문에 공간적 일관성이 고려되지 않아 정확성이 떨어졌습니다.

Mask R-CNN 과 같은 two stage 방식은 이러한 문제를 해결하기 위해 localization 단계와 mask를 생성하는 단계를 구분해 conv 레이어로 마스크를 생성할 수 있게 했으나, 위에서 언급했듯, RPN의 localization을 기다려야만 하기 때문에 속도가 느렸습니다.

그러나 이 논문처럼, prototype mask와 계수를 생성하는 부분으로 나누어 병렬적으로 진행시키면 공간적인 일관성을 유지하면서도 빠른속도를 달성할 수 있다고 합니다.

### Prototype Mask

<figure style="max-width: 350px" class="align-center">
  <img src="/assets/images/post/191203/prototype_behavior.png" alt="">
  <center><figcaption>Prototype Behavior</figcaption></center>
</figure> 

위의 그림은 protonet을 지나 생성된 6개의 prototype mask들입니다. 각각의 prototype들이 의미하는 바는 다음과 같습니다.

> 1-3\. 경계선을 기준으로 한 쪽의 물체만 활성화  
> 4\. 객체의 왼쪽 아래부분 활성화  
> 5\. 배경 및 객체 사이의 경계 활성화  
> 6\. 네트워크가 ground 라고 여기는 부분 활성화  

특이한 점은, 박스를 통한 crop 없이도, 마스크가 어느정도 localization을 수행한다는 것입니다. 이는 위 그림의 b-c에서 보이듯이, **같은 물체여도 위치에 따라 다른 마스크를 생성해냄으로 써 다른 위치에 있는 객체를 구분할 수 있게 됩니다.**

그 동안 FCN은 translation invaraint(서로 다른 위치에 있어도 결과가 같음)한 특성을 갖고 있다는게 일반적인 생각이었습니다. 때문에 기존의 FCIS나 Mask R-CNN과 같은 모델들은 translation variant 특성을 얻기 위해 directional map이나 positional-sensitive repooling 등의 방법을 이용했습니다.

그러나 FCN에 패딩이 있으면, 이미지의 edge로 부터 얼마나 먼지 등의 표현을 할 수 있다고 합니다(위 그림의 a가 이를 보여주고 있습니다). 따라서, ResNet과 같은 FCN들은 잠재적으로 transition varinat한 특성을 가지고 있으며, 이 논문은 이 특성을 많이 이용했다고 합니다.

## Fast NMS

또한 이 논문은 새로운 NMS 방법을 제안합니다. NMS(Non Maximum suppression)이란 여러개의 박스가 검출되었을 때, 중복된 검출을 없애기 위한 알고리즘으로, 통상적으로는 다음과 같이 작동합니다.

1. 각 클래스의 검출된 박스마다 confidence 기준으로 내림차순 정렬
2. 각 박스마다 자신보다 낮은 confidence를 가진 박스 중, 일정 이상의 IoU를 가진 박스들을 제거

이러한 방법은 순차적으로 작동되어 낮은 속도에서 작동하는 모델에서는 큰 문제가 안되었으나, 30fps 이상의 속도를 얻는데는 큰 문제가 되었습니다.

이 논문에서 제안하는 방법은 단순히 위와 같은 방법에서, 이미 제거된 박스가 다른 박스들을 제거하는 것을 허용했습니다. 이를 통해, 각 박스들은 제거될지 혹은 유지될지를 병렬적으로 결정될 수 있어 더 빠른 속도를 얻을 수 있다고 합니다. 자세한 방법은 논문을 참조하시면 될 것 같습니다.

<figure style="max-width: 350px" class="align-center">
  <img src="/assets/images/post/191203/fast_nms.png" alt="">
</figure>

위와 같이, Fast NMS를 적용하여 굉장히 적은 성능 손실로도 상당한 양의 속도 상승을 가져올 수 있었습니다.

## Result

<figure class="align-center">
  <img src="/assets/images/post/191203/result_sample.png" alt="">
</figure>

<figure class="align-center">
  <img src="/assets/images/post/191203/result_coco.png" alt="">
  <center><figcaption>MS COCO Results</figcaption></center>
</figure>

COCO 데이터셋에 최신 instance segmentation 모델들을 적용한 결과입니다. YOLACT가 기존의 모델들보다 약 4배 빠른 압도적인 속도를 보여주고 있습니다. 대신 성능은 좀 떨어졌네요.

<figure class="align-center">
  <img src="/assets/images/post/191203/mask_quality.png" alt="">
</figure>

대신 큰 물체에 대한 segmentation 성능은 다른 모델들보다 결과가 좋다고 합니다. 이유는 마스크를 생성할 때 비교적으로 큰 138 x 138 크기의 feature를 사용하고, 원래의 feature에서 repooling 과 같은 변형을 가하지 않기 때문에 그렇다고 하네요.

## Conclusion

그동안 실시간으로 사용이 어려웠던 instance segmentation 작업을 프로토타입과 계수를 사용한 병렬적인 구조로 풀어냄으로써 실시간으로 사용이 가능하다는 것을 보여주었습니다. 또한 이러한 프로토타입과 계수를 이용한 방법은 다른 대부분의 최신 object detector에도 적용이 가능하다고 합니다. 이를 바탕으로 다양한 이미지 인식 모델의 성능향상이 가능할 것으로 기대가 됩니다. 