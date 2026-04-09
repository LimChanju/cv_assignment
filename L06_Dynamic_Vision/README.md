# Dynamic Vision (6주차 Assignment 1~2)

## 과제 1: SORT 알고리즘을 활용한 다중 객체 추적기 구현 🚘🚔
* **설명:** 텐서플로우(TensorFlow)를 활용하여 MNIST 손글씨 숫자 데이터셋을 0부터 9까지 10개의 클래스로 분류하는 간단한 모델을 구축하고, 모델의 학습 과정(정확도 및 손실)을 시각화한다.
* **배경 지식:**
  - 다층 퍼셉트론 (Multi-Layer Perceptron, MLP): 은닉층(Hidden Layer)을 포함하는 기본적인 순방향 신경망이다.
  - 활성화 함수 (Activation Function) - ReLU: 은닉층에 주로 사용되며, 입력이 0보다 크면 그대로 출력하고 0 이하이면 0을 출력하여 기울기 소실(Vanishing Gradient) 문제를 완화한다.
  ```math
  f(x) = max(0,x)
  ```
  - Softmax: 다중 클래스 분류의 출력층에 사용되며, 각 클래스에 대한 예측값을 0과 1 사이의 확률값으로 변환하고 총합이 1이 되도록 만든다.
  ```math
  $$\sigma(z_i)=\frac{e^{z_i}}{\sum_{j}e^{z_j}}$$
  ```
  - 손실 함수 (Loss Function): 정수형(Integer)으로 인코딩된 정답 레이블과 예측 확률 간의 오차를 계산할 때는 `sparse_categorical_crossentropy`를 사용한다.

* **주요 구현 포인트:**
1. **데이터 정규화 (Normalization):** 0~255 범위의 이미지 픽셀 값을 255.0으로 나누어 0~1 범위로 스케일링함으로써 가중치 학습의 수렴 속도와 안정성을 높인다.
2. **Flatten 레이어:** 28x28 픽셀 크기의 2차원 이미지를 입력받아 1차원 벡터(크기 784)로 평탄화하여 완전 연결 계층(Dense Layer)이 처리할 수 있는 형태로 변환한다.
3. **학습 결과 시각화:** `model.fit()` 함수가 반환하는 `history` 객체를 통해 에포크(Epoch)별 훈련(Train) 및 검증(Validation) 세트의 정확도와 손실 변화 추이를 그래프로 그려 과적합 여부를 객관적으로 파악한다.

* **핵심 코드:**
```python
# 1. 2차원 이미지를 1차원으로 평탄화하고 은닉층과 출력층을 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 2. 다중 클래스 분류에 맞는 손실 함수와 최적화 알고리즘 설정
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 모델 훈련 (학습 과정 데이터를 history 변수에 저장)
history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)```
```

<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python

```

</details>

* **결과 이미지**



---

## 과제 2: Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화 🥸
* **설명:** 텐서플로우(TensorFlow)를 활용하여 CIFAR-10 데이터셋의 3채널 컬러 이미지를 10개의 클래스(비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭)로 분류하는 합성곱 신경망(CNN) 모델을 구축하고, 특정 테스트 이미지(dog.jpg)에 대한 예측 결과를 터미널에 출력 및 시각화한다.
* **배경 지식:**
  - 합성곱 신경망 (Convolutional Neural Network, CNN): 다층 퍼셉트론(MLP)과 달리 1차원 평탄화 과정에서 발생하는 공간적/지역적 정보 손실을 방지하고, 이미지의 2차원 구조를 유지하면서 특징을 추출하는 데 특화된 신경망이다.
  - 합성곱 층 (Convolutional Layer): 입력 이미지에 지정된 크기(예: 3x3)의 필터(커널)를 슬라이딩하며 합성곱 연산을 수행하여 이미지의 특징 맵(Feature Map)을 생성한다.
  - 최대 풀링 층 (Max Pooling Layer): 특징 맵을 하위 샘플링(Sub-sampling)하여 공간적 차원을 축소한다. 이를 통해 모델의 파라미터 수와 연산량을 줄이고, 이미지 내 객체의 미세한 위치 변화에 덜 민감해지는 이동 불변성(Translation Invariance)을 부여한다.

* **주요 구현 포인트:**
1. **데이터 전처리:** 32x32 크기의 RGB 컬러 이미지 픽셀 값(0~255)을 255.0으로 나누어 0~1 범위로 정규화(Normalization)하여 학습 속도와 안정성을 확보한다.
2. **계층적 특징 추출 (Hierarchical Feature Extraction):** 3개의 `Conv2D` 층과 2개의 MaxPooling2D 층을 교차로 배치하여, 얕은 층에서는 에지(Edge)나 질감 같은 저수준 특징을, 깊은 층에서는 객체의 일부와 같은 고수준 특징을 추출하도록 객관적으로 설계되었다.

* **핵심 코드:**
```python
# CNN 모델 설계: 합성곱 층과 풀링 층을 통한 특징 추출 후 완전 연결 층으로 분류
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python

```

</details>

* **주요 결과물:**

