import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # 1. CIFAR-10 데이터셋 로드
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # 2. 데이터 전처리 (0~255 픽셀 값을 0~1 범위로 정규화)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # CIFAR-10 클래스 이름 정의
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 3. CNN 모델 설계 (Conv2D, MaxPooling2D, Flatten, Dense 활용)
    model = tf.keras.models.Sequential([
        # 첫 번째 합성곱 층
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # 두 번째 합성곱 층
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        # 세 번째 합성곱 층
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        
        # 특성 맵을 1차원 벡터로 평탄화
        tf.keras.layers.Flatten(),
        # 완전 연결 층
        tf.keras.layers.Dense(64, activation='relu'),
        # 출력 층 (10개 클래스)
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. 모델 훈련
    print("CNN 모델 훈련을 시작합니다...")
    # 학습 에포크는 10으로 설정했습니다.
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # 5. 모델 성능 평가
    print("\n테스트 데이터로 모델을 평가합니다...")
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print(f"테스트 정확도: {test_acc:.4f}")

    # 6. 결과 폴더 생성 및 테스트 이미지(dog.jpg) 예측
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    img_path = os.path.join(results_dir, 'cifar10_prediction_result.png')
    
    test_image_path = 'dog.jpg'
    plt.figure(figsize=(5, 5))

    if os.path.exists(test_image_path):
        # 로컬에 dog.jpg가 있는 경우
        print(f"\n'{test_image_path}' 파일을 로드하여 예측을 수행합니다.")
        img = tf.keras.utils.load_img(test_image_path, target_size=(32, 32))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, 0) / 255.0 # 정규화 및 배치 차원 추가
        
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        
        print(f"\n===========================================")
        print(f"모델의 최종 예측 결과 클래스: {predicted_class}")
        print(f"===========================================\n")

        
        plt.imshow(img)
        plt.title(f"Prediction: {predicted_class}")
        
    else:
        # 로컬에 dog.jpg가 없는 경우 (대체재로 테스트 데이터셋의 첫 번째 이미지 사용)
        print(f"\n현재 디렉토리에 '{test_image_path}' 파일이 없어 테스트 데이터셋의 샘플로 예측을 수행합니다.")
        sample_img = x_test[0]
        sample_label = y_test[0][0]
        
        predictions = model.predict(np.expand_dims(sample_img, 0))
        predicted_class = class_names[np.argmax(predictions[0])]
        
        print(f"\n===========================================")
        print(f"모델의 최종 예측 결과 클래스: {predicted_class}")
        print(f"===========================================\n")
        
        true_class = class_names[sample_label]
        
        plt.imshow(sample_img)
        plt.title(f"Pred: {predicted_class} (True: {true_class})")

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(img_path)
    print(f"예측 결과 이미지가 저장되었습니다: {img_path}")

if __name__ == "__main__":
    main()