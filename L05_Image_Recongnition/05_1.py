import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    # 1. MNIST 데이터셋 로드 및 훈련/테스트 세트 분할
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 데이터 정규화 (0~255 픽셀 값을 0~1 사이로 변환)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 2. 간단한 신경망 모델 구축 (Sequential 및 Dense 레이어 활용)
    model = tf.keras.models.Sequential([
        # 28x28 픽셀 크기의 2차원 이미지를 1차원 배열로 평탄화
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # 은닉층: 128개의 노드와 ReLU 활성화 함수 사용
        tf.keras.layers.Dense(128, activation='relu'),
        # 출력층: 10개의 클래스(0~9 숫자) 분류를 위한 Softmax 활성화 함수 사용
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 3. 모델 훈련
    print("모델 훈련을 시작합니다...")
    history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)

    # 4. 정확도 평가
    print("\n테스트 데이터로 모델을 평가합니다...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"테스트 정확도: {test_acc:.4f}")

    # 5. 이미지 저장 (학습 결과 그래프)
    img_path = 'results/training_history.png'
    
    plt.figure(figsize=(10, 4))
    
    # Accuracy 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(img_path)
    print(f"\n학습 결과 이미지가 저장되었습니다: {img_path}")

if __name__ == "__main__":
    main()