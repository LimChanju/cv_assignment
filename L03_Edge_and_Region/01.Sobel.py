import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    
    # 1. 이미지 불러오기
    img_path = 'images/edgeDetectionImage.jpg' 
    img = cv.imread(img_path)

    if img is None:
        print(f"에러: '{img_path}' 이미지를 불러올 수 없습니다. 경로를 확인하세요.")
        return

    # 2. 그레이스케일로 변환
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 3. x축과 y축 방향의 에지 검출 (ksize=3 사용)
    # cv.Sobel(src, ddepth, dx, dy, ksize)에서 src는 입력 이미지, ddepth는 출력 이미지의 깊이, dx와 dy는 각각 x축과 y축 방향의 미분 차수, ksize는 커널 크기
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    # 4. 에지 강도 계산
    # cv.magnitude(x, y)는 x와 y의 크기를 계산하여 에지 강도를 구하는 함수. 여기서는 sobel_x와 sobel_y를 사용하여 각 픽셀에서의 에지 강도를 계산합니다.
    magnitude = cv.magnitude(sobel_x, sobel_y)

    # 5. 에지 강도 이미지를 uint8로 변환
    # cv.convertScaleAbs(src)는 입력 이미지의 절대값을 계산하고, 결과를 uint8로 변환하는 함수. 에지 강도는 양수이므로 절대값을 취하는 것은 큰 의미가 없지만, 이 함수를 사용하여 결과를 uint8로 변환할 수 있음.
    magnitude_uint8 = cv.convertScaleAbs(magnitude)

    # 6. Matplotlib를 사용한 시각화 (원본 이미지 RGB 변환 포함)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))

    # 원본 이미지 시각화
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # 에지 강도 이미지 시각화 (cmap='gray' 사용)
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_uint8, cmap='gray')
    plt.title('Edge Magnitude')
    plt.axis('off')

    plt.tight_layout()
    
    # 이미지 저장
    output_dir = 'result_images'
    
    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Matplotlib의 플롯 전체(비교 화면)를 이미지로 저장
    plt_output_path = os.path.join(output_dir, 'edge_comparison_plot.png')
    plt.savefig(plt_output_path)
    
    plt.show()
    
if __name__ == "__main__":
    main()