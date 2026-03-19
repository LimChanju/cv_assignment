import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. dabo 이미지 불러오기 (실제 파일 경로 및 확장자로 변경 필요)
    img_path = 'images/dabo.jpg' 
    img = cv.imread(img_path)

    if img is None:
        print(f"에러: '{img_path}' 이미지를 불러올 수 없습니다. 경로를 확인하세요.")
        return

    # 직선을 그릴 원본 이미지의 복사본 생성
    line_img = img.copy()

    # 2. 캐니 에지 검출을 위한 그레이스케일 변환
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # ----------------------------(추가))---------------------------------
    # 5x5 커널을 사용하여 자잘한 노이즈(나무, 구름 질감)를 블러 처리
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # 3. 캐니 에지 맵 생성 (힌트: threshold1=100, threshold2=200)
    edges = cv.Canny(blurred, 100, 200)

    # 4. 확률적 허프 변환을 이용한 직선 검출
    # 힌트: 파라미터 조정 필요. 아래는 일반적으로 사용되는 초기 임의값입니다.
    # dabo 이미지의 특성에 맞게 조정해야 최적의 성능이 나옵니다.
    rho = 1              # 거리 해상도 (픽셀 단위)
    theta = np.pi / 180  # 각도 해상도 (라디안 단위, 여기서는 1도)
    threshold = 120       # 직선으로 판단할 최소 교차점(투표) 수
    min_line_length = 120 # 선분의 최소 길이
    max_line_gap = 10    # 동일한 선상에 있는 선분들 사이의 최대 허용 간격

    lines = cv.HoughLinesP(edges, rho, theta, threshold, 
                           minLineLength=min_line_length, 
                           maxLineGap=max_line_gap)

    # 5. 검출된 직선을 원본 이미지에 그리기
    # 힌트: 색상은 (0, 0, 255) 즉 BGR 기준 빨간색, 두께는 2
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        print("검출된 직선이 없습니다. 허프 변환 파라미터를 조정해 보세요.")

    # 6. Matplotlib 시각화 준비 (OpenCV의 BGR을 Matplotlib의 RGB로 변환)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    line_img_rgb = cv.cvtColor(line_img, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))

    # 원본 이미지 시각화
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # 직선이 그려진 이미지 시각화
    plt.subplot(1, 2, 2)
    plt.imshow(line_img_rgb)
    plt.title('Hough Line Detection')
    plt.axis('off')

    plt.tight_layout()

    # 결과 이미지 저장
    output_dir = 'result_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt_output_path = os.path.join(output_dir, 'hough_lines_comparison.png')
    plt.savefig(plt_output_path)
    
    # 화면 출력
    plt.show()

if __name__ == "__main__":
    main()