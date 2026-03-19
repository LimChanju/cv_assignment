import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    img_path = 'images/dabo.jpg' 
    img = cv.imread(img_path)

    if img is None:
        print(f"에러: '{img_path}' 이미지를 불러올 수 없습니다. 경로를 확인하세요.")
        return

    # 직선을 그릴 복사본 생성
    line_img = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 1. 다보탑 위주 ROI 좌표 적용
    x1, y1 = 196, 64
    x2, y2 = 636, 484

    # Numpy 슬라이싱 [y축:y축, x축:x축] 을 이용해 다보탑 영역만 크롭
    roi_gray = gray[y1:y2, x1:x2]

    # 원본 이미지에 ROI 영역 시각적 표시 (파란색 사각형)
    cv.rectangle(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 2. ROI 영역 내 노이즈 제거 및 에지 검출
    blurred = cv.GaussianBlur(roi_gray, (7, 7), 0) # 가우시안 블러를 적용해 노이즈 제거
    edges = cv.Canny(blurred, 100, 200) # 노이즈 제거된 이미지를 이용해 캐니 에지 검출 진행

    # 3. 다보탑 구조 검출에 맞춘 허프 변환 파라미터 
    # (배경 노이즈가 제거되었으므로 다보탑의 짧은 선들을 찾도록 완화된 기준 적용)
    rho = 1                     # 거리 해상도 (픽셀 단위)
    theta = np.pi / 180         # 각도 해상도 (라디안 단위, 여기서는 1도)
    threshold = 60              # 직선으로 판단할 최소 교차점(투표) 수
    min_line_length = 30        # 선분의 최소 길이
    max_line_gap = 5            # 동일한 선상에 있는 선분들 사이의 최대 허용 간격

    lines = cv.HoughLinesP(edges, rho, theta, threshold, 
                           minLineLength=min_line_length, 
                           maxLineGap=max_line_gap)

    # 4. 검출된 직선을 원본 이미지의 절대 좌표로 변환하여 그리기
    if lines is not None:
        for line in lines:
            x_line1, y_line1, x_line2, y_line2 = line[0]
            
            # 절대 좌표 = ROI 내부 좌표 + ROI 시작 오프셋
            # 검출된 좌표는 ROI 내부에서의 좌표이기 때문에 ROI 시작 오프셋을 더해 원본 이미지의 절대 좌표로 변환해야 함
            abs_x1 = x_line1 + x1
            abs_y1 = y_line1 + y1
            abs_x2 = x_line2 + x1
            abs_y2 = y_line2 + y1
            
            cv.line(line_img, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
    else:
        print("ROI 영역 내에서 직선이 검출되지 않았습니다.")

    # 5. 시각화 및 결과 저장
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    line_img_rgb = cv.cvtColor(line_img, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(line_img_rgb)
    plt.title('Hough Lines on ROI')
    plt.axis('off')

    plt.tight_layout()
    
    # 저장 디렉토리 처리
    output_dir = 'result_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'roi_hough_lines.png'))
    
    plt.show()

if __name__ == "__main__":
    main()