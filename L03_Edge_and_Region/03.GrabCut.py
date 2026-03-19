import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. 이미지 경로 설정
    img_path = 'images/coffee_cup.jpg' 
    
    # 이미지 읽기 및 예외 처리
    img = cv.imread(img_path)
    if img is None:
        print(f"[{img_path}] 이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
        return

    # matplotlib 출력을 위해 BGR 색상 공간을 RGB로 변환
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # 2. 마스크 및 GMM 모델 초기화 (bgdModel과 fgdModel은 np.zeros((1, 65), np.float64)로 초기화)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 3. 초기 사각형 영역 설정 (x, y, width, height)
    rect = (50, 50, img.shape[1]-100, img.shape[0]-100) # 이미지 가장자리에서 50 픽셀씩 떨어진 사각형 영역 설정 (x=50, y=50, width=이미지 너비-100, height=이미지 높이-100)

    # 4. GrabCut 알고리즘 수행 (대화식 분할)
    # grabCut 함수는 원본 이미지, 마스크, 초기 사각형 영역, 배경 모델, 객체 모델, 반복 횟수, 모드를 인자로 받음
    # 여기서는 초기 사각형 영역을 사용하여 GrabCut을 수행하도록 설정
    cv.grabCut(img_rgb, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_RECT)

    # 5. 마스크 값을 0 또는 1로 변경
    # cv.GC_BGD(0), cv.GC_PR_BGD(2) -> 0 (배경)
    # cv.GC_FGD(1), cv.GC_PR_FGD(3) -> 1 (전경)
    # np.where 함수를 사용하여 mask에서 배경과 객체를 구분
    # 배경으로 분류된 픽셀은 0, 객체로 분류된 픽셀은 1로 설정하여 mask2를 생성
    mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
    

    # 6. 배경 제거 (원본 이미지에 마스크 곱하기)
    # mask2는 2차원이므로 원본 이미지(3차원)와 곱하기 위해 차원을 하나 추가(np.newaxis)
    result_img = img_rgb * mask2[:, :, np.newaxis]

    # 7. matplotlib을 이용한 세 가지 이미지 나란히 시각화
    plt.figure(figsize=(15, 5))
    
    # 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # 마스크 이미지 (시각화를 위해 1인 값을 255로 스케일링)
    plt.subplot(1, 3, 2)
    plt.imshow(mask2 * 255, cmap='gray') 
    plt.title('Mask Image')
    plt.axis('off')

    # 배경 제거 이미지
    plt.subplot(1, 3, 3)
    plt.imshow(result_img)
    plt.title('Background Removed')
    plt.axis('off')

    plt.tight_layout()
    
    # 8. 결과 이미지 저장
    output_dir = 'result_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt_output_path = os.path.join(output_dir, 'grabcut_result_plot.png')
    plt.savefig(plt_output_path)
    
    # 화면에 출력
    plt.show()

if __name__ == '__main__':
    main()