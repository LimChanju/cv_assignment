import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. 이미지 경로 설정 (실제 커피잔 이미지가 있는 경로로 수정 필요)
    img_path = 'images/coffee_cup.jpg' 
    
    # 이미지 읽기 및 예외 처리
    img = cv.imread(img_path)
    if img is None:
        print(f"[{img_path}] 이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
        return

    # matplotlib 출력을 위해 BGR 색상 공간을 RGB로 변환
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # 2. 마스크 및 GMM 모델 초기화 (힌트 참고)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 3. 초기 사각형 영역 설정 (x, y, width, height)
    # ※ 주의: 이 값은 실제 coffee cup 이미지에서 컵이 위치한 영역에 맞게 직접 조정해야 합니다.
    # 아래는 임의로 이미지 내부를 지정한 예시 값입니다.
    rect = (50, 50, img.shape[1]-100, img.shape[0]-100) 

    # 4. GrabCut 알고리즘 수행 (대화식 분할)
    cv.grabCut(img_rgb, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_RECT)

    # 5. 마스크 값을 0 또는 1로 변경 (힌트 참고)
    # cv.GC_BGD(0), cv.GC_PR_BGD(2) -> 0 (배경)
    # cv.GC_FGD(1), cv.GC_PR_FGD(3) -> 1 (전경)
    mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

    # 6. 배경 제거 (원본 이미지에 마스크 곱하기)
    # mask2는 2차원이므로 원본 이미지(3차원)와 곱하기 위해 차원을 하나 추가(np.newaxis)합니다.
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
    # 현재는 원본, 마스크, 결과가 모두 포함된 matplotlib Figure 전체를 저장하도록 작성했습니다.
    output_dir = 'result_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt_output_path = os.path.join(output_dir, 'grabcut_result_plot.png')
    plt.savefig(plt_output_path)
    
    # 화면에 출력
    plt.show()

if __name__ == '__main__':
    main()