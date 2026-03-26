import cv2
import matplotlib.pyplot as plt
import os # 요청하신 os 모듈 임포트

def main():
    # 1. 경로 설정 및 결과 저장 디렉토리 생성
    img_path = 'images/mot_color70.jpg'
    
    # 결과물을 저장할 디렉토리 이름
    results_dir = 'result_images'
    # os 모듈을 사용하여 디렉토리 존재 여부를 확인하고 없으면 생성
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"디렉토리 '{results_dir}'가 생성되었습니다.")
    else:
        print(f"디렉토리 '{results_dir}'가 이미 존재합니다.")

    # 저장할 플롯 이미지의 전체 경로 구성 (os.path.join 사용)
    plot_save_path = os.path.join(results_dir, 'sift_result_comparison_plot.png')

    # 2. 이미지 로드
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"오류: '{img_path}'를 불러올 수 없습니다. 경로와 파일명을 객관적으로 확인해 주십시오.")
        return

    # SIFT 특징 추출을 위해 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. SIFT 객체 생성
    # 힌트에 따라 nfeatures=500으로 제한하여 과도한 특징점 추출 방지
    sift = cv2.SIFT_create(nfeatures=500)

    # 4. 특징점 검출 및 디스크립터 계산
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # 5. 특징점 시각화
    # 힌트에 명시된 flags를 사용하여 특징점의 크기(scale)와 방향(orientation) 시각화
    img_with_keypoints = cv2.drawKeypoints(
        img, 
        keypoints, 
        None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # 6. matplotlib을 이용한 결과 출력 (BGR -> RGB 변환)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_with_keypoints_rgb = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)

    # 1x2 형태로 원본과 결과 이미지 나란히 배치
    fig, axes = plt.subplots(1, 2, figsize=(14, 7)) # plot 크기를 조금 더 키움
    
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(img_with_keypoints_rgb)
    axes[1].set_title(f'SIFT Keypoints (n={len(keypoints)})')
    axes[1].axis('off')

    plt.tight_layout()

    # 7. 전체 플롯 저장 (show() 호출 전에 저장해야 함)
    # 'results' 폴더 안에 'sift_result_comparison_plot.png'라는 이름으로 저장됨
    # bbox_inches='tight'는 플롯 주변의 흰색 여백을 제거하여 저장
    plt.savefig(plot_save_path, bbox_inches='tight', pad_inches=0.1)
    print(f"전체 플롯 결과가 객관적으로 '{plot_save_path}'에 저장되었습니다.")

    # 화면에 플롯 출력
    plt.show()

if __name__ == "__main__":
    main()