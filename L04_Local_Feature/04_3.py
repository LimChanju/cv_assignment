import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. 경로 변수 설정 및 결과 디렉토리 준비
    # 슬라이드 설명에 따라 3개 중 2개를 선택합니다. 샘플로 img1.jpg와 img2.jpg를 사용합니다.
    img_path1 = 'images/img2.jpg'
    img_path2 = 'images/img3.jpg'
    
    results_dir = 'result_images'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    plot_save_path = os.path.join(results_dir, 'homography_alignment_result(2&3).png')

    # 2. 이미지 로드
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    if img1 is None or img2 is None:
        print("오류: 입력 이미지를 불러올 수 없습니다. 파일 경로를 객관적으로 재확인해 주십시오.")
        return

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 3. SIFT 특징점 및 디스크립터 추출
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 4. 특징점 매칭 및 좋은 매칭점 선별 (Lowe's Ratio Test)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    # 힌트에 명시된 대로 거리 비율 임계값을 0.7로 설정
    ratio_thresh = 0.7 
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # 5. 호모그래피 행렬 계산
    # 호모그래피를 계산하려면 최소 4개의 매칭점이 필요합니다.
    MIN_MATCH_COUNT = 4
    if len(good_matches) >= MIN_MATCH_COUNT:
        # 매칭점들의 좌표 추출
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # cv.findHomography를 사용하여 img2의 좌표계를 img1의 좌표계로 변환하는 3x3 행렬 H 계산
        # 힌트에 따라 RANSAC 알고리즘을 적용하여 이상점(Outlier)을 배제합니다.
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        
        # 6. 이미지 정합 (Warp Perspective)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 힌트 적용: 출력 크기를 파노라마 크기 (w1+w2, max(h1,h2))로 설정
        panorama_w = w1 + w2
        panorama_h = max(h1, h2)
        
        # img2를 H 행렬을 이용해 img1의 평면으로 기하학적 변환
        warped_img = cv2.warpPerspective(img2, H, (panorama_w, panorama_h))
        
        # 변환된 도화지(warped_img)의 좌측 영역에 원본 img1을 덮어씌워 파노라마 형태로 결합
        warped_img[0:h1, 0:w1] = img1
        
        # 7. 매칭 결과 시각화 (RANSAC을 통과한 Inlier 매칭점만 표시)
        matches_mask = mask.ravel().tolist()
        match_img = cv2.drawMatches(
            img1, kp1, 
            img2, kp2, 
            good_matches, 
            None, 
            matchesMask=matches_mask, # RANSAC inlier만 표시
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # 8. Matplotlib을 이용한 결과 병렬 출력
        warped_img_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
        match_img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        axes[0].imshow(warped_img_rgb)
        axes[0].set_title('Warped Image (Panorama Alignment)')
        axes[0].axis('off')
        
        inlier_count = np.sum(matches_mask)
        axes[1].imshow(match_img_rgb)
        axes[1].set_title(f'Matching Result (RANSAC Inliers: {inlier_count})')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(plot_save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"이미지 정합 및 매칭 결과가 객관적으로 '{plot_save_path}'에 저장되었습니다.")
        plt.show()

    else:
        print(f"오류: 호모그래피를 계산하기 위한 매칭점 개수가 부족합니다. (현재: {len(good_matches)}, 필요: {MIN_MATCH_COUNT})")

if __name__ == "__main__":
    main()