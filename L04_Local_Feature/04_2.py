import cv2
import matplotlib.pyplot as plt
import os

def main():
    # 1. 경로 변수 설정
    # 두 개의 이미지를 입력받아야 하므로 img_path1, img_path2로 설정했습니다.
    img_path1 = 'images/mot_color70.jpg'
    img_path2 = 'images/mot_color83.jpg'
    
    # 결과 저장 디렉토리 설정
    results_dir = 'result_images'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    plot_save_path = os.path.join(results_dir, 'sift_matching_result.png')

    # 2. 이미지 로드
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    if img1 is None or img2 is None:
        print("오류: 입력 이미지를 불러올 수 없습니다. 경로에 파일이 존재하는지 객관적으로 확인해 주십시오.")
        return

    # SIFT 특징 추출을 위한 그레이스케일 변환
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 3. SIFT 객체 생성 및 특징점, 디스크립터 추출
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 디스크립터가 정상적으로 추출되었는지 확인
    if des1 is None or des2 is None:
        print("오류: 특징점 디스크립터를 추출하지 못했습니다.")
        return

    # 4. 특징점 매칭 (BFMatcher 사용)
    # SIFT는 실수형 디스크립터를 가지므로 거리 측정 방식으로 cv2.NORM_L2를 사용합니다.
    # 힌트에 따라 knnMatch를 사용하기 위해 crossCheck=False (기본값)로 둡니다.
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    # k=2로 설정하여 각 특징점당 가장 가까운 2개의 매칭 결과를 반환받습니다.
    matches = bf.knnMatch(des1, des2, k=2)

    # 5. 최근접 이웃 거리 비율 테스트 (Lowe's Ratio Test) 적용
    # 힌트에 명시된 "최근접 이웃 거리 비율을 적용하여 매칭 정확도를 높임"을 구현한 부분입니다.
    good_matches = []
    ratio_thresh = 0.75 # 일반적으로 0.7 ~ 0.8 사이의 값을 사용합니다.
    
    for m, n in matches:
        # 가장 가까운 거리(m.distance)가 두 번째로 가까운 거리(n.distance)에 일정 비율을 곱한 것보다 작으면 유효한 매칭으로 판단합니다.
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # 6. 매칭 결과 시각화
    # cv2.drawMatches를 사용하여 매칭된 특징점을 선으로 연결하여 보여줍니다.
    # NOT_DRAW_SINGLE_POINTS 플래그를 사용하여 매칭되지 않은 잉여 특징점은 그리지 않습니다.
    img_matches = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        good_matches, 
        None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 7. matplotlib을 이용한 결과 출력
    img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 6))
    plt.imshow(img_matches_rgb)
    plt.title(f'SIFT Feature Matching (Good Matches: {len(good_matches)})')
    plt.axis('off')

    # 8. 매칭 결과를 포함한 전체 플롯 저장
    plt.savefig(plot_save_path, bbox_inches='tight', pad_inches=0.1)
    print(f"매칭 시각화 결과가 객관적으로 '{plot_save_path}'에 저장되었습니다.")

    plt.show()

if __name__ == "__main__":
    main()