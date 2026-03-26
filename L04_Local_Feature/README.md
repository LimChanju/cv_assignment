# 이미지 Local Feature (4주차 Assignment 1~3)

## 과제 1: SIFT 특징점 검출 📍
* **설명:** 주어진 이미지에서 SIFT(Scale-Invariant Feature Transform) 알고리즘을 사용하여 영상의 크기 변화와 회전에 강건한(Robust) 특징점을 검출하고, 이를 시각화한다.
* **배경 지식:**
  - 스케일 공간 극값 검출 (Scale-space Extrema Detection): 이미지 피라미드를 구성하고, 인접한 가우시안 블러 이미지 간의 차이인 DoG(Difference of Gaussian)를 계산하여 다양한 크기에서의 극대/극소점을 특징점 후보로 찾는다.
  ```math
  D(x,y,\sigma) = L(x,y,k\sigma) - L(x,y,\sigma) \\
  \text{(단, } L(x,y,\sigma) \text{는 원본 이미지와 가우시안 커널의 컨볼루션 결과)}
  ```
  - 방향 할당 (Orientation Assignment): 검출된 특징점 주변 이웃 픽셀들의 그래디언트 크기와 방향을 계산하여 히스토그램을 생성하고, 가장 지배적인 방향(Peak)을 해당 특징점의 기준 방향으로 할당함으로써 회전 불변성을 확보한다.
  - 디스크립터 생성 (Descriptor Generation): 기준 방향을 바탕으로 특징점 주변 16x16 영역의 로컬 그래디언트 정보를 요약하여 128차원의 특징 벡터(Feature Vector)를 생성한다.

* **주요 구현 포인트:**
1. **연산 효율성 확보:** 컬러 이미지를 그레이스케일로 변환하여 SIFT의 픽셀 밝기 변화 패턴 분석에 불필요한 색상 정보를 제거하고 연산량을 줄임
2. **특징점 개수 제어:** 복잡한 텍스처를 가진 이미지에서는 특징점이 과도하게 추출되어 후속 매칭 과정의 병목이 될 수 있다. `cv.SIFT_create(nfeatures=N)`을 설정하여 유의미한 상위 $N$개의 특징점만 검출한다.
3. **정보 손실 없는 시각화:** `cv.drawKeypoints()`적용 시 `cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS`플래그를 사용하여 특징점의 단순 위치뿐만 아니라 검출된 스케일(원 크기)과 주 방향(선)을 동시에 렌더링한다.  

* **핵심 코드:**
```python
# 1. 흑백 이미지 변환 (연산량 감소 및 밝기 정보 집중)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. SIFT 객체 생성 및 특징점 검출 (과적합 방지를 위해 500개로 제한)
    sift = cv.SIFT_create(nfeatures=500)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

# 3. 특징점 시각화 (스케일 및 방향 정보 포함)
    img_with_keypoints = cv.drawKeypoints(
        img, 
        keypoints, 
        None, 
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
```

<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python
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
```

</details>

* **결과 이미지**
<img width="1389" height="423" alt="sift_result_comparison_plot" src="https://github.com/user-attachments/assets/3ad17d05-7a68-4e06-a1f1-5e0d5e40ab93" />


---

## 과제 2: SIFT 특징점 매칭 🎏
* **설명:** 두 개의 다른 시점 또는 조건에서 촬영된 이미지에서 SIFT 특징점을 각각 추출하고, 특징 벡터(Descriptor) 간의 거리 계산을 통해 두 이미지 간의 의미 있는 대응점(Correspondences)을 찾아 시각화한다.
* **배경 지식:**
  - 유클리디안 거리 (Euclidean Distance): SIFT 디스크립터는 128차원의 실수형 벡터로 구성된다. 두 특징점 간의 유사도를 측정하기 위해 L2 노름(Norm), 즉 유클리디안 거리를 계산한다. 거리가 짧을수록 유사도가 높다.
  - 최근접 이웃 탐색 (Nearest Neighbor Search): 기준 이미지의 한 특징점에 대해 대상 이미지의 모든 특징점과 거리를 비교하는 Brute-Force(전수 조사) 방식을 사용하거나, 대규모 데이터의 경우 FLANN(Fast Library for Approximate Nearest Neighbors) 알고리즘을 사용하여 탐색 속도를 최적화할 수 있다.
  - Lowe's Ratio Test (거리 비율 테스트): 특징점 매칭 시 발생하는 오매칭(False Positive)을 제거하기 위한 수학적 기법이다. 가장 가까운 특징점과의 거리($d_1$)와 두 번째로 가까운 특징점과의 거리($d_2$)의 비율이 특정 임계값(일반적으로 0.7~0.8)보다 작은 경우에만 유효한 매칭으로 인정한다. 이는 올바른 매칭은 오매칭에 비해 거리가 압도적으로 가깝다는 통계적 특성에 기반한다.

* **주요 구현 포인트:**
1. **적절한 매칭 알고리즘 선택:** SIFT 디스크립터 특성에 맞추어 거리 측정 방식으로 `cv2.NORM_L2`를 사용하는 `cv2.BFMatcher` 객체를 생성한다.
2. **knnMatch 적용:** 단순 1:1 매칭(`match`)이 아닌, Lowe's Ratio Test를 적용하기 위해 가장 가까운 이웃 2개를 반환하는 `knnMatch(k=2)`를 사용한다.
3. **정확도 필터링:** 임계값(0.75)을 기준으로 두 이웃 간의 거리를 비교하여, 모호한 매칭(비율이 높은 경우)을 객관적으로 배제한다.
4. **결과 시각화:** `cv2.drawMatches()`를 사용하여 양쪽 이미지의 매칭된 점들을 선으로 연결하되, `NOT_DRAW_SINGLE_POINTS` 플래그로 매칭에 실패한 잉여 특징점들은 화면에서 숨겨 가독성을 확보한다.

* **핵심 코드:**
```python
# 1. SIFT 연산을 위한 그레이스케일 변환
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 2. SIFT 특징점 및 디스크립터 추출
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

# 3. BFMatcher 생성 및 knnMatch (k=2) 수행
    # SIFT는 실수형 디스크립터이므로 NORM_L2를 사용합니다.
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

# 4. Lowe's Ratio Test를 통한 오매칭 제거
    good_matches = []
    ratio_thresh = 0.75
    
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

# 5. 매칭 결과 시각화
    img_matches = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        good_matches, 
        None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
```

<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python
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
```

</details>

* **주요 결과물:**
<img width="1260" height="391" alt="sift_matching_result" src="https://github.com/user-attachments/assets/6d0d2151-d398-4dff-8786-b7fab6990745" />


---

## 과제 3: 호모그래피를 이용한 Image Alignment 🧵
* **설명:** SIFT 알고리즘으로 추출한 두 이미지 간의 특징점 매칭 결과를 바탕으로, 두 평면 간의 기하학적 변환 관계인 호모그래피(Homography) 행렬을 계산한다. 추정된 변환 행렬을 적용하여 한 이미지를 다른 이미지의 좌표계로 투영(Warping)시킴으로써 파노라마 형태의 정합 결과를 시각화한다.
* **배경 지식:**
  - 호모그래피 (Homography): 두 평면 간의 투영 기하학적 관계를 나타내는 3x3 변환 행렬 $$H$$이다. 2D 좌표 $$(x, y)$$를 동차 좌표계(Homogeneous Coordinates)인 $$(x, y, 1)$$로 확장하여 표현하면, 원본 이미지의 점과 대상 이미지의 점 사이의 관계를 다음과 같은 선형 방정식으로 나타낼 수 있다. 호모그래피 행렬은 8개의 자유도를 가지므로, 행렬을 계산하기 위해서는 최소 4개의 대응점 쌍이 필요하다.
  - RANSAC (Random Sample Consensus): SIFT와 BFMatcher를 거쳐 선별된 매칭점 중에도 오매칭(Outlier)이 존재할 수 있다. 오매칭점이 하나라도 포함된 상태로 최소자승법(Least Squares)을 적용하면 행렬 추정 결과가 심각하게 왜곡된다. RANSAC은 무작위로 최소 표본(4개의 점)을 뽑아 모델(호모그래피)을 만들고, 나머지 점들을 이 모델에 대입하여 오차 범위를 만족하는 정상점(Inlier)의 개수를 평가한다. 이 과정을 반복하여 가장 많은 Inlier를 갖는 강건한 변환 행렬을 찾는다.
  - 투영 변환 (Perspective Warp): 계산된 호모그래피 행렬을 이미지 전체 픽셀에 적용하여, 대상 이미지의 기하학적 형태를 기준 이미지의 원근감에 맞게 변환한다.

* **주요 구현 포인트:**
1. **유효 매칭점 선별:** 호모그래피 계산의 정확도를 높이기 위해, `knnMatch`와 Lowe's Ratio Test(임계값 0.7)를 선행하여 노이즈 매칭을 1차적으로 걸러낸다.
2. **데이터 타입 변환:** `cv2.findHomography()`의 입력으로 사용하기 위해 추출된 특징점 객체(Keypoint)의 $x, y$ 좌표를 추출하여 `np.float32` 배열로 변환한다.
3. **파노라마 캔버스 확보:** `cv2.warpPerspective()` 적용 시 변환된 이미지가 기존 프레임을 벗어나 잘리는 현상을 방지하기 위해, 결과 출력 캔버스의 크기를 폭은 두 이미지의 합($w_1 + w_2$), 높이는 최대값($\max(h_1, h_2)$)으로 확장한다.
4. **Inlier 시각화:** RANSAC 알고리즘에 의해 정상점(Inlier)으로 판별된 매칭점들만 `matchesMask`를 통해 출력하여 최적화 결과를 객관적으로 확인한다.

* **핵심 코드:**
```python
# 1. SIFT 연산을 위한 그레이스케일 변환
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 2. SIFT 추출 및 knn 매칭
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

# 3. Lowe's Ratio Test 적용 (임계값 0.7)
    good_matches = []
    ratio_thresh = 0.7 
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

# 4. 호모그래피 계산 및 이미지 정합
    MIN_MATCH_COUNT = 4
    if len(good_matches) >= MIN_MATCH_COUNT:
        # 특징점 좌표를 float32 배열로 변환
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # RANSAC을 이용한 호모그래피 행렬(H) 및 Inlier 마스크 추출
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        
        # 캔버스 크기 계산 (너비는 합, 높이는 최대값)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        panorama_w = w1 + w2
        panorama_h = max(h1, h2)
        
        # 투영 변환 (img2를 img1의 평면으로 변환)
        warped_img = cv2.warpPerspective(img2, H, (panorama_w, panorama_h))
        
        # 기준 이미지 덮어쓰기 (단순 결합)
        warped_img[0:h1, 0:w1] = img1
        
        # RANSAC Inlier 매칭점 시각화
        matches_mask = mask.ravel().tolist()
        match_img = cv2.drawMatches(
            img1, kp1, 
            img2, kp2, 
            good_matches, 
            None, 
            matchesMask=matches_mask, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

```

<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python
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
```

</details>

* **주요 결과물:**
<img width="1790" height="334" alt="homography_alignment_result(1 2)" src="https://github.com/user-attachments/assets/9fc04b87-eada-41ae-8d40-6e91a06af7fd" />


- [img 1] + [img 2] : RANSAC Inlier = 260으로 두 이미지 간의 시야 중첩 영역이 가장 넓고 기하학적 변형이 적어, 호모그래피 행렬 추정 시 오차가 가장 적고 안정적인 영상 정합이 가능


<img width="1790" height="334" alt="homography_alignment_result(2 3)" src="https://github.com/user-attachments/assets/efc25b94-1305-472f-a757-2ea9b2919abe" />


- [img 2] + [img 3] : 1번째 이미지와 비슷한 결과

<img width="1790" height="334" alt="homography_alignment_result(1 3)" src="https://github.com/user-attachments/assets/776f47d7-16b0-4205-ad02-cc664d4380c3" />


- [img 1] + [img 3] : 공유하는 텍스처 영역이 적어 직접적인 정합은 신뢰도가 크게 떨어지며, 전체 파노라마 구성 시 img2를 브릿지(Bridge) 프레임으로 경유하여 변환하는 것이 타당

