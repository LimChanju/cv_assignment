# 이미지 형성 과제 (2주차 Assignment 1~3)

## 과제 1: 카메라 캘리브레이션 📷
* **설명:** 체크보드 이미지들을 이용하여 카메라 파라미터(카메라 행렬 K, 왜곡 계수)를 계산하고, 이를 통해 왜곡된 이미지를 보정합니다.
* **배경 지식:**
  - 카메라 캘리브레이션: 실제 3D 공간 좌표와 이미지상의 2D 좌표의 대응 관계를 찾아 카메라 내부 파라미터를 구하는 과정
  - 카메라 행렬 K: 카메라의 초점거리(focal length)와 주점(principal point)을 나타내는 3×3 행렬
  - 왜곡 계수: 렌즈의 광학적 왜곡을 보정하기 위한 계수들

* **주요 구현 포인트:**
1. **체크보드 코너 검출:** `cv2.findChessboardCorners()`를 사용하여 각 이미지에서 체크보드의 내부 코너(9×6)를 자동 검출
2. **서브 픽셀 정밀화:** `cv2.cornerSubPix()`로 검출된 코너 위치를 서브 픽셀 단위로 정밀하게 조정. 검색 윈도우(11, 11)를 통해 1픽셀 미만의 정밀한 위치 보정으로 캘리브레이션 정확도 향상
3. **정밀화 검증:** 원본 코너 좌표와 정밀화된 좌표의 차이(유클리디안 거리)를 계산하여 보정량 측정 및 정밀화 효과 확인
4. **3D-2D 좌표 매칭:** 실제 체크보드의 3D 좌표(`objpoints`)와 이미지상의 정밀화된 2D 좌표(`imgpoints`)를 대응시켜 저장
5. **카메라 파라미터 계산:** `cv2.calibrateCamera()`로 수집된 데이터로부터 카메라 행렬과 왜곡 계수 계산
6. **왜곡 보정:** `cv2.undistort()`를 사용하여 계산된 파라미터로 원본 이미지의 왜곡을 제거

* **핵심 코드:**
```python
# 1. 체크보드 코너 검출 및 정밀화
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
if ret == True:
    objpoints.append(objp) # 실제 3D 좌표 저장
    corners_original = corners.copy() # 정밀화 전 원본 저장
    
    # 서브 픽셀 단위로 코너 위치 정밀화 (11x11 윈도우, 최대 30회 반복)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2) # 정밀화된 2D 좌표 저장
    
    # 정밀화 효과 검증: 원본과 정밀화된 좌표의 유클리디안 거리 계산
    mean_shift = np.mean(np.linalg.norm(corners_original - corners2, axis=2))
    print(f"평균 보정량: {mean_shift:.4f} 픽셀")

# 2. 카메라 캘리브레이션 - 카메라 행렬 K와 왜곡 계수 dist, 카메라 외부 파라미터 계산 및 로드
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# 3. 왜곡 보정
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h)) # 1은 원본의 모든 픽셀을 보존
dst = cv2.undistort(img, K, dist, None, newcameramtx) # 왜곡 보정된 이미지
```

<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python
import os
import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 코너 정밀화 종료 조건: 최대 30회 반복 또는 0.001 픽셀 이하의 이동

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32) # (9*6, 3) 크기의 배열 생성, 각 행은 (x, y, z) 좌표를 나타냄
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) # mgrid를 이용해 (0,0), (1,0), ..., (8,5) 형태의 좌표 생성 후 2D 배열로 변환하여 objp의 x, y 좌표에 할당
objp *= square_size # 실제 크기 반영 (mm 단위로 변환)

# 저장할 좌표
objpoints = [] # 실제 3D 좌표 (체크보드의 각 코너에 대한 실제 위치)
imgpoints = [] # 이미지 상의 2D 좌표 (체크보드의 각 코너가 이미지에서 어디에 위치하는지)

images = glob.glob("images/calibration_images/left*.jpg")

img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
is_first_image = True # 첫 번째 이미지의 cornerSubPix 비교 결과를 출력하기 위한 플래그

for fname in images: # 13장의 이미지를 모두 반복해서 카메라 파라미터 K와 dist의 계산에 필요한 2D-3D 매칭 데이터를 수집
    # imread는 기본적으로 컬러 이미지(3채널)를 로드함. 이미 images가 흑백사진이므로 IMREAD_GRAYSCALE 플래그를 사용하여 1채널로 로드하여 메모리 사용량을 줄임. 또한, cornerSubPix 함수는 단일 채널 이미지를 필요로 하므로 그레이스케일로 로드하는 것이 적절함.
    gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"에러: '{fname}' 이미지를 불러올 수 없습니다. 파일 경로를 다시 확인하세요.") # 이미지가 제대로 로드되지 않았을 경우 에러 메시지 출력
        continue

    if img_size is None:
        img_size = gray.shape[::-1] # 이미지 크기 저장 (width, height)
    
    # 코너 검출 수행
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None) # 체크보드 코너 검출, ret은 성공 여부, corners는 검출된 코너 좌표
    
    # 코너 검출에 성공한 경우 배열에 추가 (실패한 이미지는 예외)
    if ret == True:
        objpoints.append(objp) # 코너 검출 성공 -> 실제 좌표 추가
        
        # cornerSubPix 적용 전 코너 좌표 복사 (비교를 위해)
        corners_original = corners.copy()
        
        # 서브 픽셀 단위로 코너 위치 정밀화
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) # 코너 검출 결과를 서브 픽셀 단위로 정밀화하는 함수, (11, 11)은 검색 윈도우 크기, (-1, -1)은 검색 영역의 최소 및 최대 크기, criteria는 정밀화 종료 조건
        imgpoints.append(corners2) # 정밀화된 코너 좌표 추가
        
        # --- cornerSubPix 적용 전후 비교 코드 ---
        if is_first_image:
            print(f"--- [{fname}] 서브 픽셀 정밀화 비교 ---")
            print("인덱스 | 적용 전 (x, y)        | 적용 후 (x, y)        | 이동 거리(픽셀)")
            
            # 상위 5개의 코너 좌표만 출력해서 비교
            for j in range(5):
                pt_before = corners_original[j][0]
                pt_after = corners2[j][0]
                # 두 좌표 간의 유클리디안 거리(픽셀 이동량) 계산
                distance = np.linalg.norm(pt_before - pt_after)
                
                print(f"  {j:2d}   | ({pt_before[0]:.3f}, {pt_before[1]:.3f}) | ({pt_after[0]:.3f}, {pt_after[1]:.3f}) | {distance:.4f} px")
            
            # 해당 이미지 내 전체 코너의 평균 이동 거리 계산
            mean_shift = np.mean(np.linalg.norm(corners_original - corners2, axis=2))
            print(f"-> 전체 {len(corners)}개 코너의 평균 위치 보정량: {mean_shift:.4f} 픽셀\n")
            
            is_first_image = False # 첫 번째 이미지 분석 후 플래그 변경
        # ----------------------------------------------
    
# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 1번에서 구한 2D-3D 매칭 데이터를 이용해 카메라 파라미터 K와 왜곡 계수 dist를 계산
if len(objpoints) > 0: # 만약 코너가 검출된 이미지가 하나라도 있다면 캘리브레이션 수행
     # 카메라 캘리브레이션 수행, K는 카메라 행렬, dist는 왜곡 계수, rvecs와 tvecs는 각 이미지에 대한 회전 및 이동 벡터(카메라 외부 파라미터)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
else:
    print("캘리브레이션에 사용할 수 있는 유효한 이미지가 없습니다.")
    exit()

print("Camera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
# 전체 이미지를 왜곡 보정 및 첫 번째 이미지만 시각화
for i, fname in enumerate(images):
    img = cv2.imread(fname) # 이미지 로드
    if img is None:
        continue
    
    h, w = img.shape[:2] # 이미지 높이와 너비 추출
    
    # 최적의 카메라 행렬 계산
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h)) # alpha=1: 원본 이미지의 모든 픽셀 보존

    # undistort()를 사용하여 왜곡 보정
    dst = cv2.undistort(img, K, dist, None, newcameramtx)

    # 원본 이미지와 보정된 이미지를 가로로 연결 (hstack)
    combined_img = np.hstack((img, dst))
    
    # combined_img를 디렉토리에 저장하기 위한 디렉토리 및 파일 이름 생성
    save_dir = "images/calibration_results"
    os.makedirs(save_dir, exist_ok=True) # 디렉토리가 존재하지 않으면 생성
    
    base_name = os.path.basename(fname) # 원본 파일 이름 추출
    save_name = base_name.replace("left", "calibrated") # 저장할 파일 이름 변경 (예: left01.jpg -> calibrated01.jpg)
    save_path = os.path.join("images/calibration_results", save_name)
    
    # 이미지 저장
    success = cv2.imwrite(save_path, combined_img)
    if not success:
        print(f"이미지 저장 실패: {save_path}")
    else:
        print(f"이미지 저장 성공: {save_path}")

    # 첫 번째 이미지만 시각화 수행
    if i == 0:
        cv2.namedWindow('First Image - Original (Left) vs Undistorted (Right)', cv2.WINDOW_NORMAL) # 창 크기 조절 가능하도록 설정 
        cv2.imshow('First Image - Original (Left) vs Undistorted (Right)', combined_img) # 원본과 보정된 이미지 함께 표시
        cv2.waitKey(0) # 키 입력 대기
        cv2.destroyAllWindows() # 모든 창 닫기

```

</details>

* **주요 결과물:**
  - **카메라 행렬 K와 왜곡 계수**
  <img width="451" height="123" alt="스크린샷 2026-03-12 165312" src="https://github.com/user-attachments/assets/87935b5f-0d35-494b-a222-cc9b9b3ee49e" />

  - **cornerSubPix 정밀화 효과 시각화:**
  ![calibrated01](https://github.com/user-attachments/assets/c808fa64-d6b4-479f-98f1-cbf9e0abab8b)


#### 📊 각 코너별 이동 거리 (상위 5개)
```
코너 0: ████████░░░░ 0.2006 px
코너 1: █████████░░░ 0.2291 px  ← 최대 보정
코너 2: ████░░░░░░░░ 0.0860 px
코너 3: ███░░░░░░░░░ 0.0699 px  ← 최소 보정
코너 4: █████░░░░░░░ 0.1694 px
────────────────────────────
평균:   █████░░░░░░░ 0.1912 px  (54개 전체 코너)
```




---

## 과제 2: 이미지 변환 (회전, 스케일, 평행이동) 🔄
* **설명:** 유사 변환(Similarity Transformation) 개념을 기반으로, 한 장의 이미지에 rotation, Scaling, translation을 동시에 적용합니다.
* **배경 지식:**
  - 유사 변환과 어파인 변환의 관계: 본 과제의 조건(회전, 축소, 이동)은 비틀림이나 찌그러짐(Shear)이 없는 유사 변환(DoF 4)에 해당. 이는 수학적으로 어파인 변환(DoF 6)의 부분집합이므로, OpenCV에서는 `cv2.warpAffine()`이라는 통합된 2×3 행렬 연산 함수를 통해 처리
  - `cv2.getRotationMatrix2D()`: 2×3 변환 행렬 M을 반환하며, 수식은 다음과 같음:

    ```math
    M = \begin{bmatrix} \alpha \cos\theta & -\alpha \sin\theta & t_x \\ \alpha \sin\theta & \alpha \cos\theta & t_y \end{bmatrix}
    
    (단, \alpha: 스케일, \theta: 회전 각도, t_x, t_y: 평행이동)
    ```
    
  - 변환 행렬의 마지막 열은 평행이동(translation) 항을 나타냄. M[0, 2]는 x축 평행이동, M[1, 2]는 y축 평행이동

* **주요 구현 포인트:**
1. **회전 및 스케일 행렬 생성:** `cv2.getRotationMatrix2D()`로 중심점 기준 회전 및 스케일 변환 행렬 생성
2. **평행이동 추가:** 생성된 변환 행렬의 마지막 열에 평행이동 값(tx=80, ty=-40)을 직접 더해서 M의 [0, 2]와 M[1, 2] 요소 업데이트하여 삼중 변환 적용
3. **어파인 변환 적용:** `cv2.warpAffine()`으로 계산된 변환 행렬을 이미지에 적용. 픽셀 $(x, y) \rightarrow (x', y') = M \cdot (x, y, 1)^T$
4. **출력 이미지 크기 유지:** 변환 후 출력 이미지의 크기는 원본과 동일하게 유지

* **핵심 코드:**
```python
# 1. 회전 중심점 계산
center = (w / 2.0, h / 2.0)

# 2. 회전 및 스케올 변환 행렬 생성 (중심 기준 +30도 회전, 0.8배 스케일)
angle = 30
scale = 0.8
M = cv2.getRotationMatrix2D(center, angle, scale)
# M = [ α·cos(θ)   -α·sin(θ)   tx ]     여기서 α=0.8, θ=30°, tx=ty=0
#     [ α·sin(θ)    α·cos(θ)   ty ]
# ≈ [ 0.693   -0.4     0 ]
#   [ 0.4     0.693   0 ]

# 3. 평행이동 추가 (x축 +80, y축 -40)
tx = 80
ty = -40
M[0, 2] += tx  # x축 평행이동 (+80), M은 2X3 행렬이므로 M[0, 2]는 x축 평행이동 요소
M[1, 2] += ty  # y축 평행이동 (-40), M은 2X3 행렬이므로 M[1, 2]는 y축 평행이동 요소

# 4. 어파인 변환 적용
result = cv2.warpAffine(img, M, (w, h))
```

<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python
import cv2
import numpy as np

# 1. 이미지 로드
img_path = 'images/rose.png'
img = cv2.imread(img_path)

if img is None:
    print(f"에러: '{img_path}' 이미지를 불러올 수 없습니다. 경로를 확인하세요.")
    exit()

h, w = img.shape[:2]

# 2. 회전 중심점 계산
center = (w / 2.0, h / 2.0)

# 3. 회전 및 크기 조절 적용
angle = 30
scale = 0.8
M = cv2.getRotationMatrix2D(center, angle, scale)
# M은 2x3 어파인 변환 행렬:
# M = [ α·cos(θ)   -α·sin(θ)   tx ]     여기서 α=0.8, θ=30°
#     [ α·sin(θ)    α·cos(θ)   ty ]
# ≈ [ 0.693   -0.4     0 ]
#   [ 0.4     0.693   0 ]

# 4. 평행이동 적용
# 변환 행렬 M의 마지막 열에 x, y 평행이동 값을 더해줌
tx = 80
ty = -40
M[0, 2] += tx  # x축 평행이동 (+80), M은 2X3 행렬이므로 M[0, 2]는 x축 평행이동 요소: 0 + 80 = 80
M[1, 2] += ty  # y축 평행이동 (-40), M은 2X3 행렬이므로 M[1, 2]는 y축 평행이동 요소: 0 - 40 = -40
# 최종 M = [ 0.693   -0.4     80  ]
#         [ 0.4     0.693  -40  ]

# 5. 어파인 변환 적용
result = cv2.warpAffine(img, M, (w, h))

# 6. 원본과 결과 이미지를 가로로 연결
transformed_img = np.hstack((img, result))

# 7. 결과 이미지 저장
save_path = 'images/rose_transformed.png'
success = cv2.imwrite(save_path, transformed_img)
if success:
    print(f"이미지 저장 성공: {save_path}")
else:
    print(f"이미지 저장 실패: {save_path}")

# 8. 시각화
cv2.namedWindow('L02 - Original vs Transformed', cv2.WINDOW_NORMAL)
cv2.imshow('L02 - Original vs Transformed', transformed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

</details>

* **주요 결과물:**
<img width="2376" height="792" alt="rose_transformed" src="https://github.com/user-attachments/assets/12b68ca7-03dd-4c75-8087-aca609ae92c6" />


---

## 과제 3: 스테레오 비전을 이용한 깊이 맵 계산 🎬
* **설명:** 좌/우 카메라로 촬영한 스테레오 이미지 쌍으로부터 disparity(시차)를 계산하고, 이를 통해 3D 깊이 정보를 추출합니다. ROI별 평균 깊이를 분석하여 가장 가까운 객체와 가장 먼 객체를 판별합니다.
* **배경 지식:**
  - 스테레오 비전: 두 개의 카메라로 같은 장면을 촬영하여 3D 정보를 복원하는 기법
  - Disparity(시차): 스테레오 이미지 쌍에서 같은 물체가 좌/우 이미지에 나타나는 위치의 차이
  - Depth 계산: Z = fB / d (Z: 깊이, f: 초점거리, B: 기선거리, d: disparity)
  - StereoBM: OpenCV의 블록 매칭 스테레오 알고리즘 (간단하고 빠름)
  - Colormap 시각화: Jet colormap을 사용하여 깊이를 색상으로 표현 (빨강: 가까움, 파랑: 멈)

* **주요 구현 포인트:**
1. **StereoBM 객체 생성:** `cv2.StereoBM_create()`로 블록 매칭 알고리즘 초기화 (disparities=80, blockSize=15)
2. **Disparity 계산:** `stereo.compute()`로 좌/우 이미지로부터 disparity 맵 계산하고 16으로 정규화
3. **깊이 맵 계산:** 스테레오 공식 Z = fB / d를 적용하여 실제 3D 깊이 값 계산
4. **유효성 마스크 생성:** disparity > 0인 픽셀만 유효한 측정값으로 처리
5. **ROI별 평균값 계산:** 지정된 ROI(Painting, Frog, Teddy) 내에서 평균 disparity 및 깊이 계산
6. **색상 맵 적용:** `cv2.applyColorMap()`으로 disparity와 깊이를 시각화 가능한 색상으로 변환

* **핵심 코드:**
```python
# 1. StereoBM 알고리즘 설정 및 Disparity 계산
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
disparity = stereo.compute(left_gray, right_gray).astype(np.float32)
disparity = disparity / 16.0  # 정규화

# 2. 깊이 맵 계산 (Z = fB / d)
f = 700.0  # 초점거리
B = 0.12   # 기선거리 (카메라 간 거리, m)
valid_mask = disparity > 0
safe_disparity = np.where(disparity > 0, disparity, 1.0) # 0으로 나누는 것을 방지하기 위해 0인 값은 임시로 1로 설정 (valid_mask로 나중에 필터링됨)
depth_map = (f * B) / safe_disparity
depth_map[~valid_mask] = 0

# 3. ROI별 평균 깊이 계산
for name, (x, y, w, h) in rois.items():
    roi_depth = depth_map[y:y+h, x:x+w]
    roi_valid = roi_depth > 0
    if np.any(roi_valid):
        avg_depth = np.mean(roi_depth[roi_depth > 0])
    else:
        avg_depth = 0

# 4. 색상 맵 시각화 (Jet colormap: 빨강=가까움, 파랑=멈)
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
```

<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python
import cv2
import numpy as np
from pathlib import Path

# 출력 폴더 생성
output_dir = Path("./03_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 좌/우 이미지 불러오기
left_color = cv2.imread("images/left.png")
right_color = cv2.imread("images/right.png")

if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")


# 카메라 파라미터
f = 700.0
B = 0.12

# ROI 설정
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# -----------------------------
# StereoBM 객체 생성
# numDisparities는 16의 배수여야 하며, blockSize는 홀수여야 함
stereo = cv2.StereoBM_create(numDisparities = 16*5, blockSize = 15)
disparity = stereo.compute(left_gray, right_gray).astype(np.float32)

disparity = disparity / 16.0 # OpenCV의 StereoBM은 내부적으로 disparity 값을 16배로 저장하므로, 실제 disparity 값을 얻기 위해 16으로 나눠줌

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
# disparity > 0 인 픽셀에 대해서만 valid_mask 생성
valid_mask = disparity > 0

# 0으로 나누는 것을 방지하기 위해 0인 값은 임시로 1로 설정 (valid_mask로 나중에 필터링됨)
safe_disparity = np.where(disparity > 0, disparity, 1.0)
depth_map = (f * B) / safe_disparity

# 유효하지 않은 곳은 depth 값을 0으로 처리
depth_map[~valid_mask] = 0

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}

for name, (x, y, w, h) in rois.items():
    # ROI 영역 추출
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    
    # 해당 ROI 내에서 유효한(disparity > 0) 픽셀의 마스크
    roi_valid = roi_disp > 0
    
    # 유효한 픽셀이 있는 경우에만 평균 disparity / depth 계산
    if np.any(roi_valid):
        avg_disp = np.mean(roi_disp[roi_disp > 0])
        avg_depth = np.mean(roi_depth[roi_depth > 0])
    else:
        avg_disp = 0
        avg_depth = 0

    results[name] = {"disparity": avg_disp, "depth": avg_depth}

# -----------------------------
# 4. 결과 출력
# -----------------------------
print("--- ROI별 평균 Disparity 및 Depth ---")
for name, data in results.items():
    print(f"{name}: Disparity = {data['disparity']:.2f}, Depth = {data['depth']:.2f}")

# 가장 가까운 영역과 가장 먼 영역 찾기
# 유효한 disparity 값이 이쓴 영역만 대상으로 함
valid_results = {k: v for k, v in results.items() if v["depth"] > 0} # depth가 0보다 큰 영역만 필터링

if valid_results:
    closest = min(valid_results.items(), key=lambda x: x[1]["depth"])
    farthest = max(valid_results.items(), key=lambda x: x[1]["depth"])
    
    print("\n--- 가장 가까운 영역과 가장 먼 영역 ---")
    print(f"\n가장 가까운 영역: {closest[0]} (Depth = {closest[1]['depth']:.2f})")
    print(f"가장 먼 영역: {farthest[0]} (Depth = {farthest[1]['depth']:.2f})")

# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan

if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:
    d_max = d_min + 1e-6

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]

    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # depth는 클수록 멀기 때문에 반전
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy()
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 8. 저장
# -----------------------------
cv2.imwrite(str(output_dir / "left_roi.png"), left_vis)
cv2.imwrite(str(output_dir / "right_roi.png"), right_vis)
cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_map.png"), depth_color)

print("\n이미지 저장 완료 (outputs 폴더)")

# -----------------------------
# 9. 시각화
# -----------------------------
combined_img = np.hstack((left_vis, disparity_color))

cv2.namedWindow('Original vs Disparity Map', cv2.WINDOW_NORMAL)
cv2.imshow('Original vs Disparity Map', combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

</details>

* **주요 결과물:**
<img width="293" height="155" alt="image" src="https://github.com/user-attachments/assets/e8e311e4-4a99-4315-adae-354cef75adcc" />
<img width="1919" height="1029" alt="image" src="https://github.com/user-attachments/assets/b77ca0e7-35b1-4ef9-890e-821c3f641e51" />

