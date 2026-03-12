# 이미지 형성 과제 (2주차 Assignment 1~3)

## 과제 1: 카메라 캘리브레이션 📷
* **설명:** 체크보드 이미지들을 이용하여 카메라 파라미터(카메라 행렬 K, 왜곡 계수)를 계산하고, 이를 통해 왜곡된 이미지를 보정합니다.
* **배경 지식:**
  - 카메라 캘리브레이션: 실제 3D 공간 좌표와 이미지상의 2D 좌표의 대응 관계를 찾아 카메라 내부 파라미터를 구하는 과정
  - 카메라 행렬 K: 카메라의 초점거리(focal length)와 주점(principal point)을 나타내는 3×3 행렬
  - 왜곡 계수: 렌즈의 광학적 왜곡을 보정하기 위한 계수들

* **주요 구현 포인트:**
1. **체크보드 코너 검출:** `cv2.findChessboardCorners()`를 사용하여 각 이미지에서 체크보드의 내부 코너(9×6)를 자동 검출
2. **서브 픽셀 정밀화:** `cv2.cornerSubPix()`로 검출된 코너 위치를 서브 픽셀 단위로 정밀하게 조정하여 캘리브레이션 정확도 향상
3. **3D-2D 좌표 매칭:** 실제 체크보드의 3D 좌표(`objpoints`)와 이미지상의 2D 좌표(`imgpoints`)를 대응시켜 저장
4. **카메라 파라미터 계산:** `cv2.calibrateCamera()`로 수집된 데이터로부터 카메라 행렬과 왜곡 계수 계산
5. **왜곡 보정:** `cv2.undistort()`를 사용하여 계산된 파라미터로 원본 이미지의 왜곡을 제거

* **핵심 코드:**
```python
# 1. 체크보드 코너 검출 및 정밀화
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
if ret == True:
    objpoints.append(objp) # 실제 3D 좌표 저장
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2) # 정밀화된 2D 좌표 저장

# 2. 카메라 캘리브레이션 - 카메라 행렬 K와 왜곡 계수 dist 계산
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# 3. 왜곡 보정
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
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
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 좌표
objpoints = [] # 실제 3D 좌표
imgpoints = [] # 이미지 상의 2D 좌표

images = glob.glob("images/calibration_images/left*.jpg")

img_size = None

# 1. 체크보드 코너 검출
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"에러: '{fname}' 이미지를 불러올 수 없습니다.")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = gray.shape[::-1]
    
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

# 2. 카메라 캘리브레이션
if len(objpoints) > 0:
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
else:
    print("캘리브레이션에 사용할 수 있는 유효한 이미지가 없습니다.")
    exit()

print("Camera Matrix K:")
print(K)
print("\nDistortion Coefficients:")
print(dist)

# 3. 왜곡 보정 시각화
for i, fname in enumerate(images):
    img = cv2.imread(fname)
    if img is None:
        continue
    
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, K, dist, None, newcameramtx)
    
    combined_img = np.hstack((img, dst))
    
    save_dir = "images/calibration_results"
    os.makedirs(save_dir, exist_ok=True)
    
    base_name = os.path.basename(fname)
    save_name = base_name.replace("left", "calibrated")
    save_path = os.path.join("images/calibration_results", save_name)
    
    success = cv2.imwrite(save_path, combined_img)
    if not success:
        print(f"이미지 저장 실패: {save_path}")

    if i == 0:
        cv2.namedWindow('First Image - Original (Left) vs Undistorted (Right)', cv2.WINDOW_NORMAL)
        cv2.imshow('First Image - Original (Left) vs Undistorted (Right)', combined_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
```

</details>

* **주요 결과물:** 
  - 카메라 행렬 K와 왜곡 계수가 출력됨
  - 원본 이미지와 왜곡이 보정된 이미지가 나란히 저장됨
  - 특히 이미지 가장자리의 왜곡이 효과적으로 제거됨

---

## 과제 2: 이미지 변환 (회전, 스케일, 평행이동) 🔄
* **설명:** 어파인 변환을 이용하여 이미지에 회전, 스케일 조정, 평행이동을 동시에 적용합니다.
* **배경 지식:**
  - 어파인 변환(Affine transformation): 2×3 행렬을 사용하여 회전, 스케일, 평행이동 등의 기하학적 변환을 수행
  - `cv2.getRotationMatrix2D()`: 회전 중심, 각도, 스케일 인자를 입력받아 어파인 변환 행렬을 생성
  - 변환 행렬의 마지막 열은 평행이동(translation) 항을 나타냄

* **주요 구현 포인트:**
1. **회전 및 스케일 행렬 생성:** `cv2.getRotationMatrix2D()`로 중심점 기준 회전 및 스케일 변환 행렬 생성 (30도 회전, 0.8배 스케일)
2. **평행이동 추가:** 생성된 변환 행렬의 마지막 열에 평행이동 값(tx=80, ty=-40)을 직접 더해서 삼중 변환 적용
3. **어파인 변환 적용:** `cv2.warpAffine()`으로 계산된 변환 행렬을 이미지에 적용
4. **출력 이미지 크기 유지:** 변환 후 출력 이미지의 크기는 원본과 동일하게 유지

* **핵심 코드:**
```python
# 1. 회전 중심점 계산
center = (w / 2.0, h / 2.0)

# 2. 회전 및 스케일 변환 행렬 생성 (중심 기준 +30도 회전, 0.8배 스케일)
angle = 30
scale = 0.8
M = cv2.getRotationMatrix2D(center, angle, scale)

# 3. 평행이동 추가 (x축 +80, y축 -40)
tx = 80
ty = -40
M[0, 2] += tx  # x축 평행이동
M[1, 2] += ty  # y축 평행이동

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

# 4. 평행이동 적용
tx = 80
ty = -40
M[0, 2] += tx  # x축 평행이동 (+80)
M[1, 2] += ty  # y축 평행이동 (-40)

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
  - 원본 이미지: 원래의 장미 사진
  - 변환 이미지: 30도 반시계방향 회전 + 0.8배 스케일 + 우측 하단 평행이동이 적용된 결과
  - 변환 과정에서 이미지 경계 밖의 영역은 검정색(기본값)으로 채워짐

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
safe_disparity = np.where(disparity > 0, disparity, 1.0)
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

# 1. Disparity 계산
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
disparity = stereo.compute(left_gray, right_gray).astype(np.float32)
disparity = disparity / 16.0

# 2. Depth 계산 (Z = fB / d)
valid_mask = disparity > 0
safe_disparity = np.where(disparity > 0, disparity, 1.0)
depth_map = (f * B) / safe_disparity
depth_map[~valid_mask] = 0

# 3. ROI별 평균 disparity / depth 계산
results = {}

for name, (x, y, w, h) in rois.items():
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    
    roi_valid = roi_disp > 0
    
    if np.any(roi_valid):
        avg_disp = np.mean(roi_disp[roi_disp > 0])
        avg_depth = np.mean(roi_depth[roi_depth > 0])
    else:
        avg_disp = 0
        avg_depth = 0

    results[name] = {"disparity": avg_disp, "depth": avg_depth}

# 4. 결과 출력
print("--- ROI별 평균 Disparity 및 Depth ---")
for name, data in results.items():
    print(f"{name}: Disparity = {data['disparity']:.2f}, Depth = {data['depth']:.2f}")

# 가장 가까운 영역과 가장 먼 영역 찾기
valid_results = {k: v for k, v in results.items() if v["depth"] > 0}

if valid_results:
    closest = min(valid_results.items(), key=lambda x: x[1]["depth"])
    farthest = max(valid_results.items(), key=lambda x: x[1]["depth"])
    
    print("\n--- 가장 가까운 영역과 가장 먼 영역 ---")
    print(f"\n가장 가까운 영역: {closest[0]} (Depth = {closest[1]['depth']:.2f})")
    print(f"가장 먼 영역: {farthest[0]} (Depth = {farthest[1]['depth']:.2f})")

# 5. disparity 시각화
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

# 6. depth 시각화
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]

    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)
    
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# 7. Left / Right 이미지에 ROI 표시
left_vis = left_color.copy()
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 8. 저장
cv2.imwrite(str(output_dir / "left_roi.png"), left_vis)
cv2.imwrite(str(output_dir / "right_roi.png"), right_vis)
cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_map.png"), depth_color)

print("\n이미지 저장 완료 (outputs 폴더)")

# 9. 시각화
combined_img = np.hstack((left_vis, disparity_color))

cv2.namedWindow('Original vs Disparity Map', cv2.WINDOW_NORMAL)
cv2.imshow('Original vs Disparity Map', combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

</details>

* **주요 결과물:**
  - **ROI별 깊이 분석:** 3개의 객체(Painting, Frog, Teddy)에 대한 평균 시차 및 깊이 값 계산
  - **Disparity 맵:** 시차를 색상으로 시각화한 이미지 (빨강: 가까움, 파랑: 멀다)
  - **Depth 맵:** 실제 3D 깊이 정보를 색상으로 표현한 이미지
  - **ROI 표시:** 좌/우 원본 이미지에 분석 대상 ROI를 초록색 사각형으로 표시
  - **객체 거리 비교:** 가장 가까운 객체와 가장 먼 객체를 자동으로 판별하여 출력

---

## 학습 요점 정리 📚
* **과제 1 (카메라 캘리브레이션):** 카메라의 내부 파라미터를 정확히 구하여 왜곡된 이미지를 보정하는 기초적이지만 매우 중요한 과정
* **과제 2 (이미지 변환):** 어파인 변환을 활용하여 이미지에 복합적인 기하학적 변환을 효율적으로 적용
* **과제 3 (깊이 맵):** 스테레오 비전의 원리를 이해하고 실제로 2D 이미지로부터 3D 정보를 추출하는 강력한 기법
