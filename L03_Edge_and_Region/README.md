# 이미지 형성 과제 (2주차 Assignment 1~3)

## 과제 1: 소벨 에지 검출 🥅
* **설명:** Sobel Filter를 이용하여 이미지의 Edge를 검출하고 시각화한다.
* **배경 지식:**
  - 에지와 미분: 이미지에서 에지는 객체의 경계, 그림자 등 픽셀의 밝기가 급격하게 변하는 지점으로, 수학적으로 이 변화량을 미분을 통해 구하고 1차 미분값이 가장 큰 지점이 에지에 해당한다.
  - Sobel 연산자: 2D 이미지에서 에지를 검출하기 위해 사용되는 이산 미분 연산자로 3X3 크기의 Convolution 커널을 사용하여 수평 방향(x축)과 수직 방향(y축)의 그래디언트 근사값을 계산한다.
  - 에지 강도: x축과 y축의 방향의 미분 결과를 결합하여 최종적인 에지의 강도를 구한다.
    ```math
    G = \sqrt{I_x^2 + I_y^2}
    ```
 - 데이터 타입 변환: 미분 연산 과정 중에 픽셀 값이 음수가 되거나 255를 초과하는 오버플로우가 발생할 수 있다. 이를 방지하기 위해 연산 중에는 64비트 부동소수점 `cv.CV_64F` 자료형을 사용하고 최종 시각화 때 `int8` 자료형으로 변환한다.

* **주요 구현 포인트:**
1. **연산 효율성 확보:** 컬러 이미지를 그레이스케일로 변환하여 에지 검출에 불필요한 색상 정보를 제거하고 연산량을 줄임
2. **데이터 손실 방지:** `cv.Sobel()` 함수 적용 시 ddepth 인자를 `cv.CV_64F`로 설정하여 미분 과정의 데이터 잘림을 방지
3. **정확한 강도 계산 및 복원:** `cv.magnitude()`를 통해 기하학적 에지 강도를 도출하고, `cv.convertScaleAbs()`를 사용하여 시각화 가능한 안전한 픽셀 값(`uint8`) 영역으로 복원

* **핵심 코드:**
```python
# 1. 흑백 이미지 변환 (연산량 감소 및 밝기 정보 집중)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. x축, y축 방향 소벨 필터 적용 (데이터 손실 방지를 위해 CV_64F 사용)
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# 3. 그래디언트 벡터의 크기(에지 강도) 계산
magnitude = cv.magnitude(sobel_x, sobel_y)

# 4. 시각화 및 저장을 위해 8비트 이미지로 정규화 및 변환
magnitude_uint8 = cv.convertScaleAbs(magnitude)
```

<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    
    # 1. 이미지 불러오기 (실제 파일 경로로 변경 필요)
    img_path = 'images/edgeDetectionImage.jpg' 
    img = cv.imread(img_path)

    if img is None:
        print(f"에러: '{img_path}' 이미지를 불러올 수 없습니다. 경로를 확인하세요.")
        return

    # 2. 그레이스케일로 변환
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 3. x축과 y축 방향의 에지 검출 (ksize=3 사용)
    # cv.Sobel(src, ddepth, dx, dy, ksize)에서 src는 입력 이미지, ddepth는 출력 이미지의 깊이, dx와 dy는 각각 x축과 y축 방향의 미분 차수, ksize는 커널 크기
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    # 4. 에지 강도 계산
    # cv.magnitude(x, y)는 x와 y의 크기를 계산하여 에지 강도를 구하는 함수. 여기서는 sobel_x와 sobel_y를 사용하여 각 픽셀에서의 에지 강도를 계산합니다.
    magnitude = cv.magnitude(sobel_x, sobel_y)

    # 5. 에지 강도 이미지를 uint8로 변환
    # cv.convertScaleAbs(src)는 입력 이미지의 절대값을 계산하고, 결과를 uint8로 변환하는 함수. 에지 강도는 양수이므로 절대값을 취하는 것은 큰 의미가 없지만, 이 함수를 사용하여 결과를 uint8로 변환할 수 있음.
    magnitude_uint8 = cv.convertScaleAbs(magnitude)

    # 6. Matplotlib를 사용한 시각화 (원본 이미지 RGB 변환 포함)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))

    # 원본 이미지 시각화
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # 에지 강도 이미지 시각화 (cmap='gray' 사용)
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_uint8, cmap='gray')
    plt.title('Edge Magnitude')
    plt.axis('off')

    plt.tight_layout()
    
    # 이미지 저장
    output_dir = 'result_images'
    
    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Matplotlib의 플롯 전체(비교 화면)를 이미지로 저장
    plt_output_path = os.path.join(output_dir, 'edge_comparison_plot.png')
    plt.savefig(plt_output_path)
    
    plt.show()
    
if __name__ == "__main__":
    main()
```

</details>

* **주요 결과물:**
  - **결과 이미지**


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
1. **StereoBM 객체 생성:** `cv2.StereoBM_create()`로 블록 매칭 알고리즘 초기화 (disparities=16*5, blockSize=15)
- disparities: OpenCV 내부 연산 구조 상 반드시 16의 배수로 설정해야 함
- blockSize: 중심을 잡기 위해 반드시 홀수여야 함
2. **Disparity 계산:** `stereo.compute()`로 좌/우 이미지로부터 disparity 맵 계산하고 16으로 정규화
- StereoBM은 내부적으로 결과값에 16을 곱해 16비트 정수형(CV_16S)으로 변환함
- 따라서 Depth를 계산하기 위해 16으로 다시 나누어 스케일을 복원해야 함
4. **깊이 맵 계산:** 스테레오 공식 Z = fB / d를 적용하여 실제 3D 깊이 값 계산
5. **유효성 마스크 생성:** disparity > 0인 픽셀만 유효한 측정값으로 처리
6. **ROI별 평균값 계산:** 지정된 ROI(Painting, Frog, Teddy) 내에서 평균 disparity 및 깊이 계산
7. **색상 맵 적용:** `cv2.applyColorMap()`으로 disparity와 깊이를 시각화 가능한 색상으로 변환

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

