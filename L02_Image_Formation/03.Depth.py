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