import cv2
import numpy as np

# 1. 이미지 로드
img_path = 'images/rose.png'
img = cv2.imread(img_path)

if img is None:
    print(f"에러: '{img_path}' 이미지를 불러올 수 없습니다. 경로를 확인하세요.")
    exit()

h, w = img.shape[:2] # 이미지 높이와 넓이 추출

# 2. 회전 중심점 계산
center = (w / 2.0, h / 2.0)

# 3. 회전 및 크기 조절 적용
# 중심 기준 +30도 회전 (OpenCV에서 양수는 반시계 방향), 크기 0.8배
angle = 30
scale = 0.8
M = cv2.getRotationMatrix2D(center, angle, scale) # 회전 및 크기 조절을 위한 어파인 변환 행렬 M 계산

# 4. 평행이동 적용
# 변환 행렬 M의 마지막 열에 x, y 평행이동 값을 더해줌
tx = 80
ty = -40
M[0, 2] += tx  # x축 평행이동 (+80), M은 2X3 행렬이므로 M[0, 2]는 x축 평행이동 요소
M[1, 2] += ty  # y축 평행이동 (-40), M은 2X3 행렬이므로 M[1, 2]는 y축 평행이동 요소

# 5. 어파인 변환 적용
# 출력 이미지 크기는 원본과 동일하게 (w, h)로 유지
result = cv2.warpAffine(img, M, (w, h)) # 어파인 변환 적용, 결과 이미지는 result에 저장

# 6. 원본과 결과 이미지를 가로로 연결 (hstack)
transformed_img = np.hstack((img, result))

# 7. 결과 이미지 저장
save_path = 'images/rose_transformed.png'
success = cv2.imwrite(save_path, transformed_img)
if success:
    print(f"이미지 저장 성공: {save_path}")
else:
    print(f"이미지 저장 실패: {save_path}")

# 8. 시각화
cv2.namedWindow('L02 - Original vs Transformed', cv2.WINDOW_NORMAL) # 창 크기 조절 가능하도록 설정
cv2.imshow('L02 - Original vs Transformed', transformed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()