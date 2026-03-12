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

images = glob.glob("calibration_images/left*.jpg")

img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    img = cv2.imread(fname) # 이미지 로드
    if img is None:
        print(f"에러: '{fname}' 이미지를 불러올 수 없습니다. 파일 경로를 다시 확인하세요.") # 이미지가 제대로 로드되지 않았을 경우 에러 메시지 출력
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 그레이스케일로 변환
    if img_size is None:
        img_size = gray.shape[::-1] # 이미지 크기 저장 (width, height)
    
    # 코너 검출 수행
    ret, corners = cv2.findChessboardCorners(gray CHECKERBOARD, None) # 체크보드 코너 검출, ret은 성공 여부, corners는 검출된 코너 좌표
    
    # 코너 검출에 성공한 경우 배열에 추가 (실패한 이미지는 예외)
    if ret == True:
        objpoints.append(objp) # 코너 검출 성공 -> 실제 좌표 추가
        # 서브 픽셀 단위로 코너 위치 정밀화
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2) # 정밀화된 코너 좌표 추가
    
# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
if len(objpoints) > 0: # 만약 코너가 검출된 이미지가 하나라도 있다면 캘리브레이션 수행
     # 카메라 캘리브레이션 수행, K는 카메라 행렬, dist는 왜곡 계수, rvecs와 tvecs는 각 이미지에 대한 회전 및 이동 벡터
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
# 시각화를 위해 첫 번째 이미지를 사용
test_image_path = image[0] # 첫 번째 이미지 경로
test_img = cv2.imread(test_image_path) # 첫 번째 이미지 로드
h, w = test_img.shape[:2] # 이미지 높이와 너비 추출

# 최적의 카메라 행렬 계산 (alpha=1: 원본 이미지의 모든 픽셀 보존)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

# undistort()를 사용하여 왜곡 보정
dst = cv2.undistort(test_img, K, dist, None, newcameramtx)

# 결과 시각화
cv2.imshow('Original Image', test_img) # 원본 이미지 표시
cv2.imshow('Undistorted Image', dst) # 왜곡 보정된 이미지 표시
cv2.waitKey(0) # 키 입력 대기
cv2.destroyAllWindows() # 모든 창 닫기
