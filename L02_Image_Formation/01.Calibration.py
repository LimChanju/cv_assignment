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
