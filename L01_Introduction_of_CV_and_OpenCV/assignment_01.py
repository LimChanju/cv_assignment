import cv2 as cv
import numpy as np

def main():
    # 1. 이미지 로드
    img_path = './img/soccer.jpg' # 이미지 경로 설정
    img = cv.imread(img_path) # 이미지 로드
    
    # 이미지 제대로 로드되었는지 확인
    if img is None:
        print(f"에러: '{img_path}' 이미지를 불러올 수 없습니다. 파일 경로를 다시 확인하세요.") # 이미지가 제대로 로드되지 않았을 경우 에러 메시지 출력
        return
    
    # 2. 그레이스케일 변환
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 그레이스케일로 변환
    print(gray.shape) # 그레이스케일 이미지의 shape 출력 (높이, 너비)
    
    save_path = 'assign_1/soccer_gray.jpg' # 그레이스케일 이미지 저장 경로 설정
    is_saved = cv.imwrite(save_path, gray) # 그레이스케일 이미지 저장
    if is_saved:
        print(f"그레이스케일 이미지가 '{save_path}'로 저장되었습니다.") # 이미지 저장 성공 메시지 출력
    else:
        print(f"에러: '{save_path}' 경로에 이미지를 저장할 수 없습니다.") # 이미지 저장에 실패한 경우 에러 메시지 출력

    # 3. np.hstack()을 위한 차원 맞추기
    # 1채널(gray) 이미지를 3채널로 형태만 변경. 색상은 흑백으로 유지
    gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    
    # 4. 원본과 그레이스케일 이미지 가로로 연결
    combined_img = np.hstack((img, gray_3ch))
    
    # 5. 결과 이미지 출력 및 키 입력 대기
    cv.namedWindow('Original and Grayscale Image', cv.WINDOW_NORMAL) # 창 크기 조절 가능하게 설정
    cv.imshow('Original and Grayscale Image', combined_img) # Original and Grayscale Image 창에 combined_img 표시
    cv.waitKey(0) # 키 입력 대기
    cv.destroyAllWindows() # 모든 창 닫기
    
if __name__ == "__main__":
    main()