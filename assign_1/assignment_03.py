# ROI(Region Of Interest) 추출하기
import cv2 as cv
import numpy as np

# 전역 변수 초기화
is_dragging = False # 마우스 드래그 상태
start_x, start_y = -1, -1 # 드로잉 시작 좌표
roi_img = None # ROI 이미지 저장 변수

# 원본 이미지를 복사해 둘 변수 (초기화용)
clone = None
img = None

def mouse_callback(event, x, y, flags, param):
    global is_dragging, start_x, start_y, img, clone, roi_img
    
    # 1. 좌클릭 : 드래그 시작점 저장
    if event == cv.EVENT_LBUTTONDOWN:
        is_drawing = True
        start_x, start_y = x, y
        
    # 2. 마우스 이동 : 드래그 중일 때 사각형 시각화
    elif event == cv.EVENT_MOUSEMOVE:
        if is_dragging:
            # 원본 이미지가 훼손되지 않도록 clone본을 복사해서 그 위에 그림
            img_draw = clone.copy()
            cv.rectangle(img_draw, (start_x, start_y), (x, y), (0, 255, 0), 2) # 녹색 사각형 그리기
            cv.imshow('ROI Selector', img_draw)
            
    # 3. 좌클릭 해제 : 드래그 종료 및 ROI 추출
    elif event == cv.EVENT_LBUTTONUP:
        is_dragging = False
        
        # 드래그 방향에 상관없이 슬라이싱이 가능하도록 최소/최대 좌표 계산
        min_x, max_x = min(start_x, x), max(start_x, x)
        min_y, max_y = min(start_y, y), max(start_y, y)
        
        # 클릭만 하고 드래그를 안 했을 경우(면적이 0) 예외 처리
        if min_x == max_x or min_y == max_y:
            return
        
        # 원본 이미지에서 ROI 영역만큼 사각형을 확정해서 그림
        cv.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2) # 빨간색 사각형으로 최종 ROI 표시
        cv.imshow('ROI Selector', img)
        
        # Numpy 슬라이싱을 이용한 ROI 추출 (배열은 y, x 순서로 인덱싱)
        roi_img = clone[min_y:max_y, min_x:max_x]
        
        # 추출한 ROI를 별도의 창에 출력
        cv.imshow('Extracted ROI', roi_img)
        
def main():
    global img, clone, roi_img
    
    # 1. 이미지 로드
    img_path = 'soccer.jpg' # 이미지 경로 설정
    img = cv.imread(img_path) # 이미지 로드
    
    if img is None:
        print(f"에러: '{img_path}' 이미지를 불러올 수 없습니다. 파일 경로를 다시 확인하세요.")
        return
    
    # 2. 원본 이미지 보존을 위해 clone 생성
    clone = img.copy()
    
    cv.namedWindow('ROI Selector')
    cv.setMouseCallback('ROI Selector', mouse_callback) # 마우스 콜백 함수 등록
    
    print("마우스로 드래그하여 ROI를 선택하세요. (q: 종료 / r: 리셋 / s: 저장)")
    
    while True:
        # 최초 1회 또는 화면이 업데이트되지 않을 때 원본 출력용
        if not is_dragging:
            cv.imshow('ROI Selector', img)
            
        key = cv.waitKey(1) & 0xFF
        
        # 'r' 키를 누르면 선택 영역 리셋
        if key == ord('r'):
            img = clone.copy() # 원본 이미지로 리셋
            roi_img = None # ROI 이미지 초기화
            cv.imshow('ROI Selector', img) # 리셋된 이미지 출력
            try:
                cv.destroyWindow('Extracted ROI') # ROI 창 닫기
            except cv.error:
                pass # ROI 창이 없으면 무시
            print("선택 영역이 리셋되었습니다.")
            
        # 's' 키를 누르면 ROI 이미지 저장
        elif key == ord('s'):
            if roi_img is not None:
                save_path = 'extracted_roi.jpg'
                cv.imwrite(save_path, roi_img)
                print(f"ROI 이미지가 '{save_path}'로 저장되었습니다.")
            else:
                print("저장할 ROI 이미지가 없습니다. 먼저 영역을 선택하세요.")
                
        # 'q' 키를 누르면 종료
        elif key == ord('q'):
            break
        
    cv.destroyAllWindows() # 모든 창 닫기
    
if __name__ == "__main__":
    main()