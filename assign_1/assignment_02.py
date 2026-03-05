import cv2 as cv
import numpy as np

# 상태를 추적하기 위한 전역 변수 설정
brush_size = 5
is_drawing = False
current_color = (255, 0, 0) # 초기 색상은 파란색 (BGR 기준)

def mouse_callback(event, x, y, flags, param):
    global brush_size, is_drawing, current_color
    
    img = param # 메인 함수에서 전달받은 캔버스 이미지
    
    # 좌클릭 : 파란색 그리기 시작
    if event == cv.EVENT_LBUTTONDOWN:
        is_drawing = True
        current_color = (255, 0, 0) # 파란색
        cv.circle(img, (x, y), brush_size, current_color, -1) # 원 그리기 (채우기)

    # 우클릭 : 빨간색 그리기 시작
    elif event == cv.EVENT_RBUTTONDOWN:
        is_drawing = True
        current_color = (0, 0, 255) # 빨간색
        cv.circle(img, (x, y), brush_size, current_color, -1) # 원 그리기 (채우기)
        
    # 드래그 중 : 현재 설정된 색상과 붓 크기로 연속 그리기
    elif event == cv.EVENT_MOUSEMOVE and is_drawing:
        cv.circle(img, (x, y), brush_size, current_color, -1) # 원 그리기 (채우기)
        
    # 클릭 해제 시 그리기 종료
    elif event in [cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP]:
        is_drawing = False
        
def main():
    global brush_size
    
    # 그릴 수 있는 캔버스 생성 (600x800 크기, 흰색 배경)
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    window_name = 'Drawing Canvas'
    
    cv.namedWindow(window_name)
    
    # 마우스 콜백 함수 등록 (img 객체 -> param으로 전달)
    cv.setMouseCallback(window_name, mouse_callback, param=img)
    
    print("페인팅을 시작합니다. (q: 종료 / +: 붓 크기 증가 / -: 붓 크기 감소)")
    
    while True:
        cv.imshow(window_name, img) # 캔버스 이미지 표시
        
        # 1ms 대기하며 키보드 입력 받기
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('q'): # 'q' 키를 누르면 종료
            break
        
        # 붓 크기 조절
        # 키보드 '+'와 '=' 키 모두 고려
        elif key == ord('+') or key == ord('='):
            brush_size += 1
            if brush_size > 15: # 최대 붓 크기 제한
                brush_size = 15
            print(f"현재 붓 크기: {brush_size}")
            
        # 붓 크기 감소
        # 키보드 '-'와 '_' 키 모두 고려
        elif key == ord('-') or key == ord('_'):
            brush_size -= 1
            if brush_size < 1: # 최소 붓 크기 제한
                brush_size = 1
            print(f"현재 붓 크기: {brush_size}")
            
    cv.destroyAllWindows() # 모든 창 닫기
    
if __name__ == "__main__":
    main()