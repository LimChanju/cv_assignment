# OpenCV 기초 과제 (1주차 Assignment 1~3)

## 과제 1: 이미지 불러오기 및 그레이스케일 변환 🏁
* **설명:** 원본 이미지를 불러온 뒤 그레이스케일로 변환하고, 두 이미지를 가로로 연결하여 출력합니다.
* **배경 지식:**
<img width="1244" height="682" alt="image" src="https://github.com/user-attachments/assets/0d85257f-d35e-4cef-9249-e143bac849e0" />
* **주요 구현 포인트:** `np.hstack()`을 사용하여 이미지를 병합할 때 배열의 차원(shape)을 일치시키는 것이 중요합니다. 1채널인 그레이스케일 이미지를 배열 병합이 가능하도록 3채널 차원으로 변환(`cv.COLOR_GRAY2BGR`)하여 차원 불일치 에러를 방지했습니다.
* **핵심 코드:**
```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 그레이스케일로 변환

gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # 1ch img를 3ch shape으로만 변경. 색상은 흑백으로 유지

combined_img = np.hstack((img, gray_3ch)) # 원본과 그레이스케일 이미지 가로로 연결
```

<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python
import cv2 as cv
import numpy as np

def main():
    # 1. 이미지 로드
    img_path = 'soccer.jpg' # 이미지 경로 설정
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
```
</div>
</details>

* **중간 결과물:** ![흑백으로 변환한 이미지](assign_1/soccer_gray.jpg)
* **최종 결과물:**
  ![합성된 결과 화면](assign_1/assignment_01_result_img.png)

## 과제 2: 페인팅 붓 크기 조절 기능 추가 🖌️
* **배경 지식:**
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/f0e2b78d-ed7f-43fe-9bcb-a6d9c96b249f" />
<img width="749" height="129" alt="image" src="https://github.com/user-attachments/assets/3e8c32d6-3340-4bea-a162-fbd2324ac682" />


* **설명:** 마우스 콜백을 이용해 드래그로 그림을 그리고, 키보드 입력(+, -)으로 붓 크기를 조절하는 페인팅 과제입니다.
* **주요 구현 포인트:** 콜백 함수(`mouse_callback`)와 메인`while` 루프 간의 상태(State) 동기화를 위해 전역 변수를 활용하여 현재 마우스 드래그 여부, 붓 크기, 색상 상태를 추적하고 업데이트하도록 구현하였습니다.
* **핵심 코드:**
```python
# 1. 마우스 콜백: 전역 변수를 활용한 상태 동기화 및 드래그 그리기
def mouse_callback(event, x, y, flags, param):
    global brush_size, is_drawing, current_color 
    img = param 
    
    # 좌클릭(파란색) 시작
    if event == cv.EVENT_LBUTTONDOWN:
        is_drawing, current_color = True, (255, 0, 0)
        cv.circle(img, (x, y), brush_size, current_color, -1)
        
    # 드래그 중: 설정된 색상과 크기로 연속 그리기
    elif event == cv.EVENT_MOUSEMOVE and is_drawing:
        cv.circle(img, (x, y), brush_size, current_color, -1)
        
# 2. 메인 루프: 키보드 입력을 통한 붓 크기 동적 조절 (1~15 제한)
key = cv.waitKey(1) & 0xFF
if key == ord('+') or key == ord('='):
    brush_size = min(15, brush_size + 1) # 최대 제한
elif key == ord('-') or key == ord('_'):
    brush_size = max(1, brush_size - 1)  # 최소 제한
```
<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python
import cv2 as cv
import numpy as np

# 상태를 추적하기 위한 전역 변수 설정
brush_size = 5 # 붓 크기 초기값
is_drawing = False # 그리기 상태 추적
current_color = (255, 0, 0) # 초기 색상은 파란색으로 설정

def mouse_callback(event, x, y, flags, param):
    global brush_size, is_drawing, current_color # 전역 변수를 활용하여 main함수의 while 루프 간의 상태 동기화
    
    img = param # 메인 함수에서 전달받은 캔버스 이미지
    
    # 좌클릭 : 파란색 그리기 시작
    if event == cv.EVENT_LBUTTONDOWN:
        is_drawing = True # 그리기 시작
        current_color = (255, 0, 0) # 파란색으로 설정 (초기값과 같음)
        cv.circle(img, (x, y), brush_size, current_color, -1) # 원 그리기 (채우기)

    # 우클릭 : 빨간색 그리기 시작
    elif event == cv.EVENT_RBUTTONDOWN:
        is_drawing = True # 그리기 시작
        current_color = (0, 0, 255) # 빨간색으로 설정
        cv.circle(img, (x, y), brush_size, current_color, -1) # 원 그리기 (채우기)
        
    # 드래그 중 : 현재 설정된 색상과 붓 크기로 연속 그리기
    elif event == cv.EVENT_MOUSEMOVE and is_drawing: # 마우스가 이동 중이고 그리기 상태인 경우
        cv.circle(img, (x, y), brush_size, current_color, -1) # 원 그리기 (채우기)
        
    # 클릭 해제 시 그리기 종료
    elif event in [cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP]:
        is_drawing = False
        
def main():
    global brush_size # main 함수에서 키보드 입력을 통해 붓 크기를 조절하기 위해 전역 변수로 선언
    
    # 그릴 수 있는 캔버스 생성 (600x800 크기, 흰색 배경)
    img_path = 'soccer.jpg' # 이미지 경로 설정
    img = cv.imread(img_path) # 이미지 로드
    window_name = 'Drawing Canvas' # 창 이름 설정
    
    cv.namedWindow(window_name) # 창 생성
    
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
```
</div>
</details>

* **최종 결과물 (크기 조절 및 드래그 적용):**
  ![최종 드로잉 결과](assign_1/assignment_02_result.png)

## 과제 3: 마우스로 영역 선택 및 ROI 추출 👀
* **배경 지식:**
<img width="992" height="570" alt="image" src="https://github.com/user-attachments/assets/b09540d2-27e4-4bbf-b431-ad276911ae4b" />

* **설명:** 드래그하여 관심 영역(ROI)을 지정하고, 이를 잘라내어 별도의 창에 띄우거나 파일로 저장합니다.
* **주요 구현 포인트:**
1. **슬라이싱 방향 예외 처리:** 마우스 드래그 방향(역방향 드래그)에 상관없이 Numpy 슬라이싱이 정상 작동하도록 `min()`, `max()` 함수를 사용해 좌표를 정렬했습니다.
2. **잔상 초기화:** 드래그 중이거나 새로운 영역을 선택할 때 이전 사각형의 잔상이 남지 않도록, `clone.copy()`를 활용해 매 렌더링마다 이미지를 원본 상태로 초기화하는 로직을 적용했습니다.
* **핵심 코드:**
```python
# 1. 역방향 드래그 시 Numpy 슬라이싱 오류(빈 배열 반환)를 방지하기 위한 좌표 정렬
min_x, max_x = min(start_x, x), max(start_x, x)
min_y, max_y = min(start_y, y), max(start_y, y)

# 2. 잔상 초기화: 이전 사각형 궤적을 지우기 위해 도화지를 원본으로 덮어씌움
img = clone.copy()
cv.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2) # 최종 ROI 표시

# 3. Numpy 슬라이싱을 이용한 실제 ROI 이미지 데이터 추출 (배열은 y(행), x(열) 순서)
roi_img = clone[min_y:max_y, min_x:max_x]
```
<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python# ROI(Region Of Interest) 추출하기
import cv2 as cv
import numpy as np

# 전역 변수 초기화
is_dragging = False # 마우스 드래그 상태
start_x, start_y = -1, -1 # 드로잉 시작 좌표
roi_img = None # ROI 이미지 저장 변수

# 원본 이미지를 복사해 둘 변수 (초기화용)
clone = None # 원본 이미지의 복사본, 드로잉 시 원본 훼손 방지용
img = None # 현재 드로잉 중인 이미지 (원본에서 드로잉이 적용된 버전)

def mouse_callback(event, x, y, flags, param):
    global is_dragging, start_x, start_y, img, clone, roi_img # 마우스 콜백 함수에서 사용할 전역 변수들
    
    # 1. 좌클릭 : 드래그 시작점 저장
    if event == cv.EVENT_LBUTTONDOWN:
        is_dragging = True # 드래그 시작
        start_x, start_y = x, y # 드래그 시작 좌표 저장
        
    # 2. 마우스 이동 : 드래그 중일 때 사각형 시각화
    elif event == cv.EVENT_MOUSEMOVE:
        if is_dragging:
            # 원본 이미지가 훼손되지 않도록 clone본을 복사해서 그 위에 그림
            img_draw = clone.copy() # 드로잉 시 원본 훼손 방지용 복사본
            cv.rectangle(img_draw, (start_x, start_y), (x, y), (0, 255, 0), 2) # 녹색 사각형 그리기
            cv.imshow('ROI Selector', img_draw) # 드로잉된 이미지 실시간으로 또 다른 창을 띄워 보여주기
            
    # 3. 좌클릭 해제 : 드래그 종료 및 ROI 추출
    elif event == cv.EVENT_LBUTTONUP:
        is_dragging = False # 드래그 종료
        
        # 드래그 방향에 상관없이 슬라이싱이 가능하도록 최소/최대 좌표 계산
        min_x, max_x = min(start_x, x), max(start_x, x)
        min_y, max_y = min(start_y, y), max(start_y, y)
        
        # 클릭만 하고 드래그를 안 했을 경우(면적이 0) 예외 처리
        if min_x == max_x or min_y == max_y:
            return
        
        # 이전 빨간 사각형들의 잔상을 없애기 위해 도화지를 원본으로 초기화
        img = clone.copy()
        
        # 원본 이미지에서 ROI 영역만큼 사각형을 확정해서 그림
        cv.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2) # 빨간색 사각형으로 최종 ROI 표시
        cv.imshow('ROI Selector', img) # 최종 ROI가 그려진 이미지 출력
        
        # Numpy 슬라이싱을 이용한 ROI 추출 (배열은 y, x 순서로 인덱싱)
        roi_img = clone[min_y:max_y, min_x:max_x]
        
        # 추출한 ROI를 별도의 창에 출력
        cv.imshow('Extracted ROI', roi_img)
        
def main():
    global img, clone, roi_img # main 함수 내 while 루프 안에서 사용할 전역 변수들
    
    # 1. 이미지 로드
    img_path = 'soccer.jpg' # 이미지 경로 설정
    img = cv.imread(img_path) # 이미지 로드
    
    if img is None:
        print(f"에러: '{img_path}' 이미지를 불러올 수 없습니다. 파일 경로를 다시 확인하세요.")
        return
    
    # 2. 원본 이미지 보존을 위해 clone 생성
    clone = img.copy()
    
    cv.namedWindow('ROI Selector') # ROI Selector 창 생성
    cv.setMouseCallback('ROI Selector', mouse_callback) # 마우스 콜백 함수 등록
    
    print("마우스로 드래그하여 ROI를 선택하세요. (q: 종료 / r: 리셋 / s: 저장)")
    
    while True:
        # 최초 1회 또는 화면이 업데이트되지 않을 때 원본 출력용
        if not is_dragging:
            cv.imshow('ROI Selector', img) # 드로잉이 없는 경우 원본 이미지 출력
            
        key = cv.waitKey(1) & 0xFF # 키 입력 대기
        
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
```
</div>
</details>

* **중간 결과물 (영역 드래그 중):**
  ![드래그 중인 화면](assign_1/assignment_03_intermediate_img.png)
  
* **최종 결과물 (추출 및 저장된 ROI):**
  ![저장된 ROI 이미지](assign_1/extracted_roi.jpg)
