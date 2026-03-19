# 이미지 형성 과제 (3주차 Assignment 1~3)

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
# cv.magnitude(x, y)는 x와 y의 크기를 계산하여 에지 강도를 구하는 함수. 여기서는 sobel_x와 sobel_y를 사용하여 각 픽셀에서의 에지 강도를 계산합니다.
magnitude = cv.magnitude(sobel_x, sobel_y)

# 4. 시각화 및 저장을 위해 8비트 이미지로 정규화 및 변환
# cv.convertScaleAbs(src)는 입력 이미지의 절대값을 계산하고, 결과를 uint8로 변환하는 함수. 에지 강도는 양수이므로 절대값을 취하는 것은 큰 의미가 없지만, 이 함수를 사용하여 결과를 uint8로 변환할 수 있음.
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

* **결과 이미지**
<img width="1000" height="500" alt="edge_comparison_plot" src="https://github.com/user-attachments/assets/93b4a770-c241-4672-814d-fd56bccd1a15" />

---

## 과제 2: 캐니 에지 및 허프 변환을 이용한 직선 검출 🔍
* **설명:** 캐니 에지 검출과 확률적 허프 변환을 이용하여 복잡한 배경을 가진 이미지에서 의미 있는 직선을 검출하는 것을 목표로 함
* **배경 지식:**
  - 캐니 에지 검출 (Canny Edge Detection): 노이즈에 민감한 미분 연산의 단점을 보완하기 위해 가우시안 필터를 이용해 노이즈를 줄이고 그래디언트를 계산 후 비최대 억제와 이중 임계값 처리를 거쳐 더 정확하고 얇은 윤곽선을 추출하는 알고리즘이다.
  - 허프 변환 (Hough Transform): 이미지 내의 모든 에지 픽셀을 (ρ,θ)와 같은 파라미터 공간으로 변환하고 축적 배열에서 투표가 많이 모이는 지점을 찾아 직선을 검출하는 방법이다.

* **주요 구현 포인트:**
1. **노이즈 제거:** `cv.GaussianBlur`를 적용해 이미지의 노이즈가 많은 배경과 질감의 미세한 패턴이 에지로 검출되는 것을 방지함
2. **파라미터 튜닝 최적화:** `cv.HoughLinesP()`의 파라미터를 수정하여 이미지에 맞는 파라미터를 찾아내는 과정을 거침
3. **ROI 적용:** ROI를 다보탑 이미지에 적용하여 다보탑의 직선만이라도 제대로 검출할 수 있도록 영역을 제한하였음 

* **핵심 코드:**
```python
# 1. ROI 설정 (Numpy Slicing)
x1, y1 = 196, 64
x2, y2 = 636, 484
roi_gray = gray[y1:y2, x1:x2]

# 2. ROI 영역 내 노이즈 제거 및 에지 검출
blurred = cv.GaussianBlur(roi_gray, (7, 7), 0) # 가우시안 블러(7X7)를 적용해 노이즈 제거
edges = cv.Canny(blurred, 100, 200) # 노이즈 제거된 이미지를 이용해 캐니 에지 검출 진행

# 3. 허프 변환
# (배경 노이즈가 제거되었으므로 다보탑의 짧은 선들을 찾도록 완화된 기준 적용)
rho = 1                     # 거리 해상도 (픽셀 단위)
theta = np.pi / 180         # 각도 해상도 (라디안 단위, 여기서는 1도)
threshold = 60              # 직선으로 판단할 최소 교차점(투표) 수
min_line_length = 30        # 선분의 최소 길이
max_line_gap = 5            # 동일한 선상에 있는 선분들 사이의 최대 허용 간격
lines = cv.HoughLinesP(edges, rho, theta, threshold, 
                        minLineLength=min_line_length, 
                        maxLineGap=max_line_gap)

# 4. 검출된 좌표는 ROI 내부에서의 좌표(상대 좌표)이기 때문에 ROI 시작 오프셋을 더해 원본 이미지의 절대 좌표로 변환해야 함
if lines is not None:
    for line in lines:
        x_line1, y_line1, x_line2, y_line2 = line[0]
        abs_x1, abs_y1 = x_line1 + x1, y_line1 + y1
        abs_x2, abs_y2 = x_line2 + x1, y_line2 + y1
        cv.line(line_img, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
```

<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    img_path = 'images/dabo.jpg' 
    img = cv.imread(img_path)

    if img is None:
        print(f"에러: '{img_path}' 이미지를 불러올 수 없습니다. 경로를 확인하세요.")
        return

    # 직선을 그릴 복사본 생성
    line_img = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 다보탑 위주 ROI 좌표 적용
    x1, y1 = 196, 64
    x2, y2 = 636, 484

    # Numpy 슬라이싱 [y축:y축, x축:x축] 을 이용해 다보탑 영역만 크롭
    roi_gray = gray[y1:y2, x1:x2]

    # 원본 이미지에 ROI 영역 시각적 표시 (파란색 사각형)
    cv.rectangle(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 2. ROI 영역 내 노이즈 제거 및 에지 검출
    blurred = cv.GaussianBlur(roi_gray, (7, 7), 0) # 가우시안 블러를 적용해 노이즈 제거
    edges = cv.Canny(blurred, 100, 200) # 노이즈 제거된 이미지를 이용해 캐니 에지 검출 진행

    # 3. 다보탑 구조 검출에 맞춘 허프 변환 파라미터 
    # (배경 노이즈가 제거되었으므로 다보탑의 짧은 선들을 찾도록 완화된 기준 적용)
    rho = 1                     # 거리 해상도 (픽셀 단위)
    theta = np.pi / 180         # 각도 해상도 (라디안 단위, 여기서는 1도)
    threshold = 60              # 직선으로 판단할 최소 교차점(투표) 수
    min_line_length = 30        # 선분의 최소 길이
    max_line_gap = 5            # 동일한 선상에 있는 선분들 사이의 최대 허용 간격

    lines = cv.HoughLinesP(edges, rho, theta, threshold, 
                           minLineLength=min_line_length, 
                           maxLineGap=max_line_gap)

    # 4. 검출된 직선을 원본 이미지의 절대 좌표로 변환하여 그리기
    if lines is not None:
        for line in lines:
            x_line1, y_line1, x_line2, y_line2 = line[0]
            
            # 절대 좌표 = ROI 내부 좌표 + ROI 시작 오프셋
            # 검출된 좌표는 ROI 내부에서의 좌표이기 때문에 ROI 시작 오프셋을 더해 원본 이미지의 절대 좌표로 변환해야 함
            abs_x1 = x_line1 + x1
            abs_y1 = y_line1 + y1
            abs_x2 = x_line2 + x1
            abs_y2 = y_line2 + y1
            
            cv.line(line_img, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
    else:
        print("ROI 영역 내에서 직선이 검출되지 않았습니다.")

    # 5. 시각화 및 결과 저장
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    line_img_rgb = cv.cvtColor(line_img, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(line_img_rgb)
    plt.title('Hough Lines on ROI')
    plt.axis('off')

    plt.tight_layout()
    
    # 저장 디렉토리 처리
    output_dir = 'result_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'roi_hough_lines.png'))
    
    plt.show()

if __name__ == "__main__":
    main()
```

</details>

* **주요 결과물:**
<img width="1200" height="600" alt="hough_lines_comparison(threshold=150,max=100,min=10)" src="https://github.com/user-attachments/assets/f3006098-3fe4-4e17-8e55-7ac8f57713c8" />

- 초기 결과 : `cv.HoughLinesP()`의 파라미터만을 수정한 결과, 아래 직선보다 배경의 나무나 기와지붕에 더 많이 생성되고 있음을 확인한 후, blur처리하여 배경의 패턴을 직선으로 검출하지 못하도록 함


<img width="2327" height="1113" alt="canny_hough_strategy" src="https://github.com/user-attachments/assets/9c042535-f623-4a2a-9d21-edc1d0669d2b" />

- 중간 결과 : `cv.GaussianBlur()`를 5X5로 적용하여 실험 진행, 7X7까지 적용해 봄

<img width="1200" height="600" alt="hough_lines_comparison(blur=7, threshold=90,min=25,max=4)" src="https://github.com/user-attachments/assets/de453b3f-34d4-492c-ad1e-d26f30fd3352" />

- 중간 결과 2 : blur를 7X7, threshold=90, min=25, max=4로 적용, 기와 지붕에 여전히 FP 존재


<img width="1200" height="600" alt="roi_hough_lines" src="https://github.com/user-attachments/assets/aa85e360-35f6-486d-b7fa-13b1beca07f6" />

- 최종 결과 : ROI를 적용하여 다보탑 위주로만 직선을 검출하도록 수정
```
x1, y1 = 196, 64
x2, y2 = 636, 484
roi_gray = gray[y1:y2, x1:x2]
```

---

## 과제 3: GrabCut을 이용한 대화식 영역 분할 및 객체 추출 💭
* **설명:** GrabCut을 활용하여 원본 이미지에서 사용자가 지정한 사각형 영역을 바탕으로 객체와 배경을 분리하는 대화식 영상 분할 기법
* **배경 지식:**
  - 가우시안 혼합 모델 (GMM, Gaussian Mixture Model): 사용자가 지정한 ROI의 외부를 확실한 배경으로 간주하고, 내부를 객체와 배경이 혼합한 영역으로 가정한 뒤, 객체와 배경의 색상 분포를 GMM으로 모델링
  - 그래프 컷 (Graph Cut): 각 픽셀을 그래프의 노드로 삼고, 픽셀 간의 색상 유사도를 간선의 가중치로 설정. 이후 Min-Cut/Max-Flow 알고리즘을 적용해 에너지를 최소화하는 방향으로 객체와 배경을 분리함.

* **주요 구현 포인트:**
1. **모델 초기화:** `bgdModel`과 `fgdModel`을 `np.zeros((1, 65), np.float64)` 배열로 초기화하여 GMM이 사용할 메모리 공간을 할당
2. **초기 사각형 지정 `(rect)`:** 추출하고자 하는 객체를 포함하는 최소한의 사각형의 좌표 `(x, y, width, height)`를 지정
3. **GrabCut 수행 `(cv.grabCut)`:** `cv.GC_INIT_WITH_RECT` 모드를 사용하여 사각형 기반의 분할을 수행, 알고리즘 실행 후 입력된 mask 배열은 0~3 사이의 값으로 업데이트됨
   - 0 (`cv.GC_BGD`): 확실한 배경
   - 1 (`cv.GC_FGD`): 확실한 객체
   - 2 (`cv.GC_PR_BGD`): 배경일 가능성이 높은 영역
   - 3 (`cv.GC_PR_FGD`): 객체일 가능성이 높은 영역
4. **마스크 이진화 및 배경 제거:** `np.where()`를 사용하여 `mask` 값이 0 또는 2인 경우 0으로, 1 또는 3인 경우 1로 변환한 후 이진화된 마스크를 원본 이미지에 곱하여 배경 픽셀을 검은색으로 제거함

* **핵심 코드:**
```python
# 1. 마스크 및 GMM 모델 초기화 (bgdModel과 fgdModel은 np.zeros((1, 65), np.float64)로 초기화)
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 2. 초기 사각형 영역 설정 (x, y, width, height)
# 이미지 가장자리에서 50 픽셀씩 떨어진 사각형 영역 설정 (x=50, y=50, width=이미지 너비-100, height=이미지 높이-100)
rect = (50, 50, img.shape[1]-100, img.shape[0]-100)

# 3. GrabCut 알고리즘 수행
cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_RECT)

# 4. 마스크 값을 0(배경) 또는 1(전경)로 변경
# 배경으로 분류된 픽셀은 0, 객체로 분류된 픽셀은 1로 설정하여 mask2를 생성
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

# 5. 배경 제거 (원본 이미지에 마스크 곱하기)
# mask2는 2차원이므로 원본 이미지(3차원)와 곱하기 위해 차원을 하나 추가(np.newaxis)
result_img = img_rgb * mask2[:, :, np.newaxis]

```

<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. 이미지 경로 설정
    img_path = 'images/coffee_cup.jpg' 
    
    # 이미지 읽기 및 예외 처리
    img = cv.imread(img_path)
    if img is None:
        print(f"[{img_path}] 이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
        return

    # matplotlib 출력을 위해 BGR 색상 공간을 RGB로 변환
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # 2. 마스크 및 GMM 모델 초기화 (bgdModel과 fgdModel은 np.zeros((1, 65), np.float64)로 초기화)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 3. 초기 사각형 영역 설정 (x, y, width, height)
    rect = (50, 50, img.shape[1]-100, img.shape[0]-100) # 이미지 가장자리에서 50 픽셀씩 떨어진 사각형 영역 설정 (x=50, y=50, width=이미지 너비-100, height=이미지 높이-100)

    # 4. GrabCut 알고리즘 수행 (대화식 분할)
    # grabCut 함수는 원본 이미지, 마스크, 초기 사각형 영역, 배경 모델, 객체 모델, 반복 횟수, 모드를 인자로 받음
    # 여기서는 초기 사각형 영역을 사용하여 GrabCut을 수행하도록 설정
    cv.grabCut(img_rgb, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_RECT) 

    # 5. 마스크 값을 0 또는 1로 변경
    # cv.GC_BGD(0), cv.GC_PR_BGD(2) -> 0 (배경)
    # cv.GC_FGD(1), cv.GC_PR_FGD(3) -> 1 (전경)
    # np.where 함수를 사용하여 mask에서 배경과 객체를 구분
    # 배경으로 분류된 픽셀은 0, 객체로 분류된 픽셀은 1로 설정하여 mask2를 생성
    mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
    

    # 6. 배경 제거 (원본 이미지에 마스크 곱하기)
    # mask2는 2차원이므로 원본 이미지(3차원)와 곱하기 위해 차원을 하나 추가(np.newaxis)
    result_img = img_rgb * mask2[:, :, np.newaxis]

    # 7. matplotlib을 이용한 세 가지 이미지 나란히 시각화
    plt.figure(figsize=(15, 5))
    
    # 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # 마스크 이미지 (시각화를 위해 1인 값을 255로 스케일링)
    plt.subplot(1, 3, 2)
    plt.imshow(mask2 * 255, cmap='gray') 
    plt.title('Mask Image')
    plt.axis('off')

    # 배경 제거 이미지
    plt.subplot(1, 3, 3)
    plt.imshow(result_img)
    plt.title('Background Removed')
    plt.axis('off')

    plt.tight_layout()
    
    # 8. 결과 이미지 저장
    output_dir = 'result_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt_output_path = os.path.join(output_dir, 'grabcut_result_plot.png')
    plt.savefig(plt_output_path)
    
    # 화면에 출력
    plt.show()

if __name__ == '__main__':
    main()
```

</details>

* **주요 결과물:**
<img width="1500" height="500" alt="grabcut_result_plot" src="https://github.com/user-attachments/assets/b45f58c6-2029-48c3-b17d-6686519dd4a7" />

