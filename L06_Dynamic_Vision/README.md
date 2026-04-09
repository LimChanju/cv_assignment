# Dynamic Vision (6주차 Assignment 1~2)

## 과제 1: SORT 알고리즘을 활용한 다중 객체 추적기 구현 🚘🚔
* **설명:** YOLOv3 딥러닝 모델을 이용하여 동영상 내의 객체를 실시간으로 탐지하고, 탐지된 객체들에 대해 Standard SORT와 Deep SORT 두 가지 다중 객체 추적(MOT, Multiple Object Tracking) 알고리즘을 동시에 적용하여 성능과 특징을 좌우 분할 화면(Side-by-Side)으로 비교 시각화하였다.
* **배경 지식:**
  - YOLOv3 (You Only Look Once): 빠르고 정확한 1-Stage 실시간 객체 탐지 모델입니다. 영상 프레임에서 객체의 위치(Bounding Box)와 클래스 신뢰도(Confidence)를 추출하여 추적기의 입력 데이터로 제공합니다.
  - SORT (Simple Online and Realtime Tracking): 칼만 필터(Kalman Filter)로 객체의 다음 위치를 예측하고, 헝가리안 알고리즘(Hungarian Algorithm)으로 예측된 위치와 실제 탐지된 위치 간의 IoU(교차 영역 비율)를 비교하여 동일 객체를 추적하는 가볍고 빠른 알고리즘입니다.
  - Deep SORT: 기존 SORT 알고리즘에 딥러닝 기반의 외형 특징(Appearance Feature) 추출 모델을 결합한 방식입니다. 객체들이 겹치거나 가려지는 상황(Occlusion)에서 단순 IoU 기반의 SORT가 겪는 ID 변경(ID Switching) 문제를 외형 유사도를 통해 보완합니다.
  - NMS (Non-Maximum Suppression): 하나의 객체에 여러 개의 바운딩 박스가 겹쳐서 탐지되었을 때, 신뢰도가 가장 높은 하나의 박스만 남기고 나머지를 제거하여 추적기의 입력 노이즈를 줄이는 필수 후처리 기법입니다.

* **주요 구현 포인트:**
1. **추적기별 맞춤형 입력 데이터 포맷팅:** YOLOv3와 NMS를 거친 탐지 결과를 추출한 뒤, SORT는 `[x1, y1, x2, y2, score]` 형태로, Deep SORT는 `([left, top, w, h], confidence, class_id)` 형태로 각각의 라이브러리가 요구하는 규격에 맞춰 데이터를 분리 및 가공합니다.
2. **Deep SORT의 원본 프레임 참조:** 단순 좌표만 계산하는 SORT와 달리, Deep SORT(`update_tracks`)는 객체의 외형적 특징점 임베딩(Embedding)을 추출하기 위해 탐지 좌표와 함께 현재 화면의 `frame` 원본 데이터를 파라미터로 넘겨주어 업데이트를 수행합니다.
3. **OpenCV 병합 연산을 통한 직관적 성능 비교:** 각 추적기가 그린 프레임을 원본 크기 대비 50%로 축소한 후, cv2.hconcat() 함수를 사용하여 좌측에는 SORT, 우측에는 Deep SORT 결과 화면을 나란히 이어 붙여 실시간 성능 차이를 시각적으로 비교 분석합니다.

* **핵심 코드:**
```python
# --- 1. 추적기별 입력 포맷팅 ---
dets_sort = []
dets_deepsort = []

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        conf = confidences[i]
        class_id = class_ids[i]
        
        # SORT 포맷: [x1, y1, x2, y2, score]
        dets_sort.append([x, y, x + w, y + h, conf])
        # Deep SORT 포맷: ([left, top, w, h], confidence, class_id_str)
        dets_deepsort.append(([x, y, w, h], conf, str(class_id)))

# --- 2. Standard SORT 업데이트 ---
dets_sort_np = np.array(dets_sort) if len(dets_sort) > 0 else np.empty((0, 5))
trackers_sort = mot_tracker_sort.update(dets_sort_np)

# --- 3. Deep SORT 업데이트 (특징 추출용 frame 파라미터 필수) ---
tracks_deepsort = mot_tracker_deepsort.update_tracks(dets_deepsort, frame=frame)

# --- 4. 좌우 분할 비교 화면 생성 ---
resized_sort = cv2.resize(frame_sort, (width // 2, height // 2))
resized_deepsort = cv2.resize(frame_deepsort, (width // 2, height // 2))
combined_frame = cv2.hconcat([resized_sort, resized_deepsort])
cv2.imshow("SORT vs Deep SORT Comparison", combined_frame)
```

<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python
import setuptools # Python 3.12 이상 환경에서 pkg_resources 에러 우회용
import cv2
import numpy as np
import os
import sys

# --- 라이브러리 임포트 및 예외 처리 ---
try:
    from sort import Sort
except ImportError:
    print("오류: sort 모듈을 찾을 수 없습니다. 공식 GitHub에서 sort.py를 다운로드하여 같은 폴더에 배치하세요.")
    sys.exit(1)

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("오류: deep_sort_realtime 모듈을 찾을 수 없습니다. 'pip install deep-sort-realtime'을 실행하세요.")
    sys.exit(1)

def main():
    # 1. 파일 경로 설정 (본인 환경에 맞게 수정)
    cfg_path = "yolov3.cfg"                     # YOLO 네트워크 구조 설정 파일
    weights_path = "yolov3.weights"             # YOLO 사전 학습된 가중치 파일
    video_path = "slow_traffic_small.mp4"       # 추적 테스트용 입력 동영상 파일
    img_path = "./results/comparison_result.jpg" # 최종 비교 결과가 저장될 이미지 경로
    
    # 필수 파일 존재 여부 검증
    if not (os.path.exists(cfg_path) and os.path.exists(weights_path) and os.path.exists(video_path)):
        print("오류: YOLO 파일 또는 동영상 파일이 지정된 경로에 존재하지 않습니다.")
        return

    # 2. YOLOv3 딥러닝 모델 로드
    net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = net.getLayerNames()
    
    # OpenCV 버전에 따른 호환성 처리 (출력 레이어 인덱스 추출)
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except TypeError:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 3. 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("오류: 비디오를 열 수 없습니다. 파일 손상이나 코덱 문제를 확인하세요.")
        return

    # 4. 추적기(Tracker) 초기화
    # [Standard SORT] 
    # 오직 바운딩 박스의 좌표와 칼만 필터(Kalman Filter)만 사용하여 객체의 다음 위치를 예측함.
    mot_tracker_sort = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # [Deep SORT]
    # 칼만 필터 예측과 더불어, 객체의 외형 특징(Appearance Feature)을 딥러닝으로 추출해 ID 스위칭을 방지함.
    mot_tracker_deepsort = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
    
    last_combined_frame = None 

    # 5. 메인 프레임 처리 루프 시작
    while True:
        ret, frame = cap.read()
        if not ret: # 동영상이 끝나면 루프 종료
            break

        height, width, _ = frame.shape

        # 화면 비교를 위해 원본 프레임을 두 개의 복사본으로 분리
        frame_sort = frame.copy()
        frame_deepsort = frame.copy()

        # --- [Phase 1: 객체 탐지 (YOLOv3)] ---
        # 이미지를 모델 입력 크기(416x416)에 맞게 전처리(정규화 및 크기 조정)
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers) # 네트워크 순전파(Forward pass) 실행

        class_ids = []  # 탐지된 객체의 클래스 번호
        confidences = [] # 탐지 신뢰도(확률)
        boxes = []       # [x, y, width, height] 좌표 배열

        # 출력 레이어에서 탐지 정보 추출
        for out in outs:
            for detection in out:
                scores = detection[5:] # 0~4는 박스 정보, 5부터는 80개 클래스에 대한 확률값
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # 신뢰도가 0.5(50%) 이상인 의미 있는 객체만 필터링
                if confidence > 0.5:
                    # YOLO는 박스의 중심점(center_x, center_y)을 반환하므로, 좌상단(x, y) 좌표로 변환
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-Maximum Suppression (NMS) 적용
        # 동일 객체에 대해 여러 개의 겹치는 박스가 생기는 것을 방지하여 가장 확실한 하나만 남김
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # --- [Phase 2: 추적기 입력 포맷팅] ---
        # 라이브러리마다 요구하는 입력 리스트의 형태가 다르므로 데이터를 재가공함
        dets_sort = []
        dets_deepsort = []
        
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                conf = confidences[i]
                class_id = class_ids[i]
                
                # Standard SORT 입력 규격: [좌상단X, 좌상단Y, 우하단X, 우하단Y, 신뢰도스코어]
                dets_sort.append([x, y, x + w, y + h, conf])
                
                # Deep SORT 입력 규격: ([좌상단X, 좌상단Y, 너비, 높이], 신뢰도스코어, 클래스ID_문자열)
                dets_deepsort.append(([x, y, w, h], conf, str(class_id)))

        # --- [Phase 3: Standard SORT 업데이트 및 시각화] ---
        # 탐지 결과가 없으면 빈 넘파이 배열을 전달, 있으면 array 변환 후 전달
        dets_sort_np = np.array(dets_sort) if len(dets_sort) > 0 else np.empty((0, 5))
        trackers_sort = mot_tracker_sort.update(dets_sort_np)

        # SORT가 반환한 추적 결과(좌표 및 부여된 고유 ID)를 프레임에 그림
        for d in trackers_sort:
            x1, y1, x2, y2, track_id = [int(i) for i in d]
            cv2.rectangle(frame_sort, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_sort, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 화면 좌측 상단에 타이틀 표기
        cv2.putText(frame_sort, "Standard SORT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # --- [Phase 4: Deep SORT 업데이트 및 시각화] ---
        # 외형 특징 추출(Re-ID)을 위해 단순 좌표뿐만 아니라 이미지 원본(frame=frame)을 반드시 함께 넘겨주어야 함.
        tracks_deepsort = mot_tracker_deepsort.update_tracks(dets_deepsort, frame=frame)

        for track in tracks_deepsort:
            # 연속으로 일정 횟수 이상 탐지되어 확정(Confirmed)된 객체만 시각화
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb() # Left, Top, Right, Bottom 좌표 반환 함수
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            
            cv2.rectangle(frame_deepsort, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_deepsort, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        cv2.putText(frame_deepsort, "Deep SORT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # --- [Phase 5: 결과 병합 및 출력] ---
        # 원본 크기의 이미지를 2장 나란히 붙이면 모니터 해상도를 초과할 수 있으므로 가로세로를 50% 축소
        resized_sort = cv2.resize(frame_sort, (width // 2, height // 2))
        resized_deepsort = cv2.resize(frame_deepsort, (width // 2, height // 2))
        
        # 축소된 두 이미지를 수평(Horizontal) 방향으로 병합 (좌: SORT, 우: Deep SORT)
        combined_frame = cv2.hconcat([resized_sort, resized_deepsort])
        
        # 결과 화면 출력
        cv2.imshow("SORT vs Deep SORT Comparison", combined_frame)
        last_combined_frame = combined_frame.copy() # 마지막 프레임 저장을 위해 백업

        # 'q' 키를 누르면 루프 강제 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. 리소스 반환 및 종료
    cap.release()
    cv2.destroyAllWindows()

    # 동영상 처리가 끝난 후, 과제 제출용으로 마지막 비교 프레임을 이미지로 저장
    if last_combined_frame is not None:
        cv2.imwrite(img_path, last_combined_frame)
        print(f"객체 추적이 완료되었습니다. 비교 결과 마지막 프레임이 저장되었습니다: {img_path}")
    else:
        print("저장할 프레임 데이터가 존재하지 않습니다.")

# 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()
```

</details>

* **결과 이미지**
<img width="640" height="211" alt="1" src="https://github.com/user-attachments/assets/b092c18e-32c5-4696-9977-168e97b84093" />

<img width="636" height="208" alt="2" src="https://github.com/user-attachments/assets/3147b952-ce0d-4eb1-94b6-fd299383f38a" />

![3](https://github.com/user-attachments/assets/2be121fe-d809-459e-80fc-8fa417e0566e)

---

## 과제 2: Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화 🥸
* **설명:** OpenCV를 통해 웹캠의 실시간 영상을 캡처하고, Google의 MediaPipe Face Mesh 머신러닝 파이프라인을 활용하여 얼굴 표면의 정밀한 랜드마크(기본 468개 및 눈동자 포함 478개)를 추출한 뒤, 이를 원본 프레임 위에 오버레이(Overlay)하여 실시간으로 시각화하는 과제입니다.
* **배경 지식:**
  - MediaPipe Face Mesh: 모바일 및 데스크톱 환경에서 실시간으로 작동하도록 최적화된 구글의 경량화된 3D 얼굴 랜드마크 모델입니다. 얼굴의 윤곽, 눈, 눈썹, 입술 등의 위치를 기하학적 3차원 좌표로 추정하여 AR 필터나 표정 인식 등에 널리 쓰입니다.
  - 정규화된 좌표계 (Normalized Coordinates): 딥러닝 모델이 반환하는 랜드마크의 x, y 좌표는 절대 픽셀 값이 아닌 0.0에서 1.0 사이의 비율 값(정규화)으로 출력됩니다. 따라서 실제 화면에 점을 찍으려면 원본 프레임의 너비(Width)와 높이(Height)를 곱해주는 스케일링 과정이 필수적입니다.
  - BGR vs RGB 색상 공간: OpenCV 라이브러리는 비디오 프레임을 읽어올 때 기본적으로 BGR(Blue, Green, Red) 순서의 채널을 사용하지만, MediaPipe 내부의 신경망 모델은 RGB 순서로 학습되었기 때문에 추론 전에 반드시 `cv2.cvtColor`를 이용한 색상 채널 변환이 선행되어야 합니다.

* **주요 구현 포인트:**
1. **메모리 복사 방지를 통한 추론 속도 최적화:** 프레임을 MediaPipe 모델에 전달하기 직전 `image.flags.writeable = False`로 설정하여 메모리 내부에서의 불필요한 데이터 복사를 막고 처리 속도를 높인 후, 렌더링 단계에서 다시 `True`로 되돌리는 메모리 관리 기법을 적용했습니다.
2. **내장 유틸리티를 활용한 자동 좌표 매핑 및 렌더링:** 힌트에 제시된 정규화 좌표 스케일링 계산을 직접 구현하는 대신, MediaPipe의 `drawing_utils`에 포함된 `draw_landmarks` 함수를 활용하여 400개가 넘는 점들의 좌표 변환과 시각화(점 찍기)를 단일 함수 호출로 깔끔하고 효율적으로 처리했습니다.

* **핵심 코드:**
```python
# --- 1. 색상 공간 변환 및 추론 최적화 ---
# 메모리 쓰기 방지로 연산 효율을 높이고 MediaPipe 규격에 맞게 BGR을 RGB로 변환
image.flags.writeable = False
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(image)

# 화면 출력을 위해 다시 BGR로 복구
image.flags.writeable = True
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# --- 2. 랜드마크 추출 및 시각화 ---
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # mp_drawing 유틸리티가 정규화된 좌표(0.0~1.0)를 이미지 해상도에 맞게 자동 변환하여 그려줌
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )
```

<details>
<summary><b>전체 코드 보기 (클릭)</b></summary>

```python
import cv2
import mediapipe as mp
import os

def main():
    # --- [1. 설정 및 초기화] ---
    # 과제 요구사항에 따른 결과 이미지 저장 경로
    img_path = "./results/facemesh_result.jpg" 
    
    # MediaPipe Face Mesh 모듈 및 그리기 유틸리티 초기화
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    
    # 랜드마크 시각화 스타일 설정 (요구사항: 점으로 표시)
    # 굵기(thickness)와 반지름(circle_radius)을 1로 설정하여 정밀한 점으로 표현하고, 색상은 초록색(0, 255, 0)으로 지정
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

    # FaceMesh 모델 파라미터 설정
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,       # False: 비디오 스트림 처리에 최적화 (이전 프레임의 정보를 활용해 추적 속도 향상)
        max_num_faces=1,               # 최대 탐지할 얼굴 수 (성능을 위해 1명으로 제한)
        refine_landmarks=True,         # True: 눈동자 주변을 포함하여 더 정밀한 478개의 랜드마크 추출
        min_detection_confidence=0.5,  # 초기 얼굴 탐지 신뢰도 임계값 (50%)
        min_tracking_confidence=0.5    # 랜드마크 추적 신뢰도 임계값 (50%)
    )

    # --- [2. 웹캠 캡처 시작] ---
    # 시스템의 기본 카메라(인덱스 0) 객체 생성
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다. 카메라 연결 및 권한을 확인하세요.")
        return

    print("프로그램이 시작되었습니다. 랜드마크 추적을 종료하려면 ESC 키를 누르세요.")
    
    last_frame = None # 최종 종료 시 저장할 프레임을 담을 변수

    # --- [3. 메인 프레임 처리 루프] ---
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라로부터 프레임을 읽을 수 없습니다.")
            break

        # [최적화 및 전처리 단계]
        # 1. 성능 향상: image 데이터의 메모리 복사를 방지하기 위해 쓰기 불가능 상태로 변경
        image.flags.writeable = False
        # 2. 색상 공간 변환: OpenCV는 BGR을 사용하지만, MediaPipe 모델은 RGB 이미지로 학습되었으므로 변환 필수
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 신경망 모델을 통해 얼굴 랜드마크 추론 실행
        results = face_mesh.process(image)

        # [후처리 및 렌더링 단계]
        # 화면에 점을 그려야 하므로 다시 쓰기 가능 상태로 변경하고, 화면 출력을 위해 BGR로 원복
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 탐지된 얼굴 랜드마크가 하나라도 존재한다면 시각화 진행
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # mp_drawing 유틸리티가 정규화된 랜드마크 좌표(0.0~1.0)를 
                # 현재 프레임의 실제 해상도(Width x Height)에 맞게 픽셀 좌표로 자동 매핑하여 렌더링함
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS, # 얼굴 윤곽선(Mesh) 연결 정보
                    landmark_drawing_spec=drawing_spec,         # 점 스타일
                    connection_drawing_spec=drawing_spec        # 선 스타일 (여기서는 둘 다 점 스타일로 통일)
                )
        
        # 완성된 프레임을 화면에 출력
        cv2.imshow('MediaPipe Face Mesh', image)
        
        # 마지막 프레임을 안전하게 복사하여 저장용 변수에 보관
        last_frame = image.copy()

        # --- [4. 종료 조건 처리] ---
        # cv2.waitKey(5): 5ms 동안 키 입력을 대기
        # 0xFF == 27: 입력된 키의 ASCII 코드가 27(ESC 키)인지 확인
        if cv2.waitKey(5) & 0xFF == 27:
            print("ESC 키가 눌려 프로그램을 종료합니다.")
            break

    # --- [5. 리소스 해제 및 결과 저장] ---
    # 할당된 메모리와 웹캠 하드웨어 제어권 반환
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

    # 루프 종료 후, 백업해둔 마지막 프레임이 존재한다면 지정된 경로에 이미지 파일로 저장
    if last_frame is not None:
        cv2.imwrite(img_path, last_frame)
        print(f"✅ 처리가 완료되었습니다. 캡처된 결과 프레임 저장 경로: {img_path}")
    else:
        print("저장할 프레임 데이터가 존재하지 않습니다.")

# 스크립트 직접 실행 시 메인 함수 호출
if __name__ == "__main__":
    main()
```

</details>

* **주요 결과물:**

![facemesh_result](https://github.com/user-attachments/assets/79bd48a7-ec71-4a97-bdd2-695ae993335a)

