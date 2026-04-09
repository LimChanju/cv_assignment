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