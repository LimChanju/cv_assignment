import setuptools # Python 3.12 이상 환경에서 pkg_resources 에러 우회용
import cv2
import numpy as np
import os
import sys

try:
    from sort import Sort
except ImportError:
    print("오류: sort 모듈을 찾을 수 없습니다. 공식 GitHub에서 sort.py를 다운로드하세요.")

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("오류: deep_sort_realtime 모듈을 찾을 수 없습니다. 'pip install deep-sort-realtime'을 실행하세요.")

def main():
    # 1. 경로 변수 설정
    cfg_path = "yolov3.cfg"
    weights_path = "yolov3.weights"
    video_path = "slow_traffic_small.mp4"
    img_path = "./results/comparison_result.jpg"
    
    if not (os.path.exists(cfg_path) and os.path.exists(weights_path) and os.path.exists(video_path)):
        print("오류: YOLO 파일 또는 동영상 파일이 존재하지 않습니다.")
        return

    # 2. YOLOv3 로드
    net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except TypeError:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 3. 비디오 캡처 및 두 개의 추적기 초기화
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("오류: 비디오를 열 수 없습니다.")
        return

    # SORT 추적기
    mot_tracker_sort = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    # Deep SORT 추적기
    mot_tracker_deepsort = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
    
    last_combined_frame = None 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # 화면 비교를 위해 두 개의 프레임 복사본 생성
        frame_sort = frame.copy()
        frame_deepsort = frame.copy()

        # --- 객체 검출 (YOLOv3) ---
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # --- 추적기 입력 포맷팅 ---
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

        # --- SORT 업데이트 및 시각화 ---
        dets_sort_np = np.array(dets_sort) if len(dets_sort) > 0 else np.empty((0, 5))
        trackers_sort = mot_tracker_sort.update(dets_sort_np)

        for d in trackers_sort:
            x1, y1, x2, y2, track_id = [int(i) for i in d]
            cv2.rectangle(frame_sort, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_sort, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_sort, "Standard SORT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # --- Deep SORT 업데이트 및 시각화 ---
        # 특징 추출을 위해 원본 frame을 파라미터로 함께 전달해야 합니다.
        tracks_deepsort = mot_tracker_deepsort.update_tracks(dets_deepsort, frame=frame)

        for track in tracks_deepsort:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cv2.rectangle(frame_deepsort, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame_deepsort, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame_deepsort, "Deep SORT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # --- 결과 병합 및 출력 ---
        # 두 화면을 원본 크기 그대로 이어붙이면 화면이 너무 커질 수 있어 절반으로 축소
        resized_sort = cv2.resize(frame_sort, (width // 2, height // 2))
        resized_deepsort = cv2.resize(frame_deepsort, (width // 2, height // 2))
        
        # 좌우 병합 (hconcat)
        combined_frame = cv2.hconcat([resized_sort, resized_deepsort])
        
        cv2.imshow("SORT vs Deep SORT Comparison", combined_frame)
        last_combined_frame = combined_frame.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 저장 로직
    if last_combined_frame is not None:
        cv2.imwrite(img_path, last_combined_frame)
        print(f"객체 추적이 완료되었습니다. 비교 결과 마지막 프레임이 저장되었습니다: {img_path}")
    else:
        print("저장할 프레임 데이터가 존재하지 않습니다.")

if __name__ == "__main__":
    main()