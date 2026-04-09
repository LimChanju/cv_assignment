import cv2
import mediapipe as mp
import os

def main():
    # 1. 설정 및 경로 변수
    img_path = "./results/facemesh_result.jpg" # 항상 사용하는 이미지 저장 경로
    
    # MediaPipe Face Mesh 모듈 초기화
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    # 테두리나 연결선 없이 점(landmark)만 강조하기 위한 설정
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

    # FaceMesh 모델 설정
    # static_image_mode=False: 비디오 스트림 처리 최적화
    # max_num_faces=1: 한 명의 얼굴만 추적
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 2. 웹캠 캡처 시작
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다. 카메라 연결을 확인하세요.")
        return

    print("프로그램이 시작되었습니다. 종료하려면 ESC 키를 누르세요.")
    
    last_frame = None

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라 프레임을 읽을 수 없습니다.")
            break

        # 성능 향상을 위해 이미지 쓰기 불가능 설정 후 RGB로 변환
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # 다시 BGR로 변환하여 그리기 준비
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 힌트: 정규화된 좌표를 이미지 크기에 맞게 변환하여 그리기
                # mp_drawing을 사용하면 468개(또는 정밀 모델 478개) 점을 자동으로 그려줍니다.
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )
        
        # 화면 출력
        cv2.imshow('MediaPipe Face Mesh', image)
        last_frame = image.copy()

        # 3. ESC 키 종료 설정 (ASCII 27 = ESC)
        if cv2.waitKey(5) & 0xFF == 27:
            print("ESC 키가 눌려 프로그램을 종료합니다.")
            break

    # 리소스 해제
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

    # 4. 이미지 저장 로직
    if last_frame is not None:
        cv2.imwrite(img_path, last_frame)
        print(f"✅ 처리가 완료되었습니다. 결과 프레임 저장: {img_path}")
    else:
        print("저장할 프레임이 존재하지 않습니다.")

if __name__ == "__main__":
    main()