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