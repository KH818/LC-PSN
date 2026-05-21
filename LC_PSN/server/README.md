# LC-PSN FastAPI Mock Inference Server

## 1. 개요

본 서버는 LC-PSN / TransMUSIC 기반 DOA 추정 모델을 실제 서비스 형태로 사용하기 위한 FastAPI 기반 추론 서버의 초기 뼈대이다.

1주차 구현 범위에서는 실제 PyTorch 모델을 연결하지 않고, Mock 추론 결과를 생성하여 다음 흐름을 검증한다.

클라이언트 요청
→ FastAPI 서버 수신
→ Mock DOA 추론 결과 생성
→ REST API 응답 반환
→ WebSocket으로 실시간 결과 송출

## 2. 서버 실행 방법
서버 실행 명령어: python -m uvicorn server.main:app --reload    
서버 기본 주소: http://127.0.0.1:8000 -> 서버 실행 여부 확인    
Health check: http://127.0.0.1:8000/health -> 서버 상태 확인     
API 문서 테스트: http://127.0.0.1:8000/docs -> Swagger UI에서 API 테스트    
Websocket 테스트: ws://127.0.0.1:8000/ws -> 실시간 결과 수신용 웹소켓 주소    
npy 파일 생성, POST 요청 실행 명령어: python send_npy_client.py -> 임시 .npy 파일([8, 200] 수신 행렬 하나) 생성 후 서버로 POST 요청 

## 3. 현재까지 구현한 내용

현재 서버는 `.h5` 파일을 직접 읽는 방식이 아니라, 외부에서 `.npy` 형태의 복소수 수신 행렬을 POST 요청으로 보내는 구조로 구성되어 있다.

즉, 실제 수신 모듈이나 시뮬레이터가 하나의 수신 행렬을 서버에 보내면, FastAPI 서버가 이를 받아 LC-PSN 모델 입력 형태로 전처리하고, 현재는 Mock DOA 결과를 반환한다.

---

### `GET /`

서버 기본 접속 확인용 엔드포인트이다.

- FastAPI 서버가 실행 중인지 간단히 확인할 수 있다.
- 브라우저에서 `http://127.0.0.1:8000/`로 접속하면 서버 상태 메시지를 반환한다.

---

### `GET /health`

서버 상태 확인용 엔드포인트이다.

- 서버가 정상적으로 동작 중인지 확인할 수 있다.
- 서버 실행 테스트나 상태 점검에 사용한다.

---

### `POST /infer-npy`

복소수 수신 행렬이 저장된 `.npy` 파일을 입력으로 받아 전처리하는 엔드포인트이다.

- `.npy` 파일을 업로드한다.
- 파일 안의 복소수 행렬을 읽는다.
- 입력 행렬의 shape이 `[8, 200]`인지 확인한다.
- 입력 데이터가 complex 형식인지 확인한다.
- complex 행렬을 real/imag 성분으로 분리한다.
- 1-bit sign 양자화를 적용한다.
- LC-PSN 모델 입력 형태인 `[1, 16, 200]` tensor로 변환한다.
- 현재는 실제 모델 대신 Mock DOA 결과를 반환한다.
- 반환 결과는 WebSocket으로도 전송된다.

동작 흐름:

```text
.npy 파일 업로드
→ [8, 200] complex 수신 행렬 읽기
→ real/imag 분리
→ 1-bit sign 양자화
→ [1, 16, 200] tensor 생성
→ Mock DOA 결과 생성
→ REST 응답 반환
→ WebSocket으로 결과 송출
