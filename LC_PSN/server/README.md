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

## 3. 현재까지 구현한 내용
### `GET /`

서버 기본 접속 확인용 엔드포인트이다.

- 서버가 정상적으로 실행 중인지 간단히 확인할 수 있다.
- 브라우저에서 `http://127.0.0.1:8000/`로 접속하면 서버 상태 메시지를 반환한다.

---

### `GET /health`

서버 상태 확인용 엔드포인트이다.

- FastAPI 서버가 정상적으로 동작 중인지 확인할 수 있다.
- 정상 동작 시 `healthy` 상태를 반환한다.
- 서버 실행 테스트나 상태 점검에 사용한다.

---

### `POST /inspect-h5`

업로드한 `.h5` 파일의 내부 구조를 확인하는 엔드포인트이다.

- `.h5` 파일을 업로드하면 내부 dataset 목록을 확인한다.
- `X`, `Y` dataset이 존재하는지 확인한다.
- 각 dataset의 shape과 dtype을 반환한다.
- 모델 입력으로 사용할 수 있는 파일인지 점검하는 용도로 사용한다.

확인 가능한 정보:

```text
filename
keys
X_shape
X_dtype
Y_shape
Y_dtype
```

---

### `POST / infer-file`
업로드한 .h5 파일을 모델 입력 형태로 전처리하는 엔드포인트이다.

.h5 파일을 업로드한다.
파일 내부의 첫 번째 샘플 X[0]을 읽는다.
complex 수신 신호를 real/imag로 분리한다.
1-bit sign 양자화를 적용한다.
LC-PSN 모델 입력 형태인 [1, 16, 200] tensor로 변환한다.
현재는 실제 모델 대신 Mock DOA 결과를 반환한다.
반환 결과는 WebSocket으로도 전송된다.

동작 흐름:

.h5 파일 업로드
→ X[0] 읽기
→ [1, 8, 200] complex 데이터 추출
→ real/imag 분리
→ 1-bit sign 양자화
→ [1, 16, 200] tensor 생성
→ Mock 결과 반환
→ WebSocket으로 결과 송출

---
### 자동 스트리밍 기능

서버 시작 시 train_mini_1bit.h5 파일을 미리 로드하고, 일정 시간 간격마다 샘플을 순서대로 처리한다.

서버가 시작되면 .h5 파일을 자동으로 로드한다.
X[0], X[1], X[2] ... 순서대로 샘플을 선택한다.
각 샘플을 [1, 16, 200] tensor로 변환한다.
현재는 Mock DOA 결과를 생성한다.
결과는 WebSocket 화면에 자동으로 표시된다.
Swagger에서 매번 Execute를 누르지 않아도 결과가 주기적으로 출력된다.

동작 흐름:

서버 시작
→ train_mini_1bit.h5 로드
→ 3초 마다 X[index] 선택
→ [1, 16, 200] tensor 변환
→ Mock 결과 생성
→ WebSocket으로 자동 송출
