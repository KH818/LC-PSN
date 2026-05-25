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

현재 서버는 FastAPI 기반 LC-PSN 추론 서버로 구성되어 있다.  
외부 클라이언트가 복소수 수신 행렬을 `.npy` 파일 형태로 서버에 전송하면, 서버는 이를 LC-PSN 모델 입력 형태로 전처리한 뒤 실제 모델 추론을 수행하고, 결과를 REST API 응답과 WebSocket을 통해 반환한다.

현재 처리 흐름은 다음과 같다.

```text
send_npy_client.py
→ DOA가 포함된 복소수 수신 행렬 [8, 200] 생성
→ sample_input.npy 파일로 저장
→ FastAPI 서버의 POST /infer-npy로 전송
→ 서버에서 [1, 16, 200] tensor로 전처리
→ LC-PSN 모델 추론 수행
→ DOAInferenceEvent 형태로 결과 변환
→ REST 응답 반환
→ WebSocket으로 웹 화면에 실시간 출력
```

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

복소수 수신 행렬이 저장된 .npy 파일을 입력으로 받아 LC-PSN 모델 추론을 수행하는 엔드포인트이다.

입력 조건:

```text
파일 형식: .npy
입력 shape: [8, 200]
입력 dtype: complex64 또는 complex128
```

동작 흐름:

```text
.npy 파일 업로드
→ complex 수신 행렬 읽기
→ shape 및 dtype 확인
→ real/imag 분리
→ 1-bit sign 양자화
→ [1, 16, 200] tensor 생성
→ LC-PSN 모델 추론
→ DOAInferenceEvent 생성
→ REST 응답 반환
→ WebSocket으로 결과 송출
```

---

## 4. 전체 동작 흐름

### 4.1 입력 데이터 생성

send_npy_client.py는 테스트용 클라이언트 역할을 한다.

* 신호원 개수 K를 랜덤으로 선택한다.
* DOA 각도를 랜덤으로 생성한다.
* physics.construct_signal()을 사용하여 실제 DOA 구조를 가진 복소수 수신 행렬을 생성한다.
* 생성된 수신 행렬은 [8, 200] 형태의 complex 행렬이다.
* 해당 행렬을 .npy 파일로 임시 저장한 뒤 FastAPI 서버로 POST 요청을 보낸다.

``` text
랜덤 K, 랜덤 DOA 생성
→ construct_signal()로 수신 행렬 생성
→ [8, 200] complex 행렬 생성
→ sample_input.npy 저장
→ POST /infer-npy 요청
```
---
### 4.2 서버 입력 처리
FastAPI 서버는 /infer-npy 엔드포인트에서 .npy 파일을 받는다.

서버는 입력 파일에 대해 다음을 확인한다.

``` text
파일 확장자가 .npy인지 확인
입력 데이터가 complex 형식인지 확인
입력 shape이 [8, 200]인지 확인
```

이후 모델 입력 형태로 변환한다.

``` text
[8, 200] complex
→ batch 차원 추가
→ [1, 8, 200] complex
→ real/imag 분리
→ [1, 16, 200]
→ 1-bit sign 양자화
→ float32 tensor 변환
```

최종 모델 입력 shape은 다음과 같다.

```text
[1, 16, 200]
```
---
### 4.3 모델 추론

서버 시작 시 LC-PSN 모델과 checkpoint를 로드한다.

```text
best_lcpsn_1bit.pt 로드
LCPSN 모델 생성
steering vector 생성
angle grid 생성
```

/infer-npy 요청이 들어오면 전처리된 tensor를 모델에 입력한다.

``` text
x: [1, 16, 200]
→ model(x, a_steering)
→ mu, K_logits 등 모델 output 생성
```

모델 output에서 다음 값을 추출한다.

```text
K_logits → 신호원 개수 K 추정
mu       → 각도별 spectrum score
mu peak  → DOA 각도 추정
```

---

### 4.4 출력 결과 생성

모델 내부 output은 그대로 프론트엔드에 전달하지 않고, API 응답용 JSON 형태로 변환한다.

현재 출력 형식은 DOAInferenceEvent 구조를 따른다.

주요 출력 정보는 다음과 같다.

```text
event_id
sensor_id
server_received_at
inference_finished_at
input metadata
model metadata
output.k_estimate
output.k_confidence
output.doa_deg
output.doa_rad
output.peak_scores
output.spectrum
diagnostics.latency_ms
```

---

### 4.5 WebSocket 실시간 출력

모델 추론 결과가 생성되면 서버는 REST 응답을 반환하는 동시에 WebSocket으로 결과를 전송한다.

```text
FastAPI 서버
→ WebSocket /ws
→ test_ws.html
→ 실시간 DOA 결과 표시
```

웹 화면에서는 주로 다음 정보를 확인한다.

```text
추정 K
추정 DOA
K confidence
입력 shape
추론 latency
```

## 6. 주요 파일 역할

### server/main.py

FastAPI 서버의 진입점이다.

주요 역할:

* FastAPI 앱 생성
* /, /health, /infer-npy, /ws 엔드포인트 정의
* 서버 시작 시 LC-PSN 모델 로드
* .npy 입력 파일 수신
* 전처리 함수 호출
* 실제 모델 추론 함수 호출
* REST 응답 반환
* WebSocket broadcast 수행

---

### server/file_preprocess.py

입력 파일을 모델 입력 tensor로 변환하는 전처리 파일이다.

주요 역할:

* .npy 파일 bytes를 읽는다.
* 입력 데이터가 complex 행렬인지 확인한다.
* 입력 shape이 [8, 200]인지 확인한다.
* complex 행렬을 real/imag로 분리한다.
* 1-bit sign 양자화를 적용한다.
* 최종적으로 [1, 16, 200] 형태의 tensor를 생성한다.

전처리 흐름:

```text
[8, 200] complex
→ [1, 8, 200] complex
→ [1, 16, 200] real/imag
→ sign quantization
→ torch.float32 tensor
```

---

### server/model_loader.py

LC-PSN 모델과 관련 설정을 로드하는 파일이다.

주요 역할:

* LCPSN 모델 객체 생성
* checkpoint 파일 로드
* 모델을 evaluation mode로 설정
* steering vector 생성
* angle grid 생성
* 모델을 서버 시작 시 한 번만 로드하도록 관리

---

### server/real_inference.py

실제 LC-PSN 모델 추론을 수행하는 파일이다.

주요 역할:

* 전처리된 입력 tensor를 모델에 전달
* model(x, a_steering) 실행
* 모델 output을 API 응답 형식으로 변환하기 위해 adapter 호출
* 최종 DOAInferenceEvent 반환

---

### server/api_adapter.py

LC-PSN 모델 내부 output을 API 응답용 JSON 형태로 변환하는 파일이다.

주요 역할:

* K_logits에서 추정 신호원 개수 계산
* softmax(K_logits)로 K confidence 계산
* mu spectrum에서 peak를 선택하여 DOA 추정
* DOA 값을 radian과 degree 단위로 변환
* peak score 추출
* spectrum 전체 값을 응답에 포함
* timestamp, input metadata, model metadata, diagnostics 정보를 조합
* 최종 DOAInferenceEvent 생성

---

### server/websocket_manager.py

WebSocket 연결을 관리하는 파일이다.

주요 역할:

* WebSocket 클라이언트 연결 등록
* 연결 종료 시 클라이언트 제거
* 추론 결과를 연결된 모든 클라이언트에 broadcast

---

### send_npy_client.py

FastAPI 서버에 입력 데이터를 자동으로 보내는 테스트용 클라이언트이다.

주요 역할:

* 랜덤 K와 랜덤 DOA 생성
* physics.construct_signal()을 사용하여 DOA가 포함된 복소수 수신 행렬 생성
* 생성된 행렬을 .npy 파일로 저장
* /infer-npy 엔드포인트로 POST 요청 전송
* true DOA와 모델 예측 결과를 터미널에 출력

터미널 출력 예시:

```text
true_k: 2
true_doa_deg: [-19.76, 65.1]
pred_k: 2
pred_doa_deg: [-19.30, 65.43]
k_confidence: 0.93
```

여기서 true_doa_deg는 client가 신호 생성에 사용한 실제 DOA이고, pred_doa_deg는 LC-PSN 모델이 추정한 DOA이다.

---

### test_ws.html

WebSocket 결과를 확인하기 위한 테스트용 웹 페이지이다.

주요 역할:

* FastAPI 서버의 /ws에 WebSocket 연결
* 서버에서 전송되는 doa_inference_event 수신
* 추정 K, DOA, confidence, latency 등을 화면에 표시

---

### model.py

LC-PSN 모델 구조가 정의된 파일이다.

주요 역할:

* LCPSN 클래스 정의
* 입력 tensor [B, 2M, T]를 받아 spectrum과 K 추정 결과 출력
* mu, sigma2, K_logits, R_ref, S_base 등의 내부 output 생성

---

### physics.py

DOA 시뮬레이션 신호를 생성하는 파일이다.

주요 역할:

* ULA steering vector 생성
* 지정된 DOA 각도에 따른 수신 신호 행렬 생성
* send_npy_client.py에서 테스트용 실제 DOA 신호를 만들 때 사용

---

### criterion.py

각도 grid 생성 및 후처리에 필요한 함수들이 포함된 파일이다.

주요 역할:

* angle grid 생성
* spectrum peak 후처리 관련 함수 제공
* 모델 output에서 DOA를 해석할 때 필요한 보조 함수 제공
