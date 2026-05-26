# LC-PSN DOA Estimation System Structure

본 시스템은 LC-PSN 기반 DOA(Direction of Arrival) 추정 시스템으로,  
raw IQ signal 데이터를 입력받아 모델 추론을 수행하고,  
추론 결과를 HDF5 + InfluxDB에 저장하며,  
React UI를 통해 실시간 결과 및 저장된 데이터를 조회할 수 있도록 구성되어 있다.

---

# 1. 전체 시스템 구조

전체 시스템은 다음 구성요소로 이루어진다.

```text
1. send_npy_client.py
   - 테스트용 입력 데이터 생성
   - POST 요청 전송

2. FastAPI API Server
   - 입력 데이터 수신
   - id 생성
   - HDF5 저장
   - InfluxDB 저장
   - 모델 추론
   - WebSocket 송출
   - 조회 API 제공

3. LC-PSN Model
   - DOA 추론 수행

4. HDF5 Storage
   - raw IQ signal 저장
   - spectrum 저장

5. InfluxDB
   - raw metadata 저장
   - inference result 저장
   - 조회 API 데이터 소스 역할

6. React Frontend UI
   - 실시간 추론 결과 표시
   - 최근 id 조회
   - DB query 기능 제공
```

---

# 2. 전체 데이터 흐름

전체 동작 흐름은 다음과 같다.

```text
send_npy_client.py
→ POST /infer-npy
→ FastAPI server/main.py
→ id 생성
→ raw signal HDF5 저장
→ raw metadata InfluxDB 저장
→ 모델 입력 전처리
→ LC-PSN 추론
→ spectrum HDF5 저장
→ inference result InfluxDB 저장
→ WebSocket UI 송출
→ REST 응답 반환
```

---

# 3. id 구조

본 시스템에서는 하나의 raw signal 입력에 대해 하나의 통합 id를 생성한다.

예시:

```text
raw_a1b2c3d4e5f6
```

id는 모델 추론 결과에서 생성되는 것이 아니라,  
FastAPI API server가 `.npy` 입력을 수신한 직후 생성된다.

즉:

```text
입력 raw signal 자체를 기준으로 id 생성
```

구조이다.

생성된 id는 다음 데이터들을 연결하는 공통 key 역할을 한다.

```text
- raw IQ signal HDF5
- spectrum HDF5
- raw metadata
- inference result
```

즉 하나의 signal에 대한 모든 데이터는 동일한 id를 공유한다.

---

# 4. HDF5 저장 구조

## 4.1 raw signal 저장

raw IQ signal은 HDF5 파일로 저장된다.

저장 위치:

```text
raw_storage/{id}.h5
```

예시:

```text
raw_storage/raw_a1b2c3d4e5f6.h5
```

HDF5 내부에는 다음 dataset이 저장된다.

```text
X
```

즉 실제 raw IQ signal 배열이 저장된다.

예시:

```text
shape = [8, 200]
dtype = complex64
```

---

## 4.2 spectrum 저장

모델 추론 후 생성된 spectrum도 별도의 HDF5 파일로 저장된다.

저장 위치:

```text
spectrum_storage/{id}_spectrum.h5
```

예시:

```text
spectrum_storage/raw_a1b2c3d4e5f6_spectrum.h5
```

HDF5 내부에는 다음 dataset이 저장된다.

```text
spectrum
```

---

# 5. InfluxDB 저장 구조

본 시스템에서는 InfluxDB를 Time-Series Database(TSDB)로 사용한다.

InfluxDB에는 실제 raw signal 배열은 저장하지 않고,  
metadata와 inference result만 저장한다.

즉:

```text
raw signal 배열
→ HDF5 저장

metadata / inference result
→ InfluxDB 저장
```

구조이다.

---

## 5.1 raw_signal measurement

measurement 이름:

```text
raw_signal
```

저장되는 정보:

```text
id
sensor_id
file_path
antenna_count
snapshot_count
raw_format
input_timestamp
description
```

즉:

```text
raw signal 자체가 아니라
raw signal에 대한 metadata 저장
```

구조이다.

---

## 5.2 doa_inference measurement

measurement 이름:

```text
doa_inference
```

저장되는 정보:

```text
id
k_estimate
doa
confidence
latency_ms
spectrum_path
output_timestamp
snr_estimate
```

즉 LC-PSN 모델 추론 결과를 저장한다.

---

# 6. 왜 raw_signal과 doa_inference를 분리하는가?

InfluxDB에는 다음 두 measurement가 존재한다.

```text
raw_signal
doa_inference
```

둘은 물리적으로 분리되어 저장되지만,  
동일한 id를 공유한다.

예시:

```text
id = raw_a1b2c3

raw_signal
└─ raw metadata

doa_inference
└─ inference result
```

즉:

```text
물리적 저장은 분리
논리적 연결은 id로 통합
```

구조이다.

이렇게 분리하는 이유는 다음과 같다.

```text
1. raw metadata와 inference result의 역할이 다름

2. raw signal 저장은 성공했지만
   모델 추론이 실패한 상황 표현 가능

3. 나중에 하나의 raw signal에
   여러 모델 결과 연결 가능

4. metadata와 inference result를
   목적별로 따로 조회 가능
```

---

# 7. API Server 처리 흐름

`server/main.py`는 전체 시스템의 중심 역할을 수행한다.

주요 기능:

```text
1. .npy 파일 수신
2. id 생성
3. HDF5 저장
4. InfluxDB 저장
5. 모델 입력 전처리
6. LC-PSN 추론
7. spectrum 저장
8. WebSocket 송출
9. 조회 API 제공
```

---

## 7.1 raw metadata 저장 흐름

API server는 모델 추론 전에 먼저 다음 작업을 수행한다.

```text
1. id 생성
2. raw signal HDF5 저장
3. raw metadata InfluxDB 저장
```

즉:

```text
raw signal 입력 자체를 먼저 기록
```

하는 구조이다.

---

## 7.2 inference result 저장 흐름

그 후 같은 input을 LC-PSN 모델에 넣어 추론을 수행한다.

모델 추론 완료 후:

```text
1. spectrum HDF5 저장
2. inference result InfluxDB 저장
```

을 수행한다.

즉 raw metadata와 inference result는  
동일한 id를 기준으로 연결된다.

---

# 8. latency_ms 의미

현재 저장되는 `latency_ms`는:

```text
FastAPI server가 요청을 받은 시점
→ 모델 추론 결과 event 생성 완료 시점
```

까지의 처리 시간이다.

즉:

```text
모델 inference pipeline 처리 시간
```

에 해당한다.

포함되는 것:

```text
- raw signal 수신 이후 처리
- 모델 입력 전처리
- LC-PSN 추론
- postprocessing
- event 생성
```

포함되지 않는 것:

```text
- inference result InfluxDB 저장 시간
- WebSocket 송출 시간
- REST response 반환 시간
```

---

# 9. Frontend UI 구조

Frontend는 React 기반 UI로 구현되어 있다.

Frontend는 데이터를 직접 저장하지 않는다.

즉:

```text
cache 저장 구조 ❌
DB 조회 기반 구조 ✅
```

이다.

실제 데이터 저장은 다음이 담당한다.

```text
HDF5
→ raw signal / spectrum 저장

InfluxDB
→ metadata / inference result 저장
```

Frontend는 FastAPI 조회 API를 호출하여  
DB에 저장된 데이터를 화면에 표시하는 역할만 수행한다.

즉:

```text
DB 저장 → API 조회 → UI 표시
```

구조이다.

---

# 10. Recent IDs 구조

Frontend는 최근 id 목록을 보여주는 기능을 제공한다.

사용 API:

```text
GET /raw-signals/recent
```

동작 흐름:

```text
Frontend
→ GET /raw-signals/recent
→ FastAPI
→ InfluxDB raw_signal measurement 조회
→ 최근 id 목록 반환
→ UI 사이드바 표시
```

Frontend는 이 API를 5초마다 반복 호출한다.

즉 UI는 데이터를 따로 저장하지 않고,  
DB에 저장된 최신 id를 계속 조회해서 표시한다.

---

## 10.1 Recent IDs 저장 구조

InfluxDB에는 id가 계속 누적 저장된다.

예시:

```text
3초마다 inference
→ 새로운 id 생성
→ InfluxDB raw_signal에 저장
```

즉 DB에는 계속 데이터가 누적된다.

단, UI는 최근 일부(limit 기준)만 표시한다.

예시:

```text
InfluxDB 내부
→ 모든 id 저장

UI Recent IDs
→ 최근 10개만 표시
```

---

# 11. DB Query Results 구조

Frontend에는 `DB Query Results` 영역이 존재한다.

이 영역은 InfluxDB 전체 데이터를 모두 표시하는 화면이 아니라 query 결과를 보여주는 화면이다.


예를 들어 다음 기능을 제공한다.

```text
1. 최근 추론 결과 조회
2. 최신 추론 결과 조회
3. 특정 id 기반 조회
4. DOA 범위 조회
5. K 기준 조회
```

각 기능은 FastAPI 조회 API를 호출한다.

예시:

```text
GET /inference-results/recent
GET /signals/{id}/results
GET /inference-results/doa-range
GET /inference-results/min-k
```

반환된 결과만 `DB Query Results` 테이블에 표시한다.

즉:

```text
사용자가 선택한 query 결과만 화면에 표시
```

하는 구조이다.

---

# 12. DOA 범위 조회 기능

사용 API:

```text
GET /inference-results/doa-range?min_doa=-30&max_doa=30
```

이 API는 DOA 값 중 하나라도 범위 안에 포함되는 결과를 조회한다.
---

# 13. K 기준 조회 기능

사용 API:

```text
GET /inference-results/min-k?min_k=3
```

이 API는 다음 조건을 만족하는 결과를 조회한다.

```text
k_estimate >= 3
```

---

# 14. WebSocket 구조

모델 추론이 완료되면 API server는 WebSocket으로 UI에 실시간 결과를 송출한다.

흐름:

```text
LC-PSN 추론 완료
→ WebSocket broadcast
→ React UI 수신
→ 실시간 차트 업데이트
```

UI는 다음 정보를 실시간 표시한다.

```text
K estimate
DOA angles
confidence
spectrum
waterfall
event log
```

---

# 15. 최종 시스템 흐름

```text
send_npy_client.py
→ 랜덤 raw IQ signal 생성

↓

FastAPI API Server
→ id 생성
→ raw signal HDF5 저장
→ raw metadata InfluxDB 저장
→ LC-PSN 추론
→ spectrum HDF5 저장
→ inference result InfluxDB 저장
→ WebSocket 송출

↓

Frontend UI
→ 실시간 결과 표시
→ 최근 id 조회
→ DB query 결과 표시
```

---

# 16. 최종 한 줄 요약

본 시스템은 `.npy` raw IQ signal 입력에 대해 FastAPI API server가 id를 생성하고, raw signal과 spectrum은 HDF5에 저장하며, raw metadata와 inference result는 InfluxDB에 저장한다. Frontend는 데이터를 직접 저장하지 않고, 조회 API를 반복 호출하여 DB에 저장된 데이터를 실시간으로 표시하는 구조로 동작한다.