# LC-PSN Integrated Server

LC-PSN 모델 추론 서버, DB 저장 모듈, React UI를 하나로 합친 통합 프로젝트입니다.

## 구조

```text
LC_PSN_INTEGRATED/
├─ server/
│  ├─ main.py
│  ├─ hdf5_storage.py
│  ├─ influx_client.py
│  ├─ model_loader.py
│  ├─ real_inference.py
│  ├─ api_adapter.py
│  ├─ file_preprocess.py
│  └─ websocket_manager.py
├─ frontend/
├─ raw_storage/
├─ spectrum_storage/
├─ model.py
├─ physics.py
├─ criterion.py
├─ dataset.py
├─ best_lcpsn_1bit.pt
├─ send_npy_client.py
├─ docker-compose.yml
└─ requirements.txt
└─ LC-PSN DOA Estimation System Structure.md
```

## 실행 순서

# 전체 실행 순서 요약

PC에서 처음 실행할 때 필요한 순서는 다음과 같다.

```text
1. Python 설치
2. Node.js 설치
3. Docker Desktop 설치
4. 프로젝트 다운로드 또는 git clone
5. pip install -r requirements.txt
6. docker compose up
7. uvicorn server.main:app --reload
8. cd frontend
9. npm install
10. npm run dev
11. python send_npy_client.py
```

# 실행 방법

## 1. Python requirements 설치

먼저 프로젝트 루트 폴더에서 필요한 Python 패키지를 설치한다.

```cmd
pip install -r requirements.txt
```

이미 설치된 패키지는 자동으로 건너뛰고, 설치되지 않은 패키지만 추가로 설치된다.

---

## 3. docker, InfluxDB 실행

docker desktop을 설치 후 실행하고,
프로젝트 루트 폴더에서 다음 명령어를 실행한다.

```cmd
docker compose up -d
```

정상 실행되면 InfluxDB는 다음 주소에서 실행된다.

```text
http://localhost:8086
```

Node.js 버전을 변경했거나 frontend 실행 오류 발생 시:
```text
rmdir /s /q node_modules
del package-lock.json
npm install
```
---

## 4. InfluxDB 로그인 방법

브라우저에서 다음 주소로 접속한다.

```text
http://localhost:8086
```

로그인 정보는 `docker-compose.yml`에 정의된 값을 사용한다.

확인할 항목은 다음과 같다.

```yaml
DOCKER_INFLUXDB_INIT_USERNAME
DOCKER_INFLUXDB_INIT_PASSWORD
DOCKER_INFLUXDB_INIT_ORG
DOCKER_INFLUXDB_INIT_BUCKET
DOCKER_INFLUXDB_INIT_ADMIN_TOKEN
```

예시:

```text
username: docker-compose.yml의 DOCKER_INFLUXDB_INIT_USERNAME 값
password: docker-compose.yml의 DOCKER_INFLUXDB_INIT_PASSWORD 값
org: doa-lab
bucket: doa_results
token: doa-secret-token
```

주의할 점은 `influx_client.py`에 있는 값과 `docker-compose.yml`의 값이 같아야 한다는 것이다.

```python
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "doa-secret-token"
INFLUX_ORG = "doa-lab"
INFLUX_BUCKET = "doa_results"
```

---

## 5. FastAPI 서버 실행

새 CMD 창을 열고 프로젝트 루트 폴더에서 다음 명령어를 실행한다.

```cmd
uvicorn server.main:app --reload
```

정상 실행되면 다음 주소에서 서버가 동작한다.

```text
http://127.0.0.1:8000
```

서버 상태 확인:

```text
http://127.0.0.1:8000/health
```

정상 응답:

```json
{"status":"healthy"}
```

---

## 6. Frontend 실행

새 CMD 창을 열고 `frontend` 폴더로 이동한다.

```cmd
cd frontend
```

처음 실행하는 경우 npm 패키지를 설치한다.

```cmd
npm install
```

설치 후 실행한다.

```cmd
npm run dev
```

정상 실행되면 브라우저에서 표시되는 주소로 접속한다.

보통 다음 주소이다.

```text
http://localhost:5173
```

---

## 7. 자동 입력 데이터 송신 실행

새 CMD 창을 열고 프로젝트 루트 폴더에서 다음 명령어를 실행한다.

```cmd
python send_npy_client.py
```

이 코드는 3초마다 다음 작업을 수행한다.

```text
1. 랜덤 DOA 입력 데이터 생성
2. .npy 파일 생성
3. FastAPI 서버의 /infer-npy로 POST 요청
4. 모델 추론 결과 수신
5. UI와 DB 저장 결과 확인
```

---

각 명령어는 가능하면 서로 다른 CMD 창에서 실행한다.

```text
CMD 1: docker compose up
CMD 2: uvicorn server.main:app --reload
CMD 3: frontend 실행
CMD 4: send_npy_client.py 실행
```

---

## Python

FastAPI 서버, 모델 추론 코드, DB 저장 코드, 자동 POST 클라이언트를 실행하기 위해 필요하다.

## Node.js

React/Vite 기반 frontend를 실행하기 위해 필요하다.

Vite 실행을 위해 Node.js 버전은 다음 조건을 만족해야 한다.

```text
Node.js 20.19 이상 또는 22.12 이상
```

권장 버전:

```text
Node.js 22 LTS
```

## Docker Desktop

Docker를 사용하면 InfluxDB를 직접 설치하지 않아도 된다.  
즉, 팀원 각자의 PC에서 복잡한 DB 설치 과정 없이 동일한 InfluxDB 환경을 실행할 수 있다.

Docker가 담당하는 역할은 다음과 같다.

```text
- InfluxDB 서버 실행
- org 생성
- bucket 생성
- token 설정
- 8086 포트로 DB 서버 제공
```

따라서 본 프로젝트에서 Docker는 주로 **InfluxDB 실행용**으로 사용된다.

# 저장 확인 방법

## HDF5 raw signal 확인

자동 입력을 실행하면 다음 폴더에 raw signal 파일이 생성된다.

```text
raw_storage/
```

예시:

```text
raw_storage/raw_a1b2c3d4e5f6.h5
```

## HDF5 spectrum 확인

모델 추론 후 spectrum 파일은 다음 폴더에 저장된다.

```text
spectrum_storage/
```

예시:

```text
spectrum_storage/raw_a1b2c3d4e5f6_spectrum.h5
```

## InfluxDB 저장 확인

InfluxDB 웹 화면에서 다음 bucket을 선택한다.

```text
doa_results
```

확인할 measurement는 다음 두 가지이다.

```text
raw_signal
doa_inference
```

`raw_signal`에는 raw signal의 metadata와 file path가 저장된다.  
`doa_inference`에는 모델 추론 결과가 저장된다.

### 5. 테스트 입력 송신

새 터미널에서 프로젝트 루트로 이동 후:

```bash
python send_npy_client.py
```

이 파일은 3초마다 `[8, 200] complex64` raw IQ signal을 생성해서 `POST /infer-npy`로 전송합니다.

## 주요 API

### Health check

```text
GET /health
```

### 추론 요청

```text
POST /infer-npy
```

입력:

```text
file: .npy
sensor_id: ula_01
```

동작:

```text
.npy 수신
→ raw_data_id 생성
→ raw signal HDF5 저장
→ raw metadata InfluxDB 저장
→ 모델 입력 전처리
→ LC-PSN 추론
→ spectrum HDF5 저장
→ inference result InfluxDB 저장
→ WebSocket broadcast
→ REST response 반환
```

### 조회 API

```text
GET /raw-signals/recent
GET /inference-results/latest
GET /inference-results/recent
GET /signals/{raw_data_id}/results
GET /inference-results/time-range
GET /history/summary
GET /inference-results/doa-range
GET /inference-results/min-k
```

## 저장 구조

### HDF5

Raw signal:

```text
raw_storage/{raw_data_id}.h5
dataset: X
```

Spectrum:

```text
spectrum_storage/{raw_data_id}_spectrum.h5
dataset: spectrum
```

### InfluxDB

Raw metadata:

```text
measurement: raw_signal
```

Inference result:

```text
measurement: doa_inference
```

공통 조회 기준:

```text
id = raw_data_id
```

## UI 연결

Frontend는 다음 주소를 사용합니다.

```text
VITE_API_BASE_URL=/api
VITE_WS_URL=ws://127.0.0.1:8000/ws
```

Vite 개발 서버는 `/api` 요청을 `http://127.0.0.1:8000`으로 proxy합니다.
