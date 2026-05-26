# DOA_DB

LC-PSN 기반 DOA 추정 시스템을 위한 DB 저장 모듈입니다.

본 프로젝트는 모델 서버에서 수신한 raw IQ signal 데이터를 HDF5 포맷으로 저장하고,  
LC-PSN 모델이 추론한 결과(DOA angle, estimated K, confidence, latency 등)를  
InfluxDB에 저장 및 조회할 수 있도록 구성되어 있습니다.

---

# Overview

본 시스템은 다음 기능을 제공합니다.

- Raw IQ signal 저장 (HDF5)
- Spectrum 저장 (HDF5)
- 추론 결과 저장 (InfluxDB)
<<<<<<< HEAD
- 최근 추론 결과 조회
- 특정 signal id 기반 결과 조회
- 시간 범위 기반 결과 조회
- 장기 이력 분석용 요약 조회
=======
- 최근 추론 결과 조회(1개/여러개 선택 가능)
- 특정 signal 기반 결과 조회(id를 이용해 검색)
- 추론 평균 조회 (k_estimate / snr_estimate / confidence 평균)
>>>>>>> 65ae9770e225d8134a7fe2c8345de5b34f0c8031

현재 구조에서는 raw signal을 API 응답으로 직접 반환하지 않고,  
`id`를 기준으로 metadata와 저장 경로를 조회한 뒤 storage에서 raw 데이터를 확인하는 방식으로 사용합니다.

---

# Database

## InfluxDB

본 프로젝트에서는 InfluxDB를 사용합니다.

InfluxDB는 Time-Series Database(TSDB)로,  
시간(timestamp) 기반 데이터 저장 및 조회에 최적화되어 있습니다.

DOA 추론 시스템에서는 다음과 같은 시간 기반 데이터가 지속적으로 생성됩니다.

- timestamp
- DOA angle
- estimated K
- confidence
- latency
- spectrum metadata

따라서 일반적인 RDB(MySQL 등)보다  
InfluxDB가 실시간 조회 및 장기 이력 분석에 더 적합합니다.

---

# ID Structure

본 프로젝트에서는 raw signal과 inference result를  
하나의 `id` 기준으로 통합 관리합니다.

모델 서버가 `/infer-npy` 요청을 받을 때 UUID 기반 `raw_data_id`를 생성하며,  
이 `id`를 기반으로 다음 데이터들이 연결됩니다.

- raw IQ signal (.h5)
- spectrum (.h5)
- inference result
- raw metadata

즉, 하나의 signal에 대한 모든 데이터를  
동일한 `id`로 조회할 수 있습니다.

---

# Stored Data

## 1. Raw Signal Metadata

measurement: `raw_signal`

저장 정보:

- id
- sensor_id
- file_path
- raw_format
- antenna_count
- snapshot_count
- sample_rate
- center_frequency
- description
- input_timestamp

Raw IQ signal 자체는 `.h5` 파일로 저장되며,  
InfluxDB에는 메타데이터만 저장됩니다.

보안 및 용량 문제로 인해  
raw signal 데이터 자체는 API 응답으로 반환하지 않습니다.

대신 저장된 파일 경로(`file_path`)만 제공합니다.

---

## 2. Inference Result

measurement: `doa_inference`

저장 정보:

- id
- raw_format
- k_estimate
- doa
- confidence
- latency_ms
- snr_estimate
- spectrum_path
- input_timestamp
- output_timestamp

Inference 결과는 raw signal과 동일한 `id`로 연결됩니다.

---

# HDF5

Raw IQ signal 데이터는 HDF5 포맷으로 저장됩니다.

HDF5는 대용량 배열 데이터를 효율적으로 저장할 수 있는 포맷으로,  
추후 재학습(retraining) 데이터셋 구축에도 활용 가능합니다.

Spectrum 데이터 역시 별도의 `.h5` 파일로 저장됩니다.

예시:

```text
raw_storage/
 └─ raw_a1b2c3d4e5f6.h5

spectrum_storage/
 └─ raw_a1b2c3d4e5f6_spectrum.h5
```

---

# Project Structure

```text
DOA_DB/
├─ raw_storage/
├─ spectrum_storage/
├─ docker-compose.yml
├─ requirements.txt
├─ hdf5_storage.py
├─ influx_client.py
└─ main.py
```

---

# Main Files

## hdf5_storage.py

HDF5 파일 저장을 담당합니다.

주요 기능:

- `.npy` complex raw signal을 HDF5로 저장
- 모델 output spectrum을 HDF5로 저장

현재 구조에서는 별도의 h5 파일 업로드 기능은 사용하지 않습니다.

---

## influx_client.py

InfluxDB 저장 및 조회를 담당합니다.

주요 기능:

- raw signal metadata 저장
- inference result 저장
- 모델 event 기반 inference result 저장
- 최근 inference result 조회
- 특정 id 기반 result 조회
- 시간 범위 기반 result 조회
- 최근 raw signal metadata 조회
- history summary 조회

---

# Query Usage

InfluxDB에 저장된 데이터는 다음 기준으로 조회할 수 있습니다.

- 최근 추론 결과
- 특정 id의 추론 결과
- 최근 raw signal metadata
- 시간 범위 기반 추론 결과
- 최근 n시간 요약 통계

Raw signal 자체는 API 응답으로 직접 반환하지 않고,  
조회된 `file_path`를 통해 storage에서 확인합니다.

---

# Notes

- 현재 구조에서는 h5 파일 업로드 기능을 사용하지 않습니다.
- 모델 서버에서 `.npy` 입력을 받은 뒤 HDF5로 변환 저장합니다.
- raw signal과 inference result는 동일한 `id`로 연결됩니다.
- InfluxDB에는 raw signal 자체가 아니라 metadata만 저장합니다.
- Spectrum은 별도의 HDF5 파일로 저장하고, InfluxDB에는 `spectrum_path`만 저장합니다.