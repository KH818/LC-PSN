# DOA_DB

LC-PSN 기반 DOA 추정 시스템을 위한 DB 서버 프로젝트입니다.

본 프로젝트는 raw IQ signal 데이터를 HDF5 포맷으로 저장하고,  
LC-PSN 모델이 추론한 결과(DOA angle, estimated K, SNR 등)를  
InfluxDB에 저장 및 조회할 수 있도록 구성되어 있습니다.

---

# Overview

본 시스템은 다음 기능을 제공합니다.

- Raw IQ signal 저장 (HDF5)
- 추론 결과 저장 (InfluxDB)
- 최근 추론 결과 조회
- 특정 signal 기반 결과 조회
- 장기 이력 분석용 데이터 조회

FastAPI 기반 REST API 서버로 구현되었으며,  
InfluxDB를 이용하여 time-series 형태의 추론 데이터를 관리합니다.

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
- SNR estimate
- confidence

따라서 일반적인 RDB(MySQL 등)보다  
InfluxDB가 실시간 조회 및 장기 이력 분석에 더 적합합니다.

---

# ID Structure

본 프로젝트에서는 raw signal과 inference result를  
하나의 `id` 기준으로 통합 관리합니다.

Raw signal 업로드 시 UUID 기반 `id`가 생성되며,  
이 `id`를 기반으로 다음 데이터들이 연결됩니다.

- raw IQ signal (.h5)
- spectrum (.h5)
- inference result
- metadata

즉, 하나의 signal에 대한 모든 데이터를  
동일한 `id`로 조회할 수 있습니다.

---

# Stored Data

## 1. Raw Signal Metadata

measurement: `raw_signal`

저장 정보:

- id
- file_path
- antenna_count
- snapshot_count
- sample_rate
- center_frequency
- description
- timestamp

Raw IQ signal 자체는 `.h5` 파일로 저장되며,  
InfluxDB에는 메타데이터만 저장됩니다.

보안 및 용량 문제로 인해  
raw signal 데이터 자체는 API 응답으로 반환하지 않습니다.

대신 저장된 파일 경로(file_path)만 제공합니다.

---

## 2. Inference Result

measurement: `doa_inference`

저장 정보:

- id
- estimated_k
- doa_angles_deg
- snr_estimate
- confidence
- spectrum_path
- timestamp

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
 └─ 550e8400-e29b-41d4-a716-446655440000.h5

spectrum_storage/
 └─ 550e8400-e29b-41d4-a716-446655440000_spectrum.h5
```
# Project structure

```text
 DOA_DB/
├─ raw_storage/
├─ spectrum_storage/
├─ docker-compose.yml
├─ requirements.txt
├─ hdf5_storage.py
├─ influx_client.py
└─ main.py
<<<<<<< HEAD
```
=======

#Notes
- 현재 구현된 코드는 최종본이 아니며, DB 공부를 통해 최적화 및 팀원 코드에 맞게 수정해 나갈 계획입니다.
>>>>>>> 2d671a006805f46af50925053673e3ad503fa71c
