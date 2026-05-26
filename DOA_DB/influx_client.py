import json
from datetime import datetime, timezone
from typing import List, Optional

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS


INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "doa-secret-token"
INFLUX_ORG = "doa-lab"
INFLUX_BUCKET = "doa_results"


client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG,
)

#데이터를 쓰는 API와 조회하는 API 객체 생성
write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()


#UTC 시간으로 맞춰주는 함수
def _to_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


#Raw signal metadata 저장 함수
def save_raw_metadata(
    raw_data_id: str,
    file_path: str,
    raw_format: str,
    sensor_id: str,
    input_timestamp: datetime,
    antenna_count: int,
    snapshot_count: int,
    sample_rate: Optional[float] = None,
    center_frequency: Optional[float] = None,
    description: Optional[str] = None,
):
    input_time = _to_utc_datetime(input_timestamp)

    point = (
        Point("raw_signal")
        .tag("id", raw_data_id)
        .tag("sensor_id", sensor_id)
        .tag("raw_format", raw_format)
        .field("file_path", file_path)
        .field("antenna_count", int(antenna_count))
        .field("snapshot_count", int(snapshot_count))
        .field("input_timestamp", input_time.isoformat())
        .time(input_time)
    )

    if sample_rate is not None:
        point = point.field("sample_rate", float(sample_rate))

    if center_frequency is not None:
        point = point.field("center_frequency", float(center_frequency))

    if description is not None:
        point = point.field("description", description)

    write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


#추론 결과 저장 함수
def save_inference_result(
    raw_data_id: str,
    k_estimate: int,
    doa: List[float],
    raw_format: str,
    input_timestamp: datetime,
    output_timestamp: datetime,
    spectrum_path: Optional[str] = None,
    snr_estimate: Optional[float] = None,
    confidence: Optional[float] = None,
    latency_ms: Optional[float] = None,
):
    input_time = _to_utc_datetime(input_timestamp)
    output_time = _to_utc_datetime(output_timestamp)

    point = (
        Point("doa_inference")
        .tag("id", raw_data_id)
        .tag("raw_format", raw_format)
        .field("k_estimate", int(k_estimate))
        .field("doa", json.dumps(doa))
        .field("input_timestamp", input_time.isoformat())
        .field("output_timestamp", output_time.isoformat())
        .time(output_time)
    )

    if spectrum_path is not None:
        point = point.field("spectrum_path", spectrum_path)

    if snr_estimate is not None:
        point = point.field("snr_estimate", float(snr_estimate))

    if confidence is not None:
        point = point.field("confidence", float(confidence))

    if latency_ms is not None:
        point = point.field("latency_ms", float(latency_ms))

    write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


#모델 서버 event를 바로 InfluxDB에 저장하는 함수
def save_inference_result_from_event(
    event: dict,
    spectrum_path: Optional[str] = None,
    raw_format: str = "h5",
):
    raw_data_id = event["input"]["raw_data_id"]

    input_timestamp = datetime.fromisoformat(event["server_received_at"])
    output_timestamp = datetime.fromisoformat(event["inference_finished_at"])

    output = event["output"]
    diagnostics = event.get("diagnostics", {})

    save_inference_result(
        raw_data_id=raw_data_id,
        k_estimate=output["k_estimate"],
        doa=output["doa_deg"],
        raw_format=raw_format,
        input_timestamp=input_timestamp,
        output_timestamp=output_timestamp,
        spectrum_path=spectrum_path,
        confidence=output.get("k_confidence"),
        latency_ms=diagnostics.get("latency_ms"),
    )


#문자열로 들어온 doa를 배열로 바꿈
def _parse_doa(value):
    if value is None:
        return None

    try:
        return json.loads(value)
    except Exception:
        return value


#InfluxDB 조회 결과를 API 응답 형식으로 바꾸는 함수
def _record_to_inference_result(record):
    v = record.values

    return {
        "time": str(v.get("_time")),
        "id": v.get("id"),
        "raw_format": v.get("raw_format"),
        "input_timestamp": v.get("input_timestamp"),
        "output_timestamp": v.get("output_timestamp"),
        "k_estimate": v.get("k_estimate"),
        "doa": _parse_doa(v.get("doa")),
        "spectrum_path": v.get("spectrum_path"),
        "snr_estimate": v.get("snr_estimate"),
        "confidence": v.get("confidence"),
        "latency_ms": v.get("latency_ms"),
    }


#최근 결과 조회 함수
def get_latest_inference_result():
    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r._measurement == "doa_inference")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: 1)
    """

    tables = query_api.query(query, org=INFLUX_ORG)

    for table in tables:
        for record in table.records:
            return _record_to_inference_result(record)

    return None


#여러 결과 조회 함수
def get_recent_inference_results(limit: int = 20):
    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r._measurement == "doa_inference")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: {limit})
    """

    tables = query_api.query(query, org=INFLUX_ORG)
    results = []

    for table in tables:
        for record in table.records:
            results.append(_record_to_inference_result(record))

    return results


#특정 id 에 해당하는 추론 결과 조회 함수
def get_results_by_id(raw_data_id: str):
    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r._measurement == "doa_inference")
      |> filter(fn: (r) => r.id == "{raw_data_id}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: true)
    """

    tables = query_api.query(query, org=INFLUX_ORG)
    results = []

    for table in tables:
        for record in table.records:
            results.append(_record_to_inference_result(record))

    return results


#시간 범위로 추론 결과 조회 함수
def get_results_by_time_range(start: str = "-24h", stop: Optional[str] = None):
    stop_clause = f", stop: {stop}" if stop else ""

    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {start}{stop_clause})
      |> filter(fn: (r) => r._measurement == "doa_inference")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: true)
    """

    tables = query_api.query(query, org=INFLUX_ORG)
    results = []

    for table in tables:
        for record in table.records:
            results.append(_record_to_inference_result(record))

    return results


#최근 raw signal 메타데이터 조회 함수(raw signal 자체 X)
def get_raw_signals(limit: int = 20):
    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r._measurement == "raw_signal")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: {limit})
    """

    tables = query_api.query(query, org=INFLUX_ORG)
    results = []

    for table in tables:
        for record in table.records:
            v = record.values
            results.append(
                {
                    "time": str(v.get("_time")),
                    "id": v.get("id"),
                    "sensor_id": v.get("sensor_id"),
                    "raw_format": v.get("raw_format"),
                    "input_timestamp": v.get("input_timestamp"),
                    "file_path": v.get("file_path"),
                    "antenna_count": v.get("antenna_count"),
                    "snapshot_count": v.get("snapshot_count"),
                    "sample_rate": v.get("sample_rate"),
                    "center_frequency": v.get("center_frequency"),
                    "description": v.get("description"),
                }
            )

    return results


#최근 n 시간에 대한 요약 통계 조회
def get_history_summary(hours: int = 24):
    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{hours}h)
      |> filter(fn: (r) => r._measurement == "doa_inference")
      |> filter(fn: (r) => r._field == "k_estimate" or r._field == "snr_estimate" or r._field == "confidence" or r._field == "latency_ms")
      |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
    """

    tables = query_api.query(query, org=INFLUX_ORG)
    results = []

    for table in tables:
        for record in table.records:
            v = record.values
            results.append(
                {
                    "time": str(v.get("_time")),
                    "field": v.get("_field"),
                    "mean_value": v.get("_value"),
                }
            )

    return results