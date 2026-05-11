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

write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()


def save_raw_metadata(
    raw_id: str,
    file_path: str,
    antenna_count: int,
    snapshot_count: int,
    sample_rate: Optional[float] = None,
    center_frequency: Optional[float] = None,
    description: Optional[str] = None,
):
    point = (
        Point("raw_signal")
        .tag("raw_id", raw_id)
        .field("file_path", file_path)
        .field("antenna_count", int(antenna_count))
        .field("snapshot_count", int(snapshot_count))
        .time(datetime.now(timezone.utc))
    )

    if sample_rate is not None:
        point = point.field("sample_rate", float(sample_rate))

    if center_frequency is not None:
        point = point.field("center_frequency", float(center_frequency))

    if description is not None:
        point = point.field("description", description)

    write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


def save_inference_result(
    raw_id: str,
    result_id: str,
    estimated_k: int,
    doa_angles_deg: List[float],
    snr_estimate: Optional[float] = None,
    confidence: Optional[float] = None,
):
    point = (
        Point("doa_inference")
        .tag("raw_id", raw_id)
        .tag("result_id", result_id)
        .field("estimated_k", int(estimated_k))
        .field("doa_angles_deg", ",".join(map(str, doa_angles_deg)))
        .time(datetime.now(timezone.utc))
    )

    if snr_estimate is not None:
        point = point.field("snr_estimate", float(snr_estimate))

    if confidence is not None:
        point = point.field("confidence", float(confidence))

    write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)


def get_latest_inference_result():
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -24h)
      |> filter(fn: (r) => r._measurement == "doa_inference")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: 1)
    '''

    tables = query_api.query(query, org=INFLUX_ORG)

    for table in tables:
        for record in table.records:
            v = record.values
            return {
                "time": str(v.get("_time")),
                "raw_id": v.get("raw_id"),
                "result_id": v.get("result_id"),
                "estimated_k": v.get("estimated_k"),
                "doa_angles_deg": v.get("doa_angles_deg"),
                "snr_estimate": v.get("snr_estimate"),
                "confidence": v.get("confidence"),
            }

    return None


def get_recent_inference_results(limit: int = 20):
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r._measurement == "doa_inference")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: {limit})
    '''

    tables = query_api.query(query, org=INFLUX_ORG)

    results = []

    for table in tables:
        for record in table.records:
            v = record.values
            results.append({
                "time": str(v.get("_time")),
                "raw_id": v.get("raw_id"),
                "result_id": v.get("result_id"),
                "estimated_k": v.get("estimated_k"),
                "doa_angles_deg": v.get("doa_angles_deg"),
                "snr_estimate": v.get("snr_estimate"),
                "confidence": v.get("confidence"),
            })

    return results


def get_results_by_raw_id(raw_id: str):
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r._measurement == "doa_inference")
      |> filter(fn: (r) => r.raw_id == "{raw_id}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: true)
    '''

    tables = query_api.query(query, org=INFLUX_ORG)

    results = []

    for table in tables:
        for record in table.records:
            v = record.values
            results.append({
                "time": str(v.get("_time")),
                "raw_id": v.get("raw_id"),
                "result_id": v.get("result_id"),
                "estimated_k": v.get("estimated_k"),
                "doa_angles_deg": v.get("doa_angles_deg"),
                "snr_estimate": v.get("snr_estimate"),
                "confidence": v.get("confidence"),
            })

    return results


def get_raw_signals(limit: int = 20):
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r._measurement == "raw_signal")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> sort(columns: ["_time"], desc: true)
      |> limit(n: {limit})
    '''

    tables = query_api.query(query, org=INFLUX_ORG)

    results = []

    for table in tables:
        for record in table.records:
            v = record.values
            results.append({
                "time": str(v.get("_time")),
                "raw_id": v.get("raw_id"),
                "file_path": v.get("file_path"),
                "antenna_count": v.get("antenna_count"),
                "snapshot_count": v.get("snapshot_count"),
                "sample_rate": v.get("sample_rate"),
                "center_frequency": v.get("center_frequency"),
                "description": v.get("description"),
            })

    return results


def get_history_summary(hours: int = 24):
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{hours}h)
      |> filter(fn: (r) => r._measurement == "doa_inference")
      |> filter(fn: (r) => r._field == "estimated_k" or r._field == "snr_estimate" or r._field == "confidence")
      |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
    '''

    tables = query_api.query(query, org=INFLUX_ORG)

    results = []

    for table in tables:
        for record in table.records:
            v = record.values
            results.append({
                "time": str(v.get("_time")),
                "field": v.get("_field"),
                "mean_value": v.get("_value"),
            })

    return results