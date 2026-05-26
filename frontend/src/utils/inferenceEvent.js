function getPayload(event) {
  if (!event || typeof event !== "object") return {};

  // 현재 mock 서버는 { type, data } 구조이고, 향후 이벤트는 바로 payload가 될 수 있다.
  return event.data && typeof event.data === "object" ? event.data : event;
}

function getSpectrum(payload) {
  // 현재 mock 서버 형식: data.spectrum = number[]
  if (Array.isArray(payload?.spectrum)) {
    return {
      values: payload.spectrum,
      gridStart: -90,
      gridEnd: 90,
      gridSize: payload.spectrum.length,
    };
  }

  const spectrum = payload?.spectrum ?? payload?.output?.spectrum;

  // 졸업작품.md에서 정의한 형식: spectrum.values + grid metadata
  if (Array.isArray(spectrum?.values)) {
    return {
      values: spectrum.values,
      gridStart: Number(spectrum.grid_start ?? -90),
      gridEnd: Number(spectrum.grid_end ?? 90),
      gridSize: Number(spectrum.grid_size ?? spectrum.values.length),
    };
  }

  return {
    values: [],
    gridStart: -90,
    gridEnd: 90,
    gridSize: 0,
  };
}

export function normalizeInferenceEvent(event) {
  const payload = getPayload(event);
  const output = payload.output ?? {};
  const spectrum = getSpectrum(payload);
  // 백엔드 mock 필드명(doas_deg)과 정식 이벤트 필드명(doa_deg)을 모두 지원한다.
  const doas = payload.doa_deg ?? payload.doas_deg ?? output.doa_deg ?? [];
  const peakScores = payload.peak_scores ?? output.peak_scores ?? [];
  const confidence = payload.confidence ?? payload.k_confidence ?? output.k_confidence ?? null;

  return {
    eventId: payload.event_id ?? null,
    type: event?.type ?? payload.type ?? "inference_result",
    timestamp: payload.timestamp ?? payload.server_received_at ?? null,
    sensorId: payload.sensor_id ?? null,
    kEstimate: payload.k_estimate ?? payload.estimated_k ?? output.k_estimate ?? doas.length,
    doas: Array.isArray(doas) ? doas : [],
    peakScores: Array.isArray(peakScores) ? peakScores : [],
    confidence: Number.isFinite(Number(confidence)) ? Number(confidence) : null,
    spectrum,
    sampleIndex: payload.sample_index ?? null,
    inputFile: payload.input_file ?? payload.input?.raw_data_id ?? null,
    streaming: Boolean(payload.streaming),
  };
}
