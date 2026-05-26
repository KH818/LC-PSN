function PolarChart({ inference }) {
  const doas = inference?.doas ?? [];
  const confidence = inference?.confidence;
  const width = 320;
  const height = 220;
  const cx = width / 2;
  const cy = 184;
  const radius = 132;
  const showLabels = doas.length <= 3;

  // -90~90도 DOA 값을 반원 좌표계의 SVG 좌표로 변환한다.
  const toPoint = (angleDeg, distance = radius) => {
    const clamped = Math.max(-90, Math.min(90, Number(angleDeg) || 0));
    const radians = ((clamped - 90) * Math.PI) / 180;

    return {
      x: cx + Math.cos(radians) * distance,
      y: cy + Math.sin(radians) * distance,
    };
  };

  const gridRadii = [44, 88, 132];
  const angleTicks = [-90, -60, -30, 0, 30, 60, 90];

  return (
    <div className="chart-card chart-card-large">
      <div className="chart-header">
        <div>
          <h2>DOA Polar</h2>
          <p>{doas.length > 0 ? doas.map((angle) => `${Math.round(angle)} deg`).join(" / ") : "Waiting for angles"}</p>
        </div>
        <span className="metric-pill">{doas.length} sources</span>
      </div>

      <div className="polar-frame">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="DOA polar chart">
          {gridRadii.map((gridRadius) => (
            <path
              key={gridRadius}
              className="chart-grid"
              d={`M ${cx - gridRadius} ${cy} A ${gridRadius} ${gridRadius} 0 0 1 ${
                cx + gridRadius
              } ${cy}`}
            />
          ))}

          {angleTicks.map((angle) => {
            const end = toPoint(angle);
            const label = toPoint(angle, radius + 18);

            return (
              <g key={angle}>
                <line className="chart-grid" x1={cx} y1={cy} x2={end.x} y2={end.y} />
                <text className="chart-axis-label" x={label.x} y={label.y}>
                  {angle}
                </text>
              </g>
            );
          })}

          <line className="polar-baseline" x1={cx - radius} y1={cy} x2={cx + radius} y2={cy} />

          {doas.map((angle, index) => {
            const point = toPoint(angle, radius * 0.92);

            return (
              <g key={`${angle}-${index}`}>
                <line className="doa-ray" x1={cx} y1={cy} x2={point.x} y2={point.y} />
                <circle className="doa-dot" cx={point.x} cy={point.y} r="6" />
                {/* source가 많을 때 각도 라벨이 서로 겹치므로 소수일 때만 표시한다. */}
                {showLabels && (
                  <text className="doa-label" x={point.x} y={point.y - 12}>
                    {Math.round(angle)} deg
                  </text>
                )}
              </g>
            );
          })}

          {doas.length === 0 && (
            <text className="empty-chart-label" x={cx} y={96}>
              Waiting for DOA data
            </text>
          )}
        </svg>
      </div>

      <div className="chart-footer">
        <span>Confidence</span>
        <strong>{Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : "-"}</strong>
      </div>
    </div>
  );
}

export default PolarChart;
