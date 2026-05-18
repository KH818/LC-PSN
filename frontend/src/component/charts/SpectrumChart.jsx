import { getRange, normalizeNumberList, scaleValue } from "./chartUtils";

function SpectrumChart({ inference }) {
  const spectrumMeta = inference?.spectrum ?? {};
  const spectrum = normalizeNumberList(spectrumMeta.values);
  const width = 560;
  const height = 260;
  const padding = { top: 22, right: 22, bottom: 34, left: 44 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const { min, max } = getRange(spectrum);
  const points = spectrum.map((value, index) => {
    const x =
      padding.left + (spectrum.length === 1 ? plotWidth / 2 : (index / (spectrum.length - 1)) * plotWidth);
    const y = padding.top + plotHeight - scaleValue(value, min, max, plotHeight);

    return `${x},${y}`;
  });

  return (
    <div className="chart-card">
      <div className="chart-header">
        <div>
          <h2>Spectrum</h2>
          <p>
            {spectrum.length > 0
              ? `${spectrumMeta.gridStart ?? -90} deg to ${spectrumMeta.gridEnd ?? 90} deg`
              : "Waiting for spectrum"}
          </p>
        </div>
        <span className="metric-pill">{spectrum.length} bins</span>
      </div>

      <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Spectrum line chart">
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
          const y = padding.top + plotHeight * tick;
          const value = max - (max - min) * tick;

          return (
            <g key={tick}>
              <line className="chart-grid" x1={padding.left} y1={y} x2={width - padding.right} y2={y} />
              <text className="chart-y-label" x={padding.left - 10} y={y + 4}>
                {value.toFixed(2)}
              </text>
            </g>
          );
        })}

        <line
          className="chart-axis"
          x1={padding.left}
          y1={height - padding.bottom}
          x2={width - padding.right}
          y2={height - padding.bottom}
        />
        <line
          className="chart-axis"
          x1={padding.left}
          y1={padding.top}
          x2={padding.left}
          y2={height - padding.bottom}
        />

        {points.length > 0 ? (
          <>
            <polyline
              className="spectrum-area"
              points={`${padding.left},${height - padding.bottom} ${points.join(" ")} ${
                width - padding.right
              },${height - padding.bottom}`}
            />
            <polyline className="spectrum-line" points={points.join(" ")} />
          </>
        ) : (
          <text className="empty-chart-label" x={width / 2} y={height / 2}>
            Waiting for spectrum data
          </text>
        )}
      </svg>
    </div>
  );
}

export default SpectrumChart;
