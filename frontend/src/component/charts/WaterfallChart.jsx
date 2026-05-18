import { normalizeInferenceEvent } from "../../utils/inferenceEvent";
import { getRange, normalizeNumberList } from "./chartUtils";

function WaterfallChart({ inference, messageHistory = [] }) {
  const maxRows = 36;
  const history = messageHistory
    .map((message) => normalizeNumberList(normalizeInferenceEvent(message).spectrum.values))
    .filter((spectrum) => spectrum.length > 0)
    .slice(-maxRows);
  const latestSpectrum = normalizeNumberList(inference?.spectrum?.values);
  const maxColumns = Math.max(1, ...history.map((row) => row.length), latestSpectrum.length);
  const allValues = history.flat();
  const { min, max } = getRange(allValues);

  const getHeatColor = (value) => {
    const ratio = max === min ? 0.5 : (value - min) / (max - min);
    const hue = 220 - ratio * 180;
    const lightness = 18 + ratio * 42;

    return `hsl(${hue}, 78%, ${lightness}%)`;
  };

  return (
    <div className="chart-card chart-card-wide">
      <div className="chart-header">
        <div>
          <h2>Waterfall</h2>
          <p>Recent spectrum history</p>
        </div>
        <span className="metric-pill">{history.length} frames</span>
      </div>

      {history.length > 0 ? (
        <div
          className="waterfall-grid"
          style={{
            gridTemplateColumns: `repeat(${maxColumns}, minmax(3px, 1fr))`,
            gridTemplateRows: `repeat(${maxRows}, 1fr)`,
          }}
          role="img"
          aria-label="Waterfall spectrum history heatmap"
        >
          {Array.from({ length: maxRows }).map((_, rowIndex) => {
            const row = history[rowIndex - (maxRows - history.length)] ?? [];

            return Array.from({ length: maxColumns }).map((__, columnIndex) => {
              const value = row[columnIndex];

              return (
                <span
                  key={`${rowIndex}-${columnIndex}`}
                  className="waterfall-cell"
                  style={{ backgroundColor: Number.isFinite(value) ? getHeatColor(value) : "#edf2f7" }}
                />
              );
            });
          })}
        </div>
      ) : (
        <div className="empty-panel">Waiting for spectrum history</div>
      )}

      <div className="chart-footer">
        <span>Latest confidence</span>
        <strong>{inference?.confidence == null ? "-" : `${Math.round(Number(inference.confidence) * 100)}%`}</strong>
      </div>
    </div>
  );
}

export default WaterfallChart;
