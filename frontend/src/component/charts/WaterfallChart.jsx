import { normalizeInferenceEvent } from "../../utils/inferenceEvent";
import { downsampleNumberList, getRange, normalizeNumberList } from "./chartUtils";

const MAX_ROWS = 36;
const MAX_RENDER_COLUMNS = 240;

function WaterfallChart({ inference, messageHistory = [] }) {
  // WebSocket history에서 spectrum만 꺼낸 뒤 화면용 열 개수로 축소해서 heatmap에 사용한다.
  const history = messageHistory
    .map((message) =>
      downsampleNumberList(
        normalizeNumberList(normalizeInferenceEvent(message).spectrum.values),
        MAX_RENDER_COLUMNS,
      ),
    )
    .filter((spectrum) => spectrum.length > 0)
    .slice(-MAX_ROWS);
  const latestSpectrum = downsampleNumberList(normalizeNumberList(inference?.spectrum?.values), MAX_RENDER_COLUMNS);
  const maxColumns = Math.max(1, ...history.map((row) => row.length), latestSpectrum.length);
  const allValues = history.flat();
  const { min, max } = getRange(allValues);

  // 값이 클수록 따뜻한 색에 가깝게 보여서 peak 위치를 빠르게 찾을 수 있게 한다.
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
            gridTemplateRows: `repeat(${MAX_ROWS}, 1fr)`,
          }}
          role="img"
          aria-label="Waterfall spectrum history heatmap"
        >
          {Array.from({ length: MAX_ROWS }).map((_, rowIndex) => {
            const row = history[rowIndex - (MAX_ROWS - history.length)] ?? [];

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
