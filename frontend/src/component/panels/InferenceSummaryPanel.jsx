function formatPercent(value) {
  if (!Number.isFinite(value)) return "-";

  return `${Math.round(value * 100)}%`;
}

function formatAngles(doas) {
  if (doas.length === 0) return "-";

  return doas.map((angle) => `${Math.round(Number(angle))} deg`).join(", ");
}

function InferenceSummaryPanel({ inference }) {
  return (
    <section className="summary-panel">
      <div className="summary-tile">
        <span>K estimate</span>
        <strong>{inference.kEstimate ?? "-"}</strong>
      </div>
      <div className="summary-tile">
        <span>Confidence</span>
        <strong>{formatPercent(inference.confidence)}</strong>
      </div>
      <div className="summary-tile summary-tile-wide">
        <span>DOA angles</span>
        <strong>{formatAngles(inference.doas)}</strong>
      </div>
      <div className="summary-tile">
        <span>Spectrum bins</span>
        <strong>{inference.spectrum.values.length || "-"}</strong>
      </div>
    </section>
  );
}

export default InferenceSummaryPanel;
