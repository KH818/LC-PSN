function WaterfallChart({ data }) {
  return (
    <div className="chart-card">
      <h2>Waterfall Chart</h2>
      <p>Realtime history visualization area</p>

      {data && (
        <p>
          <strong>Confidence:</strong> {data.confidence}
        </p>
      )}
    </div>
  );
}

export default WaterfallChart;