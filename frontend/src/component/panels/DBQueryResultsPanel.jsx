function formatDoa(doa) {
  if (!Array.isArray(doa)) {
    return "-";
  }

  return doa.map((value) => `${Number(value).toFixed(2)}°`).join(", ");
}

function formatConfidence(value) {
  if (value == null || value === "-") {
    return "-";
  }

  return `${Math.round(Number(value) * 100)}%`;
}

function DBQueryResultsPanel({ queryResults }) {
  return (
    <section className="chart-card chart-card-wide query-results-card">
      <div className="chart-header">
        <div>
          <h2>DB Query Results</h2>
          <p>Stored raw metadata and inference results from InfluxDB</p>
        </div>
        <span className="metric-pill">{queryResults.length} results</span>
      </div>

      {queryResults.length === 0 ? (
        <div className="empty-query-result">No query results yet</div>
      ) : (
        <div className="query-table-wrap">
          <table className="query-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Time</th>
                <th>Sensor</th>
                <th>K</th>
                <th>DOA</th>
                <th>Confidence</th>
                <th>Latency</th>
                <th>File / Spectrum Path</th>
              </tr>
            </thead>

            <tbody>
              {queryResults.map((item, index) => (
                <tr key={`${item.id}-${item.time}-${index}`}>
                  <td>{item.id || "-"}</td>
                  <td>{item.time || item.input_timestamp || item.output_timestamp || "-"}</td>
                  <td>{item.sensor_id || "-"}</td>
                  <td>{item.k_estimate ?? "-"}</td>
                  <td>{formatDoa(item.doa)}</td>
                  <td>{formatConfidence(item.confidence)}</td>
                  <td>
                    {item.latency_ms != null
                      ? `${Number(item.latency_ms).toFixed(2)} ms`
                      : "-"}
                  </td>
                  <td>{item.spectrum_path || item.file_path || "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export default DBQueryResultsPanel;