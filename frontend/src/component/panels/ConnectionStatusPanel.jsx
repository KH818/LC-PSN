function ConnectionStatusPanel({ apiBaseUrl, wsUrl, status, error }) {
  return (
    <section className="panel">
      <h2>Connection Status</h2>

      <p>
        <strong>API:</strong> {apiBaseUrl}
      </p>

      <p>
        <strong>WebSocket:</strong> {wsUrl}
      </p>

      <p>
        <strong>Status:</strong> {status}
      </p>

      {error && <p className="error">{error}</p>}
    </section>
  );
}

export default ConnectionStatusPanel;