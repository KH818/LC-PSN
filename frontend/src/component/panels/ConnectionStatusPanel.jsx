function ConnectionStatusPanel({ apiBaseUrl, wsUrl, status, error, messageCount = 0, lastMessageAt }) {
  const lastReceivedText = lastMessageAt ? new Date(lastMessageAt).toLocaleTimeString() : "No stream yet";

  return (
    <section className="panel">
      <h2>Connection Status</h2>

      <p>
        <strong>API</strong>
        <span className="panel-value">{apiBaseUrl}</span>
      </p>

      <p>
        <strong>WebSocket</strong>
        <span className="panel-value">{wsUrl}</span>
      </p>

      <p>
        <strong>Status</strong>
        <span className="panel-value">{status}</span>
      </p>

      <p>
        <strong>Messages</strong>
        <span className="panel-value">{messageCount}</span>
      </p>

      <p>
        <strong>Last received</strong>
        <span className="panel-value">{lastReceivedText}</span>
      </p>

      {error && <p className="error">{error}</p>}
    </section>
  );
}

export default ConnectionStatusPanel;
