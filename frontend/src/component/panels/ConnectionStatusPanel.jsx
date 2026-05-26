import { useEffect, useState } from "react";

function getFreshness(lastMessageAt, now) {
  if (!lastMessageAt) {
    return {
      ageText: "-",
      label: "waiting",
      level: "waiting",
    };
  }

  if (!now) {
    return {
      ageText: "-",
      label: "checking",
      level: "waiting",
    };
  }

  const ageSeconds = Math.max(0, Math.floor((now - new Date(lastMessageAt).getTime()) / 1000));

  if (ageSeconds <= 5) {
    return {
      ageText: `${ageSeconds}s ago`,
      label: "fresh",
      level: "fresh",
    };
  }

  if (ageSeconds <= 10) {
    return {
      ageText: `${ageSeconds}s ago`,
      label: "delayed",
      level: "delayed",
    };
  }

  return {
    ageText: `${ageSeconds}s ago`,
    label: "stale",
    level: "stale",
  };
}

function ConnectionStatusPanel({
  apiBaseUrl,
  wsUrl,
  status,
  error,
  messageCount = 0,
  lastMessageAt,
  reconnectAttempt,
  displayMode,
}) {
  const [now, setNow] = useState(0);
  const lastReceivedText = lastMessageAt ? new Date(lastMessageAt).toLocaleTimeString() : "No stream yet";
  const freshness = getFreshness(lastMessageAt, now);

  useEffect(() => {
    const timerId = window.setInterval(() => {
      setNow(Date.now());
    }, 1000);

    return () => {
      window.clearInterval(timerId);
    };
  }, []);

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
        <strong>Reconnect</strong>
        <span className="panel-value">{reconnectAttempt > 0 ? `attempt ${reconnectAttempt}` : "ready"}</span>
      </p>

      <p>
        <strong>Display</strong>
        <span className="panel-value">{displayMode}</span>
      </p>

      <p>
        <strong>Freshness</strong>
        <span className={`freshness-badge freshness-${freshness.level}`}>{freshness.label}</span>
      </p>

      <p>
        <strong>Data age</strong>
        <span className="panel-value">{freshness.ageText}</span>
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
