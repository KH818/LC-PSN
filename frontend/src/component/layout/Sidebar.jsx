import ConnectionStatusPanel from "../panels/ConnectionStatusPanel";
import ControlPanel from "../panels/ControlPanel";

function Sidebar({ apiBaseUrl, wsUrl, status, error, messageCount, lastMessageAt, onStart, onStop }) {
  return (
    <aside className="sidebar">
      <ConnectionStatusPanel
        apiBaseUrl={apiBaseUrl}
        wsUrl={wsUrl}
        status={status}
        error={error}
        messageCount={messageCount}
        lastMessageAt={lastMessageAt}
      />

      <ControlPanel onStart={onStart} onStop={onStop} />
    </aside>
  );
}

export default Sidebar;
