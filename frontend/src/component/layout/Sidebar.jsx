import ConnectionStatusPanel from "../panels/ConnectionStatusPanel";
import ControlPanel from "../panels/ControlPanel";

function Sidebar({ apiBaseUrl, wsUrl, status, error, messageCount, lastMessageAt, paused, onStart, onTogglePause }) {
  return (
    <aside className="sidebar">
      <ConnectionStatusPanel
        apiBaseUrl={apiBaseUrl}
        wsUrl={wsUrl}
        status={status}
        error={error}
        messageCount={messageCount}
        lastMessageAt={lastMessageAt}
        paused={paused}
      />

      <ControlPanel paused={paused} onStart={onStart} onTogglePause={onTogglePause} />
    </aside>
  );
}

export default Sidebar;
