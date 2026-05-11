import ConnectionStatusPanel from "../panels/ConnectionStatusPanel";
import ControlPanel from "../panels/ControlPanel";

function Sidebar({ apiBaseUrl, wsUrl, status, error, onStart, onStop }) {
  return (
    <aside className="sidebar">
      <ConnectionStatusPanel
        apiBaseUrl={apiBaseUrl}
        wsUrl={wsUrl}
        status={status}
        error={error}
      />

      <ControlPanel onStart={onStart} onStop={onStop} />
    </aside>
  );
}

export default Sidebar;