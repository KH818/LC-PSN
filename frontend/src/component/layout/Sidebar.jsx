import ConnectionStatusPanel from "../panels/ConnectionStatusPanel";
import ControlPanel from "../panels/ControlPanel";
import EventLogPanel from "../panels/EventLogPanel";
import ReplayPanel from "../panels/ReplayPanel";

function Sidebar({
  apiBaseUrl,
  wsUrl,
  status,
  error,
  messageCount,
  messageHistory,
  lastMessageAt,
  paused,
  displayMode,
  replayIndex,
  onStart,
  onTogglePause,
  onReplayChange,
  onReturnLive,
}) {
  return (
    <aside className="sidebar">
      <ConnectionStatusPanel
        apiBaseUrl={apiBaseUrl}
        wsUrl={wsUrl}
        status={status}
        error={error}
        messageCount={messageCount}
        lastMessageAt={lastMessageAt}
        displayMode={displayMode}
      />

      <ControlPanel paused={paused} onStart={onStart} onTogglePause={onTogglePause} />

      <ReplayPanel
        historyLength={messageCount}
        replayIndex={replayIndex}
        onReplayChange={onReplayChange}
        onReturnLive={onReturnLive}
      />

      <EventLogPanel messageHistory={messageHistory} />
    </aside>
  );
}

export default Sidebar;
