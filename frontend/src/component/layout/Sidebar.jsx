import ConnectionStatusPanel from "../panels/ConnectionStatusPanel";
import ControlPanel from "../panels/ControlPanel";
import DBQueryPanel from "../panels/DBQueryPanel";
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
  reconnectAttempt,
  paused,
  displayMode,
  replayIndex,
  onStart,
  onTogglePause,
  onReplayChange,
  onReturnLive,
  onQueryResults,
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
        reconnectAttempt={reconnectAttempt}
        displayMode={displayMode}
      />

      <ControlPanel paused={paused} onStart={onStart} onTogglePause={onTogglePause} />

      <DBQueryPanel apiBaseUrl={apiBaseUrl} onQueryResults={onQueryResults} />

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