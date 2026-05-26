import { useState } from "react";
import "./App.css";
import { useWebSocket } from "./hooks/useWebSocket";
import MainLayout from "./component/layout/MainLayout";

function App() {
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
  const wsUrl = import.meta.env.VITE_WS_URL;
  const [paused, setPaused] = useState(false);
  const [replayIndex, setReplayIndex] = useState(null);
  const { status, message, messageHistory, lastMessageAt, reconnectAttempt, error } = useWebSocket(wsUrl, { paused });
  const replayMessage = replayIndex == null ? null : messageHistory[replayIndex];
  const displayedMessage = replayMessage ?? message;
  const displayedHistory =
    replayIndex == null ? messageHistory : messageHistory.slice(Math.max(0, replayIndex - 35), replayIndex + 1);
  const displayMode = replayIndex == null ? (paused ? "paused" : "live") : "replay";

  const handleStart = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/health`);

      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }

      const result = await response.json();
      console.log("Server health:", result);
      alert("백엔드 서버가 연결되어 있습니다. 추론 결과는 WebSocket으로 자동 수신됩니다.");
    } catch (err) {
      console.error("Start request error:", err);
      alert("백엔드 연결 확인에 실패했습니다. 서버 실행 상태와 /health 엔드포인트를 확인하세요.");
    }
  };

  const handleTogglePause = () => {
    setReplayIndex(null);
    setPaused((current) => !current);
  };

  const handleReplayChange = (index) => {
    setPaused(true);
    setReplayIndex(index);
  };

  const handleReturnLive = () => {
    setReplayIndex(null);
    setPaused(false);
  };

  return (
    <MainLayout
      apiBaseUrl={apiBaseUrl}
      wsUrl={wsUrl}
      status={status}
      error={error}
      message={displayedMessage}
      messageHistory={displayedHistory}
      totalMessageCount={messageHistory.length}
      lastMessageAt={lastMessageAt}
      reconnectAttempt={reconnectAttempt}
      paused={paused}
      displayMode={displayMode}
      replayIndex={replayIndex}
      onStart={handleStart}
      onTogglePause={handleTogglePause}
      onReplayChange={handleReplayChange}
      onReturnLive={handleReturnLive}
    />
  );
}

export default App;
