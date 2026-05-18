import "./App.css";
import { useWebSocket } from "./hooks/useWebSocket";
import MainLayout from "./component/layout/MainLayout";

function App() {
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
  const wsUrl = import.meta.env.VITE_WS_URL;
  const { status, message, messageHistory, lastMessageAt, error } = useWebSocket(wsUrl);

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

  const handleStop = () => {
    console.log("Stop clicked");
  };

  return (
    <MainLayout
      apiBaseUrl={apiBaseUrl}
      wsUrl={wsUrl}
      status={status}
      error={error}
      message={message}
      messageHistory={messageHistory}
      lastMessageAt={lastMessageAt}
      onStart={handleStart}
      onStop={handleStop}
    />
  );
}

export default App;
