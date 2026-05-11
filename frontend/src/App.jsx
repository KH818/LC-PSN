import "./App.css";
import { useWebSocket } from "./hooks/useWebSocket";
import MainLayout from "./component/layout/MainLayout";

function App() {
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;
  const wsUrl = import.meta.env.VITE_WS_URL;
  const { status, message, error } = useWebSocket(wsUrl);

  const handleStart = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          source: "mock",
          snr: null,
        }),
      });

      if (!response.ok) {
        throw new Error(`Predict request failed: ${response.status}`);
      }

      const result = await response.json();
      console.log("Predict response:", result);
    } catch (err) {
      console.error("Start request error:", err);
      alert("Predict 요청 실패. 백엔드 /predict 요청 형식을 확인하세요.");
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
      onStart={handleStart}
      onStop={handleStop}
    />
  );
}

export default App;