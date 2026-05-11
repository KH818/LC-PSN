import "./App.css";
import { useWebSocket } from "./hooks/useWebSocket";

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
    <div className="app">
      <header className="header">
        <h1>LC-PSN Realtime DOA Dashboard</h1>
        <p>UI/UX · React · D3.js</p>
      </header>

      <main className="dashboard">
        <aside className="sidebar">
          <section className="panel">
            <h2>Connection Status</h2>

            <p>
              <strong>API:</strong> {apiBaseUrl}
            </p>

            <p>
              <strong>WebSocket:</strong> {wsUrl}
            </p>

            <p>
              <strong>Status:</strong> {status}
            </p>

            {error && <p className="error">{error}</p>}
          </section>

          <section className="panel">
            <h2>Control Panel</h2>
            <button type="button" onClick={handleStart}>
              Start
            </button>
            <button type="button" onClick={handleStop}>
              Stop
            </button>
          </section>
        </aside>

        <section className="charts">
          <div className="chart-card">
            <h2>Polar Chart</h2>
            <p>DOA angle visualization area</p>
          </div>

          <div className="chart-card">
            <h2>Spectrum Chart</h2>
            <p>Spectrum line chart area</p>
          </div>

          <div className="chart-card">
            <h2>Waterfall Chart</h2>
            <p>Realtime history visualization area</p>
          </div>

          <div className="chart-card">
            <h2>Latest Data</h2>
            <pre>
              {message
                ? JSON.stringify(message, null, 2)
                : JSON.stringify(
                    {
                      azimuth: null,
                      elevation: null,
                      confidence: null,
                    },
                    null,
                    2
                  )}
            </pre>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;