import Header from "./Header";
import Sidebar from "./Sidebar";
import PolarChart from "../charts/PolarChart";
import SpectrumChart from "../charts/SpectrumChart";
import WaterfallChart from "../charts/WaterfallChart";

function MainLayout({
  apiBaseUrl,
  wsUrl,
  status,
  error,
  message,
  onStart,
  onStop,
}) {
  return (
    <div className="app">
      <Header />

      <main className="dashboard">
        <Sidebar
          apiBaseUrl={apiBaseUrl}
          wsUrl={wsUrl}
          status={status}
          error={error}
          onStart={onStart}
          onStop={onStop}
        />

        <section className="charts">
          <PolarChart data={message?.data} />
          <SpectrumChart data={message?.data} />
          <WaterfallChart data={message?.data} />

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

export default MainLayout;