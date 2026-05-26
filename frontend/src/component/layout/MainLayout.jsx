import { useState } from "react";
import Header from "./Header";
import Sidebar from "./Sidebar";
import PolarChart from "../charts/PolarChart";
import SpectrumChart from "../charts/SpectrumChart";
import WaterfallChart from "../charts/WaterfallChart";
import InferenceSummaryPanel from "../panels/InferenceSummaryPanel";
import DBQueryResultsPanel from "../panels/DBQueryResultsPanel";
import { normalizeInferenceEvent } from "../../utils/inferenceEvent";

function MainLayout({
  apiBaseUrl,
  wsUrl,
  status,
  error,
  message,
  messageHistory,
  totalMessageCount,
  lastMessageAt,
  reconnectAttempt,
  paused,
  displayMode,
  replayIndex,
  onStart,
  onTogglePause,
  onReplayChange,
  onReturnLive,
}) {
  const [queryResults, setQueryResults] = useState([]);

  const inference = normalizeInferenceEvent(message);

  return (
    <div className="app">
      <Header />

      <main className="dashboard">
        <Sidebar
          apiBaseUrl={apiBaseUrl}
          wsUrl={wsUrl}
          status={status}
          error={error}
          messageCount={totalMessageCount}
          messageHistory={messageHistory}
          lastMessageAt={lastMessageAt}
          reconnectAttempt={reconnectAttempt}
          paused={paused}
          displayMode={displayMode}
          replayIndex={replayIndex}
          onStart={onStart}
          onTogglePause={onTogglePause}
          onReplayChange={onReplayChange}
          onReturnLive={onReturnLive}
          onQueryResults={setQueryResults}
        />

        <section className="charts">
          <InferenceSummaryPanel inference={inference} />
          <PolarChart inference={inference} />
          <SpectrumChart inference={inference} />
          <WaterfallChart inference={inference} messageHistory={messageHistory} />
          <DBQueryResultsPanel queryResults={queryResults} />
        </section>
      </main>
    </div>
  );
}

export default MainLayout;