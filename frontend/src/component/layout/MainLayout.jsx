import Header from "./Header";
import Sidebar from "./Sidebar";
import PolarChart from "../charts/PolarChart";
import SpectrumChart from "../charts/SpectrumChart";
import WaterfallChart from "../charts/WaterfallChart";
import InferenceSummaryPanel from "../panels/InferenceSummaryPanel";
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
  // 차트 컴포넌트들이 백엔드 이벤트 구조에 직접 의존하지 않도록 한 번 정규화한다.
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
        />

        <section className="charts">
          <InferenceSummaryPanel inference={inference} />
          <PolarChart inference={inference} />
          <SpectrumChart inference={inference} />
          <WaterfallChart inference={inference} messageHistory={messageHistory} />
        </section>
      </main>
    </div>
  );
}

export default MainLayout;
