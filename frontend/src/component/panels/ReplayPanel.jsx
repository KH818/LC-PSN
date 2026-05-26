function ReplayPanel({ historyLength, replayIndex, onReplayChange, onReturnLive }) {
  const hasHistory = historyLength > 0;
  const currentIndex = replayIndex ?? Math.max(0, historyLength - 1);
  const frameText = hasHistory ? `${currentIndex + 1} / ${historyLength}` : "No history";

  const handleChange = (event) => {
    onReplayChange(Number(event.target.value));
  };

  return (
    <section className="panel">
      <h2>Replay</h2>

      <p>
        <strong>Frame</strong>
        <span className="panel-value">{frameText}</span>
      </p>

      <input
        aria-label="Replay frame"
        className="replay-slider"
        disabled={!hasHistory}
        max={Math.max(0, historyLength - 1)}
        min="0"
        onChange={handleChange}
        type="range"
        value={currentIndex}
      />

      <div className="controls">
        <button className="secondary-button" disabled={!hasHistory} type="button" onClick={onReturnLive}>
          Back to Live
        </button>
      </div>
    </section>
  );
}

export default ReplayPanel;
