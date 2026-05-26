function ControlPanel({ paused, onStart, onTogglePause }) {
  return (
    <section className="panel">
      <h2>Control Panel</h2>

      <div className="controls">
        <button type="button" onClick={onStart}>
          Check Server
        </button>

        <button type="button" onClick={onTogglePause}>
          {paused ? "Resume Display" : "Pause Display"}
        </button>
      </div>
    </section>
  );
}

export default ControlPanel;
