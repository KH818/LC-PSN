function ControlPanel({ onStart, onStop }) {
  return (
    <section className="panel">
      <h2>Control Panel</h2>

      <div className="controls">
        <button type="button" onClick={onStart}>
          Check Server
        </button>

        <button type="button" onClick={onStop}>
          Stop
        </button>
      </div>
    </section>
  );
}

export default ControlPanel;
