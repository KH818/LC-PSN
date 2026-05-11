function ControlPanel({ onStart, onStop }) {
  return (
    <section className="panel">
      <h2>Control Panel</h2>

      <button type="button" onClick={onStart}>
        Start
      </button>

      <button type="button" onClick={onStop}>
        Stop
      </button>
    </section>
  );
}

export default ControlPanel;