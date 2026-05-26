import { buildInferenceEventLog } from "../../utils/inferenceEventLog";

function EventLogPanel({ messageHistory }) {
  const logItems = buildInferenceEventLog(messageHistory);

  return (
    <section className="panel">
      <h2>Event Log</h2>

      {logItems.length > 0 ? (
        <ul className="event-log">
          {logItems.map((item) => (
            <li className={`event-log-item event-${item.type}`} key={item.id}>
              <span className="event-frame">{item.frameLabel}</span>
              <strong>{item.title}</strong>
              <span>{item.detail}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p>
          <strong>Events</strong>
          <span className="panel-value">No notable changes</span>
        </p>
      )}
    </section>
  );
}

export default EventLogPanel;
