import { normalizeInferenceEvent } from "./inferenceEvent";

const CONFIDENCE_LOW_THRESHOLD = 0.8;
const CONFIDENCE_DROP_THRESHOLD = 0.12;
const DOA_SHIFT_THRESHOLD_DEG = 10;
const MAX_LOG_ITEMS = 8;

function formatFrame(index) {
  return `Frame ${index + 1}`;
}

function getMaxDoaShift(previousDoas, currentDoas) {
  if (previousDoas.length === 0 || currentDoas.length === 0) return 0;

  return currentDoas.reduce((maxShift, currentAngle) => {
    const nearestPrevious = previousDoas.reduce((nearest, previousAngle) => {
      const distance = Math.abs(Number(currentAngle) - Number(previousAngle));
      return Math.min(nearest, distance);
    }, Number.POSITIVE_INFINITY);

    return Math.max(maxShift, nearestPrevious);
  }, 0);
}

function createLogItem(type, title, detail, frameIndex) {
  return {
    id: `${frameIndex}-${type}-${title}`,
    type,
    title,
    detail,
    frameLabel: formatFrame(frameIndex),
  };
}

export function buildInferenceEventLog(messageHistory) {
  const normalizedHistory = messageHistory.map((message) => normalizeInferenceEvent(message));
  const logItems = [];

  normalizedHistory.forEach((current, index) => {
    if (index === 0) return;

    const previous = normalizedHistory[index - 1];

    if (previous.kEstimate !== current.kEstimate) {
      logItems.push(
        createLogItem(
          "warning",
          "K estimate changed",
          `${previous.kEstimate ?? "-"} -> ${current.kEstimate ?? "-"}`,
          index
        )
      );
    }

    const sourceDelta = current.doas.length - previous.doas.length;

    if (sourceDelta > 0) {
      logItems.push(createLogItem("notice", "Source count increased", `+${sourceDelta} source`, index));
    }

    if (sourceDelta < 0) {
      logItems.push(createLogItem("warning", "Source count decreased", `${sourceDelta} source`, index));
    }

    const maxShift = getMaxDoaShift(previous.doas, current.doas);

    if (maxShift >= DOA_SHIFT_THRESHOLD_DEG) {
      logItems.push(createLogItem("warning", "DOA shifted", `Max shift ${Math.round(maxShift)} deg`, index));
    }

    if (Number.isFinite(current.confidence) && current.confidence < CONFIDENCE_LOW_THRESHOLD) {
      logItems.push(
        createLogItem("critical", "Low confidence", `${Math.round(current.confidence * 100)}%`, index)
      );
    }

    if (
      Number.isFinite(previous.confidence) &&
      Number.isFinite(current.confidence) &&
      previous.confidence - current.confidence >= CONFIDENCE_DROP_THRESHOLD
    ) {
      logItems.push(
        createLogItem(
          "warning",
          "Confidence dropped",
          `${Math.round(previous.confidence * 100)}% -> ${Math.round(current.confidence * 100)}%`,
          index
        )
      );
    }
  });

  return logItems.slice(-MAX_LOG_ITEMS).reverse();
}
