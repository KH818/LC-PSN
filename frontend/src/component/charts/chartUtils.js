export function normalizeNumberList(values) {
  if (!Array.isArray(values)) return [];

  return values.map((value) => Number(value)).filter((value) => Number.isFinite(value));
}

export function downsampleNumberList(values, maxPoints) {
  if (!Array.isArray(values)) return [];
  if (!Number.isFinite(maxPoints) || maxPoints <= 0 || values.length <= maxPoints) return values;

  const bucketSize = Math.ceil(values.length / maxPoints);
  const sampled = [];

  for (let index = 0; index < values.length; index += bucketSize) {
    const bucket = values.slice(index, index + bucketSize);
    sampled.push(Math.max(...bucket));
  }

  return sampled;
}

export function getRange(values) {
  if (values.length === 0) return { min: 0, max: 1 };

  const min = Math.min(...values);
  const max = Math.max(...values);

  if (min === max) {
    return { min: min - 1, max: max + 1 };
  }

  return { min, max };
}

export function scaleValue(value, min, max, size) {
  return ((value - min) / (max - min)) * size;
}
