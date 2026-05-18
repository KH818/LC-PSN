export function normalizeNumberList(values) {
  if (!Array.isArray(values)) return [];

  return values.map((value) => Number(value)).filter((value) => Number.isFinite(value));
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
