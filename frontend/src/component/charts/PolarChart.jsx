function PolarChart({ data }) {
  const doas = data?.doas_deg ?? [];

  return (
    <div className="chart-card">
      <h2>Polar Chart</h2>
      <p>DOA angle visualization area</p>

      {doas.length > 0 && (
        <p>
          <strong>DOA:</strong> {doas.join(", ")}°
        </p>
      )}
    </div>
  );
}

export default PolarChart;