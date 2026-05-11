function SpectrumChart({ data }) {
  const spectrum = data?.spectrum ?? [];

  return (
    <div className="chart-card">
      <h2>Spectrum Chart</h2>
      <p>Spectrum line chart area</p>

      {spectrum.length > 0 && (
        <p>
          <strong>Spectrum length:</strong> {spectrum.length}
        </p>
      )}
    </div>
  );
}

export default SpectrumChart;