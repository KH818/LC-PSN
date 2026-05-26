import { useEffect, useState } from "react";

function DBQueryPanel({ apiBaseUrl, onQueryResults }) {
  const [searchId, setSearchId] = useState("");
  const [minDoa, setMinDoa] = useState("-30");
  const [maxDoa, setMaxDoa] = useState("30");
  const [minK, setMinK] = useState("3");
  const [limit, setLimit] = useState("20");
  const [recentIds, setRecentIds] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchJson = async (url) => {
    setLoading(true);

    try {
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      return await response.json();
    } finally {
      setLoading(false);
    }
  };

  const handleRecentIds = async () => {
    try {
      const data = await fetchJson(`${apiBaseUrl}/raw-signals/recent?limit=10`);
      setRecentIds(data);
    } catch (error) {
      console.error("최근 ID 조회 실패:", error);
    }
  };

  const handleShowRecentIds = async () => {
    try {
      const data = await fetchJson(`${apiBaseUrl}/raw-signals/recent?limit=${limit}`);
      setRecentIds(data);
      onQueryResults(data);
    } catch (error) {
      alert("최근 ID 조회 실패");
      console.error(error);
    }
  };

  const handleRecentInference = async () => {
    try {
      const data = await fetchJson(`${apiBaseUrl}/inference-results/recent?limit=${limit}`);
      onQueryResults(data);
    } catch (error) {
      alert("최근 추론 결과 조회 실패");
      console.error(error);
    }
  };

  const handleLatestInference = async () => {
    try {
      const data = await fetchJson(`${apiBaseUrl}/inference-results/latest`);
      onQueryResults(data ? [data] : []);
    } catch (error) {
      alert("최신 추론 결과 조회 실패");
      console.error(error);
    }
  };

  const handleSearchById = async (idValue = searchId) => {
    if (!idValue.trim()) {
      alert("조회할 id를 입력하세요.");
      return;
    }

    try {
      const data = await fetchJson(`${apiBaseUrl}/signals/${idValue.trim()}/results`);
      onQueryResults(data);
    } catch (error) {
      alert("ID 조회 실패");
      console.error(error);
    }
  };

  const handleSearchByDoaRange = async () => {
    try {
      const data = await fetchJson(
        `${apiBaseUrl}/inference-results/doa-range?min_doa=${minDoa}&max_doa=${maxDoa}&limit=${limit}`
      );
      onQueryResults(data);
    } catch (error) {
      alert("DOA 범위 조회 실패");
      console.error(error);
    }
  };

  const handleSearchByMinK = async () => {
    try {
      const data = await fetchJson(
        `${apiBaseUrl}/inference-results/min-k?min_k=${minK}&limit=${limit}`
      );
      onQueryResults(data);
    } catch (error) {
      alert("K 기준 조회 실패");
      console.error(error);
    }
  };

  useEffect(() => {
    handleRecentIds();

    const interval = setInterval(() => {
      handleRecentIds();
    }, 5000);

    return () => clearInterval(interval);
  }, [apiBaseUrl]);

  return (
    <section className="panel db-query-panel">
      <h2>DB Query</h2>

      <div className="query-block">
        <label>Result Limit</label>
        <input
          className="single-query-input"
          value={limit}
          onChange={(event) => setLimit(event.target.value)}
          placeholder="20"
        />
      </div>

      <div className="query-button-grid">
        <button onClick={handleShowRecentIds} disabled={loading}>
          Recent IDs
        </button>
        <button onClick={handleRecentInference} disabled={loading}>
          Recent Results
        </button>
        <button onClick={handleLatestInference} disabled={loading}>
          Latest Result
        </button>
      </div>

      <div className="query-block">
        <label>ID Search</label>
        <div className="query-row">
          <input
            value={searchId}
            onChange={(event) => setSearchId(event.target.value)}
            placeholder="raw_xxxxx"
          />
          <button onClick={() => handleSearchById()} disabled={loading}>
            Search
          </button>
        </div>
      </div>

      <div className="query-block">
        <label>DOA Range</label>
        <div className="query-row">
          <input
            value={minDoa}
            onChange={(event) => setMinDoa(event.target.value)}
            placeholder="-30"
          />
          <span>~</span>
          <input
            value={maxDoa}
            onChange={(event) => setMaxDoa(event.target.value)}
            placeholder="30"
          />
          <button onClick={handleSearchByDoaRange} disabled={loading}>
            Search
          </button>
        </div>
      </div>

      <div className="query-block">
        <label>Minimum K</label>
        <div className="query-row">
          <input
            value={minK}
            onChange={(event) => setMinK(event.target.value)}
            placeholder="3"
          />
          <button onClick={handleSearchByMinK} disabled={loading}>
            Search
          </button>
        </div>
      </div>

      <div className="recent-id-header">
        <span>Recent IDs</span>
        <small>auto 5s</small>
      </div>

      {recentIds.length > 0 ? (
        <ul className="recent-id-list">
          {recentIds.map((item) => (
            <li key={`${item.id}-${item.time}`}>
              <button
                className="id-link-button"
                onClick={() => {
                  setSearchId(item.id);
                  handleSearchById(item.id);
                }}
              >
                {item.id}
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="small-muted">No IDs yet</p>
      )}
    </section>
  );
}

export default DBQueryPanel;