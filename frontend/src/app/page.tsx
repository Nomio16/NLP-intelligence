"use client";

import { useState, useRef, useCallback } from "react";

const API_BASE = "http://localhost:8000";

interface Entity {
  word: string;
  entity_group: string;
  score: number;
}

interface Sentiment {
  label: string;
  score: number;
}

interface DocumentResult {
  id: string;
  text: string;
  clean_text: string;
  source: string;
  entities: Entity[];
  sentiment: Sentiment | null;
}

interface AnalysisResult {
  documents: DocumentResult[];
  network: { nodes: any[]; edges: any[] } | null;
  topic_summary: any[];
  sentiment_summary: { [key: string]: number };
  entity_summary: { [key: string]: { word: string; count: number }[] };
  total_documents: number;
}

interface InsightItem {
  category: string;
  title: string;
  description: string;
  count: number;
  sample_texts: string[];
}

export default function Dashboard() {
  const [data, setData] = useState<AnalysisResult | null>(null);
  const [insights, setInsights] = useState<InsightItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [textInput, setTextInput] = useState("");
  const [dragging, setDragging] = useState(false);
  const [activeTab, setActiveTab] = useState<"overview" | "documents" | "insights">("overview");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uploadCSV = useCallback(async (file: File) => {
    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_BASE}/api/upload?run_ner=true&run_sentiment=true&run_topics=false`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }
      const result: AnalysisResult = await res.json();
      setData(result);
      // Fetch insights
      const insightsRes = await fetch(`${API_BASE}/api/insights`, { method: "POST" });
      if (insightsRes.ok) {
        setInsights(await insightsRes.json());
      }
    } catch (e: any) {
      setError(e.message || "Error uploading file");
    } finally {
      setLoading(false);
    }
  }, []);

  const analyzeText = useCallback(async () => {
    if (!textInput.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textInput }),
      });
      if (!res.ok) throw new Error("Analysis failed");
      const result: AnalysisResult = await res.json();
      setData(result);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [textInput]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith(".csv")) uploadCSV(file);
    else setError("Please upload a CSV file");
  }, [uploadCSV]);

  const sentimentTotal = data
    ? Object.values(data.sentiment_summary).reduce((a, b) => a + b, 0)
    : 0;

  return (
    <div>
      {/* Upload Section */}
      {!data && !loading && (
        <section style={{ marginBottom: "2rem" }}>
          <div
            className={`upload-area ${dragging ? "dragging" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="upload-icon">📁</div>
            <p className="upload-text">
              <strong>CSV файл чирж оруулах</strong> эсвэл дарж сонгох
            </p>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.5rem" }}>
              &apos;text&apos; эсвэл &apos;Text&apos; баганатай CSV файл шаардлагатай
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              style={{ display: "none" }}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) uploadCSV(file);
              }}
            />
          </div>

          <div style={{ margin: "1.5rem 0", textAlign: "center", color: "var(--text-muted)" }}>
            — эсвэл текст шууд оруулах —
          </div>

          <div style={{ display: "flex", gap: "0.75rem" }}>
            <textarea
              className="text-input-area"
              placeholder="Монгол хэлний текст бичнэ үү..."
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              style={{ flex: 1 }}
            />
            <button className="btn btn-primary" onClick={analyzeText} style={{ alignSelf: "flex-end" }}>
              🔍 Шинжлэх
            </button>
          </div>
        </section>
      )}

      {/* Loading */}
      {loading && (
        <div className="loading">
          <div className="spinner" />
          Загвар ачааллаж, шинжилгээ хийж байна... (анх удаа удаан байж болно)
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="card" style={{ borderColor: "var(--negative)", marginBottom: "1rem" }}>
          <p style={{ color: "var(--negative)" }}>❌ {error}</p>
        </div>
      )}

      {/* Results */}
      {data && !loading && (
        <>
          {/* Stats */}
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value" style={{ color: "var(--accent)" }}>
                {data.total_documents}
              </div>
              <div className="stat-label">Нийт нийтлэл</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{ color: "var(--positive)" }}>
                {data.sentiment_summary.positive || 0}
              </div>
              <div className="stat-label">Эерэг</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{ color: "var(--neutral)" }}>
                {data.sentiment_summary.neutral || 0}
              </div>
              <div className="stat-label">Саармаг</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{ color: "var(--negative)" }}>
                {data.sentiment_summary.negative || 0}
              </div>
              <div className="stat-label">Сөрөг</div>
            </div>
          </div>

          {/* Tabs */}
          <div className="tabs">
            <button className={`tab ${activeTab === "overview" ? "active" : ""}`} onClick={() => setActiveTab("overview")}>
              📊 Ерөнхий
            </button>
            <button className={`tab ${activeTab === "documents" ? "active" : ""}`} onClick={() => setActiveTab("documents")}>
              📄 Нийтлэлүүд
            </button>
            <button className={`tab ${activeTab === "insights" ? "active" : ""}`} onClick={() => setActiveTab("insights")}>
              💡 Дүгнэлт
            </button>
          </div>

          {/* Overview Tab */}
          {activeTab === "overview" && (
            <div className="chart-grid">
              {/* Sentiment */}
              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">Сэтгэгдлийн шинжилгээ</h3>
                </div>
                <div className="sentiment-bars">
                  {["positive", "neutral", "negative"].map((label) => {
                    const count = data.sentiment_summary[label] || 0;
                    const pct = sentimentTotal > 0 ? (count / sentimentTotal) * 100 : 0;
                    return (
                      <div className="sentiment-row" key={label}>
                        <span className="sentiment-label">
                          {label === "positive" ? "Эерэг" : label === "negative" ? "Сөрөг" : "Саармаг"}
                        </span>
                        <div className="sentiment-bar-bg">
                          <div className={`sentiment-bar ${label}`} style={{ width: `${pct}%` }} />
                        </div>
                        <span className="sentiment-count">{count}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Top Entities */}
              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">Шилдэг нэрлэсэн объектууд</h3>
                </div>
                {Object.entries(data.entity_summary).map(([type, entities]) => (
                  <div key={type} style={{ marginBottom: "1rem" }}>
                    <h4 style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "0.5rem" }}>{type}</h4>
                    <div className="entity-list">
                      {entities.slice(0, 10).map((e, i) => (
                        <span className={`entity-tag ${type}`} key={i}>
                          {e.word}
                          <span className="entity-count">{e.count}</span>
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              {/* Network placeholder */}
              {data.network && data.network.nodes.length > 0 && (
                <div className="card chart-full">
                  <div className="card-header">
                    <h3 className="card-title">🕸️ Нэрлэсэн объектуудын сүлжээ (Network Graph)</h3>
                    <span className="card-subtitle">{data.network.nodes.length} зангилаа, {data.network.edges.length} холбоос</span>
                  </div>
                  <div className="network-container" style={{ padding: "2rem", display: "flex", flexWrap: "wrap", gap: "0.5rem", justifyContent: "center", alignItems: "center" }}>
                    {data.network.nodes
                      .sort((a, b) => b.frequency - a.frequency)
                      .slice(0, 30)
                      .map((node) => {
                        const size = Math.max(40, Math.min(120, node.frequency * 5));
                        const colorMap: Record<string, string> = { PER: "#ff6b6b", ORG: "#4ecdc4", LOC: "#ffd93d", MISC: "#a78bfa" };
                        return (
                          <div
                            key={node.id}
                            style={{
                              width: size, height: size,
                              borderRadius: "50%",
                              background: `${colorMap[node.entity_type] || "#6c63ff"}20`,
                              border: `2px solid ${colorMap[node.entity_type] || "#6c63ff"}`,
                              display: "flex", alignItems: "center", justifyContent: "center",
                              fontSize: Math.max(9, Math.min(14, node.frequency)),
                              color: colorMap[node.entity_type] || "#6c63ff",
                              fontWeight: 600,
                              textAlign: "center",
                              padding: "4px",
                              cursor: "pointer",
                              transition: "transform 0.2s",
                            }}
                            title={`${node.label} (${node.entity_type}) — ${node.frequency}x`}
                            onMouseOver={(e) => (e.currentTarget.style.transform = "scale(1.15)")}
                            onMouseOut={(e) => (e.currentTarget.style.transform = "scale(1)")}
                          >
                            {node.label.length > 12 ? node.label.slice(0, 10) + "…" : node.label}
                          </div>
                        );
                      })}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Documents Tab */}
          {activeTab === "documents" && (
            <div className="card">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Текст</th>
                    <th>Нэрлэсэн объектууд</th>
                    <th>Сэтгэгдэл</th>
                    <th>Эх сурвалж</th>
                  </tr>
                </thead>
                <tbody>
                  {data.documents.slice(0, 50).map((doc) => (
                    <tr key={doc.id}>
                      <td style={{ maxWidth: 400, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {doc.text.slice(0, 120)}
                      </td>
                      <td>
                        <div className="entity-list">
                          {doc.entities.slice(0, 4).map((e, i) => (
                            <span className={`entity-tag ${e.entity_group}`} key={i} style={{ fontSize: "0.7rem", padding: "0.2rem 0.5rem" }}>
                              {e.word}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td>
                        {doc.sentiment && (
                          <span className={`badge badge-${doc.sentiment.label}`}>
                            {doc.sentiment.label === "positive" ? "Эерэг" : doc.sentiment.label === "negative" ? "Сөрөг" : "Саармаг"}
                          </span>
                        )}
                      </td>
                      <td>{doc.source}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Insights Tab */}
          {activeTab === "insights" && (
            <div className="insights-grid">
              {insights.length === 0 && (
                <p style={{ color: "var(--text-muted)" }}>Дүгнэлт олдсонгүй.</p>
              )}
              {insights.map((insight, i) => (
                <div className={`insight-card ${insight.category}`} key={i}>
                  <h4 className="insight-title">{insight.title}</h4>
                  <p className="insight-description">{insight.description}</p>
                  {insight.sample_texts.length > 0 && (
                    <div style={{ marginTop: "0.75rem" }}>
                      {insight.sample_texts.slice(0, 2).map((t, j) => (
                        <p key={j} style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem", fontStyle: "italic" }}>
                          &quot;{t.slice(0, 100)}...&quot;
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Reset button */}
          <div style={{ textAlign: "center", marginTop: "2rem" }}>
            <button className="btn btn-secondary" onClick={() => { setData(null); setInsights([]); setTextInput(""); }}>
              🔄 Шинэ өгөгдөл оруулах
            </button>
          </div>
        </>
      )}
    </div>
  );
}
