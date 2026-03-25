"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import AnnotationEditor, { DocForEditor } from "./components/AnnotationEditor";

const API_BASE = "";
//const API_BASE = "";

interface Entity {
  word: string;
  entity_group: string;
  score: number;
  start?: number | null;
  end?: number | null;
}

interface Sentiment {
  label: string;
  score: number;
}

interface DocumentResult {
  id: string;
  doc_id?: number | null;
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

interface HistorySession {
  id: number;
  created_at: string;
  source_filename: string;
  total_documents: number;
  sentiment_summary: { [key: string]: number };
  topic_summary: any[];
}

interface GlobalAnalysis {
  total_documents: number;
  topic_summary: any[];
  network: { nodes: any[]; edges: any[] } | null;
}

function NetworkGraph({ network }: { network: { nodes: any[]; edges: any[] } }) {
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const W = 780, H = 500;
  const cx = W / 2, cy = H / 2;
  const colorMap: Record<string, string> = {
    PER: "#ff6b6b", ORG: "#4ecdc4", LOC: "#ffd93d", MISC: "#a78bfa",
  };

  // Pick top nodes sorted by frequency
  const topNodes = [...network.nodes]
    .sort((a, b) => b.frequency - a.frequency)
    .slice(0, 40);

  // Arrange nodes in concentric rings by entity type so same-type nodes
  // cluster together, making co-occurrence edges easier to read.
  const typeOrder = ["PER", "ORG", "LOC", "MISC"];
  const ringRadii = [105, 168, 225, 278];
  const grouped: Record<string, typeof topNodes> = {};
  for (const node of topNodes) {
    const t = node.entity_type || "MISC";
    if (!grouped[t]) grouped[t] = [];
    grouped[t].push(node);
  }

  const posMap = new Map<string, { x: number; y: number }>();
  typeOrder.forEach((type, ti) => {
    const group = grouped[type] || [];
    const r = ringRadii[Math.min(ti, ringRadii.length - 1)];
    group.forEach((node, i) => {
      // Offset each ring's start angle slightly so labels don't collide
      const offset = (ti * Math.PI) / 4;
      const angle = offset + (2 * Math.PI * i) / Math.max(group.length, 1);
      posMap.set(node.id, { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) });
    });
  });

  // Show top edges by weight (only between visible nodes)
  const topEdges = [...network.edges]
    .filter(e => posMap.has(e.source) && posMap.has(e.target) && e.source !== e.target)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 80);

  const maxWeight = topEdges.length > 0 ? topEdges[0].weight : 1;

  return (
    <div>
      <div style={{ overflowX: "auto" }}>
        <svg
          width="100%"
          viewBox={`0 0 ${W} ${H}`}
          style={{ display: "block", margin: "0 auto", minWidth: 340 }}
        >
          {/* Edges */}
          {topEdges.map((edge, i) => {
            const s = posMap.get(edge.source)!;
            const t = posMap.get(edge.target)!;
            const isHighlighted = hoveredId === edge.source || hoveredId === edge.target;
            const opacity = isHighlighted ? 0.7 : 0.12 + (edge.weight / maxWeight) * 0.18;
            const strokeW = isHighlighted
              ? Math.max(2, (edge.weight / maxWeight) * 4)
              : Math.max(0.5, (edge.weight / maxWeight) * 1.5);
            return (
              <line
                key={i}
                x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                stroke={isHighlighted ? "rgba(255,255,255,0.55)" : "rgba(255,255,255,0.25)"}
                strokeWidth={strokeW}
                strokeOpacity={opacity}
              />
            );
          })}

          {/* Nodes */}
          {topNodes.map(node => {
            const pos = posMap.get(node.id);
            if (!pos) return null;
            const r = Math.max(14, Math.min(32, 10 + node.frequency * 1.2));
            const color = colorMap[node.entity_type] || "#6c63ff";
            const isHovered = hoveredId === node.id;
            const label = node.label.length > 11 ? node.label.slice(0, 9) + "…" : node.label;
            return (
              <g
                key={node.id}
                style={{ cursor: "pointer" }}
                onMouseEnter={() => setHoveredId(node.id)}
                onMouseLeave={() => setHoveredId(null)}
              >
                <circle
                  cx={pos.x} cy={pos.y}
                  r={isHovered ? r + 5 : r}
                  fill={`${color}22`}
                  stroke={color}
                  strokeWidth={isHovered ? 3 : 1.8}
                  style={{ transition: "r 0.15s, stroke-width 0.15s" }}
                />
                <text
                  x={pos.x} y={pos.y}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill={isHovered ? "#fff" : color}
                  fontSize={Math.max(8, Math.min(11, r * 0.62))}
                  fontWeight={600}
                >
                  {label}
                </text>
                <title>{`${node.label} (${node.entity_type}) — ${node.frequency}×`}</title>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Legend */}
      <div style={{
        display: "flex", gap: "1.25rem", justifyContent: "center",
        marginTop: "0.75rem", flexWrap: "wrap",
      }}>
        {Object.entries(colorMap).map(([type, color]) => (
          <div key={type} style={{ display: "flex", alignItems: "center", gap: "0.35rem", fontSize: "0.72rem" }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: color, opacity: 0.8 }} />
            <span style={{ color: "var(--text-muted)" }}>{type}</span>
          </div>
        ))}
        <span style={{ color: "var(--text-muted)", fontSize: "0.7rem" }}>
          — {network.nodes.length} зангилаа · {network.edges.length} холбоос (шилдэг 40/80 харуулав)
        </span>
      </div>
    </div>
  );
}

// Standard headers needed for all API calls when going through Ngrok
const NGROK_HEADERS = { "ngrok-skip-browser-warning": "true" };

export default function Dashboard() {
  const [data, setData] = useState<AnalysisResult | null>(null);
  const [insights, setInsights] = useState<InsightItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [textInput, setTextInput] = useState("");
  const [dragging, setDragging] = useState(false);
  const [activeTab, setActiveTab] = useState<"overview" | "documents" | "insights" | "history">("overview");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // History
  const [history, setHistory] = useState<HistorySession[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  // Global analysis
  const [globalAnalysis, setGlobalAnalysis] = useState<GlobalAnalysis | null>(null);
  const [globalLoading, setGlobalLoading] = useState(false);
  const [globalError, setGlobalError] = useState("");

  // Annotation editor
  const [editingDoc, setEditingDoc] = useState<DocForEditor | null>(null);

  // Backend health check
  const [backendOk, setBackendOk] = useState<boolean | null>(null); // null = checking

  // Health check on mount — tells you immediately if backend is reachable
  useEffect(() => {
    const check = async () => {
      console.group("[NLP] Backend health check");
      try {
        const res = await fetch(`${API_BASE}/api/health`, { headers: NGROK_HEADERS });
        const ok = res.ok;
        setBackendOk(ok);
        console.log(ok ? "✅ Backend reachable" : `❌ Backend returned ${res.status}`);
      } catch (e) {
        setBackendOk(false);
        console.error("❌ Backend unreachable:", e);
      }
      console.groupEnd();
    };
    check();
  }, []);

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    console.group("[NLP] Load history");
    try {
      const res = await fetch(`${API_BASE}/api/history?limit=50`, { headers: NGROK_HEADERS });
      console.log(`→ GET /api/history  status=${res.status}`);
      if (res.ok) setHistory(await res.json());
    } catch (e) { console.error(e); }
    finally { setHistoryLoading(false); console.groupEnd(); }
  }, []);

  useEffect(() => {
    if (activeTab === "history") loadHistory();
  }, [activeTab, loadHistory]);

  const deleteSession = async (id: number) => {
    setDeletingId(id);
    try {
      await fetch(`${API_BASE}/api/history/${id}`, { headers: { "ngrok-skip-browser-warning": "true" }, method: "DELETE" });
      setHistory((prev) => prev.filter((s) => s.id !== id));
    } finally {
      setDeletingId(null);
    }
  };

  const openSession = async (id: number) => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/history/${id}`, { headers: { "ngrok-skip-browser-warning": "true" } });
      if (!res.ok) throw new Error("Could not load session");
      const session = await res.json();
      // Convert DB format → AnalysisResult format
      const documents: DocumentResult[] = (session.documents || []).map((d: any) => ({
        id: String(d.doc_index),
        doc_id: d.id,
        text: d.raw_text,
        clean_text: d.nlp_text,
        source: d.source,
        entities: d.entities || [],
        sentiment: d.sentiment?.label ? d.sentiment : null,
      }));
      setData({
        documents,
        network: null,
        topic_summary: session.topic_summary || [],
        sentiment_summary: session.sentiment_summary || {},
        entity_summary: session.entity_summary || {},
        total_documents: session.total_documents,
      });
      setActiveTab("overview");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const runGlobalAnalysis = async () => {
    setGlobalLoading(true);
    setGlobalError("");
    try {
      const res = await fetch(`${API_BASE}/api/global-analysis`, { headers: { "ngrok-skip-browser-warning": "true" } });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Global analysis failed");
      }
      setGlobalAnalysis(await res.json());
    } catch (e: any) {
      setGlobalError(e.message);
    } finally {
      setGlobalLoading(false);
    }
  };

  const uploadCSV = useCallback(async (file: File) => {
    setLoading(true);
    setError("");
    console.group(`[NLP] CSV Upload — ${file.name} (${(file.size/1024).toFixed(1)} KB)`);
    try {
      const formData = new FormData();
      formData.append("file", file);
      // ⚠️ IMPORTANT: ngrok-skip-browser-warning header MUST be included here.
      // Without it, Ngrok returns an HTML warning page instead of forwarding
      // the request to FastAPI → FastAPI tries to parse HTML as CSV → 500 error.
      const res = await fetch(`${API_BASE}/api/upload?run_ner=true&run_sentiment=true&run_topics=true`, {
        method: "POST",
        headers: NGROK_HEADERS,   // ← THE FIX
        body: formData,
      });
      console.log(`→ POST /api/upload  status=${res.status}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail || "Upload failed");
      }
      const result: AnalysisResult = await res.json();
      console.log(`← ${result.total_documents} documents, topics=${result.topic_summary?.length}`);
      setData(result);
      setActiveTab("overview");  // Auto-switch to results

      // Immediately fetch insights after upload
      const insightsRes = await fetch(`${API_BASE}/api/insights`, { headers: NGROK_HEADERS, method: "POST" });
      console.log(`→ POST /api/insights  status=${insightsRes.status}`);
      if (insightsRes.ok) setInsights(await insightsRes.json());
    } catch (e: any) {
      console.error("Upload error:", e);
      setError(e.message || "Error uploading file");
    } finally {
      setLoading(false);
      console.groupEnd();
    }
  }, []);

  const analyzeText = useCallback(async () => {
    if (!textInput.trim()) return;
    setLoading(true);
    setError("");
    console.group(`[NLP] Analyze text (${textInput.length} chars)`);
    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: { ...NGROK_HEADERS, "Content-Type": "application/json" },
        body: JSON.stringify({ text: textInput }),
      });
      console.log(`→ POST /api/analyze  status=${res.status}`);
      if (!res.ok) throw new Error("Analysis failed");
      const result: AnalysisResult = await res.json();
      console.log(`← entities:`, result.documents[0]?.entities?.length ?? 0,
        `sentiment:`, result.documents[0]?.sentiment?.label);
      setData(result);
    } catch (e: any) {
      console.error(e);
      setError(e.message);
    } finally {
      setLoading(false);
      console.groupEnd();
    }
  }, [textInput]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith(".csv")) uploadCSV(file);
    else setError("Please upload a CSV file");
  }, [uploadCSV]);

  const openEditor = (doc: DocumentResult) => {
    if (!doc.doc_id) return;
    setEditingDoc({
      doc_id: doc.doc_id,
      text: doc.text,
      entities: doc.entities,
      sentiment: doc.sentiment,
    });
  };

  const handleEditorSaved = (updated: DocForEditor) => {
    if (!data) return;
    setData({
      ...data,
      documents: data.documents.map((d) =>
        d.doc_id === updated.doc_id
          ? {
            ...d,
            entities: updated.entities,
            sentiment: updated.sentiment,
          }
          : d
      ),
    });
  };

  const sentimentTotal = data
    ? Object.values(data.sentiment_summary).reduce((a, b) => a + b, 0)
    : 0;

  return (
    <div>
      {/* Backend status banner */}
      {backendOk === false && (
        <div style={{
          background: "rgba(255,80,80,0.15)", border: "1px solid var(--negative)",
          borderRadius: "0.5rem", padding: "0.6rem 1rem", marginBottom: "1rem",
          display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.85rem",
        }}>
          <span>🔴</span>
          <span style={{ color: "var(--negative)", fontWeight: 600 }}>Backend холболт алдаатай.</span>
          <span style={{ color: "var(--text-muted)" }}>
            Colab дээрх сервер ажиллаж байгаа эсэхийг шалгаад, Ngrok URL зөв эсэхийг .env.local файлд шинэчилнэ үү.
          </span>
        </div>
      )}
      {backendOk === null && (
        <div style={{
          background: "rgba(100,100,200,0.1)", border: "1px solid rgba(100,100,255,0.3)",
          borderRadius: "0.5rem", padding: "0.4rem 1rem", marginBottom: "0.75rem",
          fontSize: "0.8rem", color: "var(--text-muted)",
        }}>
          ⏳ Backend холболт шалгаж байна...
        </div>
      )}

      {/* Annotation editor modal */}
      {editingDoc && (
        <AnnotationEditor
          doc={editingDoc}
          onClose={() => setEditingDoc(null)}
          onSaved={handleEditorSaved}
        />
      )}

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
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.5rem" }}>
              <p>⚠️ <strong>Санамж:</strong> Шинжлэх өгөгдөл тань заавал <code>text</code> эсвэл <code>Text</code> гэсэн нэртэй баганад байх ёстой.</p>
              <p>Хэрэв таны багана <code>Текст</code>, <code>Мессеж</code> гэх мэт Монгол нэртэй бол файлаа оруулахаас өмнө нэрийг нь <code>text</code> болгож өөрчилнө үү.</p>
            </div>
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

          {/* Quick history link when no active data */}
          <div style={{ marginTop: "1.5rem", textAlign: "center" }}>
            <button
              className="btn btn-secondary"
              onClick={() => setActiveTab("history")}
            >
              📂 Өмнөх шинжилгээнүүд харах
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

      {/* History tab shown even without data */}
      {!data && !loading && activeTab === "history" && (
        <div style={{ marginTop: "1rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
            <h3 style={{ margin: 0 }}>📂 Шинжилгээний түүх</h3>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <button className="btn btn-secondary" onClick={loadHistory} disabled={historyLoading}>
                {historyLoading ? "Уншиж байна..." : "🔄 Шинэчлэх"}
              </button>
              <button className="btn btn-primary" onClick={runGlobalAnalysis} disabled={globalLoading}>
                {globalLoading ? "Боловсруулж байна..." : "🌐 Нийт шинжилгээ"}
              </button>
            </div>
          </div>

          {globalError && <p style={{ color: "var(--negative)", fontSize: "0.85rem" }}>❌ {globalError}</p>}

          {globalAnalysis && (
            <div className="card" style={{ marginBottom: "1rem" }}>
              <div className="card-header">
                <h4 className="card-title">🌐 Нийт шинжилгээ — {globalAnalysis.total_documents} баримт</h4>
              </div>
              {globalAnalysis.topic_summary?.length > 0 && (
                <div style={{ marginBottom: "1rem" }}>
                  <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "0.5rem" }}>Сэдвүүд:</p>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                    {globalAnalysis.topic_summary.filter((t: any) => t.topic_id >= 0).map((t: any, i: number) => (
                      <div key={i} className="card" style={{ padding: "0.5rem 0.75rem", fontSize: "0.8rem" }}>
                        <strong>{t.name || t.topic_label || `Сэдэв ${t.topic_id}`}</strong>
                        <span style={{ color: "var(--text-muted)", marginLeft: "0.5rem" }}>({t.count} баримт)</span>
                        {t.keywords?.length > 0 && (
                          <div style={{ marginTop: "0.25rem", color: "var(--text-muted)", fontSize: "0.72rem" }}>
                            {t.keywords.slice(0, 5).join(", ")}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {globalAnalysis.network && globalAnalysis.network.nodes.length > 0 && (
                <div>
                  <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "0.5rem" }}>
                    🕸️ Нийт сүлжээ — {globalAnalysis.network.nodes.length} зангилаа
                  </p>
                  <NetworkGraph network={globalAnalysis.network} />
                </div>
              )}
            </div>
          )}

          {history.length === 0 && !historyLoading && (
            <p style={{ color: "var(--text-muted)" }}>Хадгалсан шинжилгээ олдсонгүй.</p>
          )}
          <div className="card" style={{ padding: 0, overflow: "hidden" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Огноо</th>
                  <th>Файл</th>
                  <th>Баримт</th>
                  <th>Сэтгэгдэл</th>
                  <th>Үйлдэл</th>
                </tr>
              </thead>
              <tbody>
                {history.map((s) => {
                  const pos = s.sentiment_summary?.positive || 0;
                  const neg = s.sentiment_summary?.negative || 0;
                  const neu = s.sentiment_summary?.neutral || 0;
                  return (
                    <tr key={s.id}>
                      <td style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>{s.id}</td>
                      <td style={{ fontSize: "0.8rem", whiteSpace: "nowrap" }}>
                        {new Date(s.created_at).toLocaleString()}
                      </td>
                      <td style={{ maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: "0.85rem" }}>
                        {s.source_filename || "—"}
                      </td>
                      <td style={{ textAlign: "center" }}>{s.total_documents}</td>
                      <td>
                        <span style={{ color: "var(--positive)", fontSize: "0.75rem" }}>+{pos} </span>
                        <span style={{ color: "var(--neutral)", fontSize: "0.75rem" }}>{neu} </span>
                        <span style={{ color: "var(--negative)", fontSize: "0.75rem" }}>-{neg}</span>
                      </td>
                      <td>
                        <div style={{ display: "flex", gap: "0.4rem" }}>
                          <button
                            className="btn btn-primary"
                            style={{ fontSize: "0.75rem", padding: "0.25rem 0.6rem" }}
                            onClick={() => openSession(s.id)}
                          >
                            Нээх
                          </button>
                          <button
                            className="btn btn-secondary"
                            style={{ fontSize: "0.75rem", padding: "0.25rem 0.6rem", color: "var(--negative)", borderColor: "var(--negative)" }}
                            onClick={() => deleteSession(s.id)}
                            disabled={deletingId === s.id}
                          >
                            {deletingId === s.id ? "..." : "🗑"}
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Results */}
      {data && !loading && (
        <>
          {/* Toolbar: new analysis + active file info */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem", flexWrap: "wrap", gap: "0.5rem" }}>
            <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              📄 {data.total_documents} нийтлэл шинжлэгдлээ
            </span>
            <button
              className="btn btn-secondary"
              style={{ fontSize: "0.8rem" }}
              onClick={() => { setData(null); setInsights([]); setError(""); setActiveTab("overview"); }}
            >
              ＋ Шинэ шинжилгээ
            </button>
          </div>

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
            <button className={`tab ${activeTab === "history" ? "active" : ""}`} onClick={() => setActiveTab("history")}>
              📂 Түүх
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

              {/* Topic summary */}
              {data.topic_summary?.length > 0 && !data.topic_summary[0]?.error && !data.topic_summary[0]?.info && (
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">🗂 Сэдвүүд</h3>
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                    {data.topic_summary.filter((t: any) => t.topic_id >= 0).map((t: any, i: number) => (
                      <div key={i} className="card" style={{ padding: "0.5rem 0.75rem", fontSize: "0.8rem", minWidth: 140 }}>
                        <strong>{t.name || t.topic_label || `Сэдэв ${t.topic_id}`}</strong>
                        <span style={{ color: "var(--text-muted)", marginLeft: "0.5rem" }}>({t.count})</span>
                        {t.keywords?.length > 0 && (
                          <div style={{ marginTop: "0.25rem", color: "var(--text-muted)", fontSize: "0.7rem" }}>
                            {t.keywords.slice(0, 5).join(", ")}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Network */}
              {data.network && data.network.nodes.length > 0 && (
                <div className="card chart-full">
                  <div className="card-header">
                    <h3 className="card-title">🕸️ Нэрлэсэн объектуудын сүлжээ</h3>
                    <span className="card-subtitle">{data.network.nodes.length} зангилаа, {data.network.edges.length} холбоос</span>
                  </div>
                  <NetworkGraph network={data.network} />
                </div>
              )}

              {/* Global analysis panel in overview */}
              <div className="card chart-full">
                <div className="card-header">
                  <h3 className="card-title">🌐 Нийт мэдээллийн шинжилгээ</h3>
                  <span className="card-subtitle">DB-д хадгалагдсан бүх баримтыг ашиглан сэдэв болон сүлжээ тооцоолно</span>
                </div>
                <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
                  <button className="btn btn-primary" onClick={runGlobalAnalysis} disabled={globalLoading}>
                    {globalLoading ? "Боловсруулж байна..." : "▶ Нийт шинжилгээ ажиллуулах"}
                  </button>
                  {globalError && <span style={{ color: "var(--negative)", fontSize: "0.85rem" }}>❌ {globalError}</span>}
                </div>
                {globalAnalysis && (
                  <div style={{ marginTop: "1rem" }}>
                    <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.75rem" }}>
                      {globalAnalysis.total_documents} баримт дээр үндэслэсэн үр дүн:
                    </p>
                    {globalAnalysis.topic_summary?.filter((t: any) => t.topic_id >= 0).length > 0 && (
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "1rem" }}>
                        {globalAnalysis.topic_summary.filter((t: any) => t.topic_id >= 0).map((t: any, i: number) => (
                          <div key={i} className="card" style={{ padding: "0.5rem 0.75rem", fontSize: "0.8rem" }}>
                            <strong>{t.name || t.topic_label || `Сэдэв ${t.topic_id}`}</strong>
                            <span style={{ color: "var(--text-muted)", marginLeft: "0.4rem" }}>({t.count})</span>
                            {t.keywords?.length > 0 && (
                              <div style={{ color: "var(--text-muted)", fontSize: "0.7rem", marginTop: 2 }}>
                                {t.keywords.slice(0, 5).join(", ")}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                    {globalAnalysis.network && globalAnalysis.network.nodes.length > 0 && (
                      <NetworkGraph network={globalAnalysis.network} />
                    )}
                  </div>
                )}
              </div>
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
                    <th>Засах</th>
                  </tr>
                </thead>
                <tbody>
                  {data.documents.slice(0, 100).map((doc, idx) => (
                    <tr key={doc.id || idx}>
                      <td style={{ maxWidth: 380, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
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
                      <td>
                        {doc.doc_id ? (
                          <button
                            className="btn btn-secondary"
                            style={{ fontSize: "0.75rem", padding: "0.25rem 0.6rem" }}
                            onClick={() => openEditor(doc)}
                          >
                            ✏️ Засах
                          </button>
                        ) : (
                          <span style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>—</span>
                        )}
                      </td>
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

          {/* History Tab (with data loaded) */}
          {activeTab === "history" && (
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                <h3 style={{ margin: 0 }}>📂 Шинжилгээний түүх</h3>
                <div style={{ display: "flex", gap: "0.5rem" }}>
                  <button className="btn btn-secondary" onClick={loadHistory} disabled={historyLoading}>
                    {historyLoading ? "..." : "🔄"}
                  </button>
                  <button className="btn btn-primary" onClick={runGlobalAnalysis} disabled={globalLoading}>
                    {globalLoading ? "Боловсруулж байна..." : "🌐 Нийт шинжилгээ"}
                  </button>
                </div>
              </div>

              {globalError && <p style={{ color: "var(--negative)", fontSize: "0.85rem" }}>❌ {globalError}</p>}

              {globalAnalysis && (
                <div className="card" style={{ marginBottom: "1rem" }}>
                  <div className="card-header">
                    <h4 className="card-title">🌐 Нийт {globalAnalysis.total_documents} баримт</h4>
                  </div>
                  {globalAnalysis.topic_summary?.filter((t: any) => t.topic_id >= 0).length > 0 && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.75rem" }}>
                      {globalAnalysis.topic_summary.filter((t: any) => t.topic_id >= 0).map((t: any, i: number) => (
                        <div key={i} className="card" style={{ padding: "0.5rem 0.75rem", fontSize: "0.8rem" }}>
                          <strong>{t.name || t.topic_label || `Сэдэв ${t.topic_id}`}</strong>
                          <span style={{ color: "var(--text-muted)", marginLeft: "0.4rem" }}>({t.count})</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {globalAnalysis.network && globalAnalysis.network.nodes.length > 0 && (
                    <NetworkGraph network={globalAnalysis.network} />
                  )}
                </div>
              )}

              <div className="card" style={{ padding: 0, overflow: "hidden" }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Огноо</th>
                      <th>Файл</th>
                      <th>Баримт</th>
                      <th>Сэтгэгдэл</th>
                      <th>Үйлдэл</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.length === 0 && !historyLoading && (
                      <tr><td colSpan={6} style={{ textAlign: "center", color: "var(--text-muted)" }}>Түүх олдсонгүй</td></tr>
                    )}
                    {history.map((s) => {
                      const pos = s.sentiment_summary?.positive || 0;
                      const neg = s.sentiment_summary?.negative || 0;
                      const neu = s.sentiment_summary?.neutral || 0;
                      return (
                        <tr key={s.id}>
                          <td style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>{s.id}</td>
                          <td style={{ fontSize: "0.8rem", whiteSpace: "nowrap" }}>
                            {new Date(s.created_at).toLocaleString()}
                          </td>
                          <td style={{ maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: "0.85rem" }}>
                            {s.source_filename || "—"}
                          </td>
                          <td style={{ textAlign: "center" }}>{s.total_documents}</td>
                          <td>
                            <span style={{ color: "var(--positive)", fontSize: "0.75rem" }}>+{pos} </span>
                            <span style={{ color: "var(--neutral)", fontSize: "0.75rem" }}>{neu} </span>
                            <span style={{ color: "var(--negative)", fontSize: "0.75rem" }}>-{neg}</span>
                          </td>
                          <td>
                            <div style={{ display: "flex", gap: "0.4rem" }}>
                              <button
                                className="btn btn-primary"
                                style={{ fontSize: "0.75rem", padding: "0.25rem 0.6rem" }}
                                onClick={() => openSession(s.id)}
                              >
                                Нээх
                              </button>
                              <button
                                className="btn btn-secondary"
                                style={{ fontSize: "0.75rem", padding: "0.25rem 0.6rem", color: "var(--negative)", borderColor: "var(--negative)" }}
                                onClick={() => deleteSession(s.id)}
                                disabled={deletingId === s.id}
                              >
                                {deletingId === s.id ? "..." : "🗑"}
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Reset button */}
          <div style={{ textAlign: "center", marginTop: "2rem" }}>
            <button className="btn btn-secondary" onClick={() => { setData(null); setInsights([]); setTextInput(""); setActiveTab("overview"); setGlobalAnalysis(null); }}>
              🔄 Шинэ өгөгдөл оруулах
            </button>
          </div>
        </>
      )}
    </div>
  );
}
