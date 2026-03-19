"use client";

import { useState, useEffect, useCallback } from "react";

//const API_BASE = "http://localhost:8000";
const API_BASE = "";

interface KnowledgeEntry {
  id: number;
  word: string;
  category: string;
  entity_type: string;
  synonyms: string[];
}

export default function AdminPage() {
  const [entries, setEntries] = useState<KnowledgeEntry[]>([]);
  const [stopwords, setStopwords] = useState<string[]>([]);
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [activeSection, setActiveSection] = useState<"knowledge" | "stopwords" | "labels">("knowledge");

  // Form states
  const [newWord, setNewWord] = useState("");
  const [newCategory, setNewCategory] = useState("");
  const [newEntityType, setNewEntityType] = useState("");
  const [newStopword, setNewStopword] = useState("");
  const [newOriginalLabel, setNewOriginalLabel] = useState("");
  const [newCustomLabel, setNewCustomLabel] = useState("");

  const [loading, setLoading] = useState(false);

  const fetchEntries = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/knowledge`);
      if (res.ok) setEntries(await res.json());
    } catch { /* server might not be running */ }
  }, []);

  const fetchStopwords = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/stopwords`);
      if (res.ok) setStopwords(await res.json());
    } catch { /* server might not be running */ }
  }, []);

  const fetchLabels = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/labels`);
      if (res.ok) setLabels(await res.json());
    } catch { /* server might not be running */ }
  }, []);

  useEffect(() => {
    fetchEntries();
    fetchStopwords();
    fetchLabels();
  }, [fetchEntries, fetchStopwords, fetchLabels]);

  const addEntry = async () => {
    if (!newWord.trim()) return;
    setLoading(true);
    try {
      await fetch(`${API_BASE}/api/admin/knowledge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          word: newWord,
          category: newCategory,
          entity_type: newEntityType,
          synonyms: [],
        }),
      });
      setNewWord("");
      setNewCategory("");
      setNewEntityType("");
      fetchEntries();
    } finally {
      setLoading(false);
    }
  };

  const deleteEntry = async (id: number) => {
    await fetch(`${API_BASE}/api/admin/knowledge/${id}`, { method: "DELETE" });
    fetchEntries();
  };

  const addStopword = async () => {
    if (!newStopword.trim()) return;
    await fetch(`${API_BASE}/api/admin/stopwords`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ word: newStopword }),
    });
    setNewStopword("");
    fetchStopwords();
  };

  const deleteStopword = async (word: string) => {
    await fetch(`${API_BASE}/api/admin/stopwords/${word}`, { method: "DELETE" });
    fetchStopwords();
  };

  const addLabel = async () => {
    if (!newOriginalLabel.trim() || !newCustomLabel.trim()) return;
    await fetch(`${API_BASE}/api/admin/labels`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        original_label: newOriginalLabel,
        custom_label: newCustomLabel,
        label_type: "entity",
      }),
    });
    setNewOriginalLabel("");
    setNewCustomLabel("");
    fetchLabels();
  };

  return (
    <div>
      <h2 style={{ fontSize: "1.5rem", fontWeight: 700, marginBottom: "0.5rem" }}>
        ⚙️ Admin — Мэдлэгийн сан удирдлага
      </h2>
      <p style={{ color: "var(--text-muted)", marginBottom: "1.5rem", fontSize: "0.875rem" }}>
        Нэрлэсэн объектууд, сэдвийн шошго, зогсох үгсийг хянах, нэмэх, засах
      </p>

      {/* Section tabs */}
      <div className="tabs">
        <button
          className={`tab ${activeSection === "knowledge" ? "active" : ""}`}
          onClick={() => setActiveSection("knowledge")}
        >
          📚 Мэдлэгийн сан
        </button>
        <button
          className={`tab ${activeSection === "labels" ? "active" : ""}`}
          onClick={() => setActiveSection("labels")}
        >
          🏷️ Шошго
        </button>
        <button
          className={`tab ${activeSection === "stopwords" ? "active" : ""}`}
          onClick={() => setActiveSection("stopwords")}
        >
          🚫 Зогсох үгс
        </button>
      </div>

      {/* Knowledge Base */}
      {activeSection === "knowledge" && (
        <div className="admin-grid">
          <div className="card">
            <h3 className="card-title" style={{ marginBottom: "1rem" }}>Шинэ нэмэх</h3>
            <div className="form-group">
              <label className="form-label">Үг / нэр</label>
              <input className="form-input" value={newWord} onChange={(e) => setNewWord(e.target.value)} placeholder="жиш: Улаанбаатар" />
            </div>
            <div className="form-group">
              <label className="form-label">Ангилал</label>
              <input className="form-input" value={newCategory} onChange={(e) => setNewCategory(e.target.value)} placeholder="жиш: Газар зүй" />
            </div>
            <div className="form-group">
              <label className="form-label">Entity төрөл</label>
              <select className="form-input" value={newEntityType} onChange={(e) => setNewEntityType(e.target.value)}>
                <option value="">Сонгох</option>
                <option value="PER">PER — Хүн</option>
                <option value="ORG">ORG — Байгууллага</option>
                <option value="LOC">LOC — Байршил</option>
                <option value="MISC">MISC — Бусад</option>
              </select>
            </div>
            <button className="btn btn-primary" onClick={addEntry} disabled={loading}>
              {loading ? "Нэмж байна..." : "+ Нэмэх"}
            </button>
          </div>

          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Мэдлэгийн сан ({entries.length})</h3>
            </div>
            {entries.length === 0 ? (
              <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                Хоосон байна. Зүүн талаас нэмнэ үү.
              </p>
            ) : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Үг</th>
                    <th>Ангилал</th>
                    <th>Төрөл</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {entries.map((e) => (
                    <tr key={e.id}>
                      <td>{e.word}</td>
                      <td>{e.category}</td>
                      <td><span className={`entity-tag ${e.entity_type}`} style={{ fontSize: "0.7rem" }}>{e.entity_type}</span></td>
                      <td>
                        <button className="btn btn-danger btn-sm" onClick={() => deleteEntry(e.id)}>✕</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}

      {/* Labels */}
      {activeSection === "labels" && (
        <div className="admin-grid">
          <div className="card">
            <h3 className="card-title" style={{ marginBottom: "1rem" }}>Шошго нэмэх</h3>
            <div className="form-group">
              <label className="form-label">Анхны шошго</label>
              <input className="form-input" value={newOriginalLabel} onChange={(e) => setNewOriginalLabel(e.target.value)} placeholder="жиш: Topic_0" />
            </div>
            <div className="form-group">
              <label className="form-label">Шинэ шошго</label>
              <input className="form-input" value={newCustomLabel} onChange={(e) => setNewCustomLabel(e.target.value)} placeholder="жиш: Улс төр" />
            </div>
            <button className="btn btn-primary" onClick={addLabel}>+ Нэмэх</button>
          </div>
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Шошгоны жагсаалт</h3>
            </div>
            {Object.keys(labels).length === 0 ? (
              <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>Шошго нэмэгдээгүй байна.</p>
            ) : (
              <table className="data-table">
                <thead>
                  <tr><th>Анхны</th><th>Шинэ</th></tr>
                </thead>
                <tbody>
                  {Object.entries(labels).map(([orig, custom]) => (
                    <tr key={orig}>
                      <td style={{ color: "var(--text-muted)" }}>{orig}</td>
                      <td>{custom}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}

      {/* Stopwords */}
      {activeSection === "stopwords" && (
        <div className="admin-grid">
          <div className="card">
            <h3 className="card-title" style={{ marginBottom: "1rem" }}>Зогсох үг нэмэх</h3>
            <div className="form-group">
              <label className="form-label">Үг</label>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <input className="form-input" value={newStopword} onChange={(e) => setNewStopword(e.target.value)} placeholder="жиш: аан" />
                <button className="btn btn-primary" onClick={addStopword}>+ Нэмэх</button>
              </div>
            </div>
          </div>
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Зогсох үгс ({stopwords.length})</h3>
            </div>
            <div className="entity-list">
              {stopwords.map((w) => (
                <span className="entity-tag MISC" key={w} style={{ cursor: "pointer" }} onClick={() => deleteStopword(w)} title="Дарж устгах">
                  {w} ✕
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
