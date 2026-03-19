"use client";

import { useState, useCallback, useRef } from "react";

//const API_BASE = "http://localhost:8000";
const API_BASE = "";

export interface Entity {
  word: string;
  entity_group: string;
  score: number;
  start?: number | null;
  end?: number | null;
}

export interface DocForEditor {
  doc_id: number;
  text: string;
  entities: Entity[];
  sentiment: { label: string; score: number } | null;
}

interface Props {
  doc: DocForEditor;
  onClose: () => void;
  onSaved?: (updated: DocForEditor) => void;
}

const ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"];

const ENTITY_COLORS: Record<string, string> = {
  PER: "#ff6b6b",
  ORG: "#4ecdc4",
  LOC: "#ffd93d",
  MISC: "#a78bfa",
};

const SENTIMENT_LABELS = ["positive", "neutral", "negative"] as const;
const SENTIMENT_MN: Record<string, string> = {
  positive: "Эерэг",
  neutral: "Саармаг",
  negative: "Сөрөг",
};

function color(group: string) {
  return ENTITY_COLORS[group] || "#6c63ff";
}

// Split text into plain/entity segments using char offsets
type Segment =
  | { kind: "text"; text: string }
  | { kind: "entity"; text: string; index: number; entity: Entity };

function buildSegments(text: string, entities: Entity[]): Segment[] {
  // Only use entities that have valid start/end offsets
  const valid = entities
    .map((e, index) => ({ e, index }))
    .filter(({ e }) => e.start != null && e.end != null && e.start >= 0 && e.end > e.start)
    .sort((a, b) => (a.e.start ?? 0) - (b.e.start ?? 0));

  if (valid.length === 0) return [{ kind: "text", text }];

  const segments: Segment[] = [];
  let cursor = 0;

  for (const { e, index } of valid) {
    const start = e.start ?? 0;
    const end = e.end ?? 0;
    if (start < cursor) continue; // overlapping span — skip
    if (start > cursor) {
      segments.push({ kind: "text", text: text.slice(cursor, start) });
    }
    segments.push({ kind: "entity", text: text.slice(start, end), index, entity: e });
    cursor = end;
  }
  if (cursor < text.length) {
    segments.push({ kind: "text", text: text.slice(cursor) });
  }
  return segments;
}

export default function AnnotationEditor({ doc, onClose, onSaved }: Props) {
  const [entities, setEntities] = useState<Entity[]>(doc.entities);
  const [sentimentLabel, setSentimentLabel] = useState(
    doc.sentiment?.label ?? "neutral"
  );
  const [sentimentScore, setSentimentScore] = useState(
    doc.sentiment?.score ?? 1.0
  );
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");

  // Popover state for entity actions
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

  // New entity from text selection
  const [newEntityPopover, setNewEntityPopover] = useState<{
    selection: string;
    start: number;
    end: number;
  } | null>(null);
  const [newEntityGroup, setNewEntityGroup] = useState("PER");
  const textRef = useRef<HTMLDivElement>(null);

  const handleEntityClick = (idx: number) => {
    setActiveIdx(activeIdx === idx ? null : idx);
    setNewEntityPopover(null);
  };

  const changeLabel = (idx: number, group: string) => {
    setEntities((prev) =>
      prev.map((e, i) => (i === idx ? { ...e, entity_group: group } : e))
    );
    setActiveIdx(null);
  };

  const removeEntity = (idx: number) => {
    setEntities((prev) => prev.filter((_, i) => i !== idx));
    setActiveIdx(null);
  };

  const handleTextMouseUp = useCallback(() => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !textRef.current) return;
    const range = sel.getRangeAt(0);

    // Walk the text container to compute char offset
    const container = textRef.current;
    let start = 0;
    let end = 0;
    let charCount = 0;
    let foundStart = false;

    function walk(node: Node) {
      if (node.nodeType === Node.TEXT_NODE) {
        const len = node.textContent?.length ?? 0;
        if (!foundStart && node === range.startContainer) {
          start = charCount + range.startOffset;
          foundStart = true;
        }
        if (node === range.endContainer) {
          end = charCount + range.endOffset;
        }
        charCount += len;
      } else {
        node.childNodes.forEach(walk);
      }
    }
    walk(container);

    const selectedText = sel.toString().trim();
    if (!selectedText || start === end) return;
    sel.removeAllRanges();
    setNewEntityPopover({ selection: selectedText, start, end });
    setActiveIdx(null);
  }, []);

  const addNewEntity = () => {
    if (!newEntityPopover) return;
    setEntities((prev) => [
      ...prev,
      {
        word: newEntityPopover.selection,
        entity_group: newEntityGroup,
        score: 1.0,
        start: newEntityPopover.start,
        end: newEntityPopover.end,
      },
    ]);
    setNewEntityPopover(null);
  };

  const save = async () => {
    setSaving(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/documents/${doc.doc_id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          entities,
          sentiment_label: sentimentLabel,
          sentiment_score: sentimentScore,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Save failed");
      }
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
      onSaved?.({
        ...doc,
        entities,
        sentiment: { label: sentimentLabel, score: sentimentScore },
      });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  const segments = buildSegments(doc.text, entities);
  const hasOffsets = entities.some((e) => e.start != null && e.end != null);

  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.55)",
      zIndex: 1000, display: "flex", alignItems: "center", justifyContent: "center",
    }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div style={{
        background: "var(--bg-card)", borderRadius: 12, padding: "1.5rem",
        width: "min(860px, 96vw)", maxHeight: "90vh", overflow: "auto",
        boxShadow: "0 8px 40px rgba(0,0,0,0.4)", position: "relative",
      }}>
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
          <h3 style={{ margin: 0, fontSize: "1rem" }}>Тэмдэглэл засах — Doc #{doc.doc_id}</h3>
          <button className="btn btn-secondary" style={{ padding: "0.3rem 0.8rem" }} onClick={onClose}>✕</button>
        </div>

        {/* Sentiment selector */}
        <div style={{ marginBottom: "1rem", display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Сэтгэгдэл:</span>
          {SENTIMENT_LABELS.map((lbl) => (
            <button
              key={lbl}
              onClick={() => { setSentimentLabel(lbl); setSentimentScore(1.0); }}
              className={`badge badge-${lbl}`}
              style={{
                cursor: "pointer", border: "none", padding: "0.3rem 0.8rem",
                opacity: sentimentLabel === lbl ? 1 : 0.35,
                fontWeight: sentimentLabel === lbl ? 700 : 400,
              }}
            >
              {SENTIMENT_MN[lbl]}
            </button>
          ))}
        </div>

        {/* Entity legend */}
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "0.75rem" }}>
          {ENTITY_TYPES.map((t) => (
            <span key={t} style={{
              fontSize: "0.72rem", padding: "0.15rem 0.5rem", borderRadius: 4,
              background: `${color(t)}20`, border: `1px solid ${color(t)}`, color: color(t)
            }}>
              {t}
            </span>
          ))}
          <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", alignSelf: "center" }}>
            — текст сонгоход шинэ нэрлэсэн объект нэмнэ
          </span>
        </div>

        {/* Annotated text */}
        <div
          ref={textRef}
          onMouseUp={handleTextMouseUp}
          style={{
            lineHeight: 2.2, fontSize: "0.95rem", padding: "1rem",
            background: "var(--bg-hover)", borderRadius: 8, marginBottom: "1rem",
            userSelect: "text", cursor: "text", position: "relative",
          }}
        >
          {hasOffsets ? (
            segments.map((seg, si) =>
              seg.kind === "text" ? (
                <span key={si}>{seg.text}</span>
              ) : (
                <span key={si} style={{ position: "relative", display: "inline" }}>
                  <mark
                    onClick={() => handleEntityClick(seg.index)}
                    style={{
                      background: `${color(seg.entity.entity_group)}25`,
                      borderBottom: `2.5px solid ${color(seg.entity.entity_group)}`,
                      borderRadius: "3px 3px 0 0",
                      padding: "0 2px", cursor: "pointer",
                      color: "inherit",
                    }}
                  >
                    {seg.text}
                    <sup style={{
                      fontSize: "0.6rem", marginLeft: 2, fontWeight: 700,
                      color: color(seg.entity.entity_group),
                    }}>
                      [{seg.entity.entity_group}]
                    </sup>
                  </mark>
                  {/* Entity action popover */}
                  {activeIdx === seg.index && (
                    <span
                      ref={popoverRef}
                      style={{
                        position: "absolute", zIndex: 10, top: "100%", left: 0,
                        background: "var(--bg-card)", border: "1px solid var(--border)",
                        borderRadius: 8, padding: "0.5rem", boxShadow: "0 4px 16px rgba(0,0,0,0.3)",
                        display: "flex", flexDirection: "column", gap: "0.3rem", minWidth: 140,
                      }}
                    >
                      <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: 2 }}>
                        Шошго өөрчлөх:
                      </span>
                      <span style={{ display: "flex", flexWrap: "wrap", gap: "0.25rem" }}>
                        {ENTITY_TYPES.map((t) => (
                          <button
                            key={t}
                            onClick={() => changeLabel(seg.index, t)}
                            style={{
                              fontSize: "0.7rem", padding: "0.2rem 0.5rem", borderRadius: 4,
                              background: `${color(t)}20`, border: `1px solid ${color(t)}`,
                              color: color(t), cursor: "pointer",
                              fontWeight: seg.entity.entity_group === t ? 700 : 400,
                            }}
                          >
                            {t}
                          </button>
                        ))}
                      </span>
                      <button
                        onClick={() => removeEntity(seg.index)}
                        style={{
                          fontSize: "0.7rem", padding: "0.2rem 0.5rem", borderRadius: 4,
                          background: "var(--negative)20", border: "1px solid var(--negative)",
                          color: "var(--negative)", cursor: "pointer", marginTop: 2,
                        }}
                      >
                        🗑 Устгах
                      </button>
                    </span>
                  )}
                </span>
              )
            )
          ) : (
            /* Fallback: no offsets — show raw text + entity list below */
            <span>{doc.text}</span>
          )}
        </div>

        {/* New entity popover from selection */}
        {newEntityPopover && (
          <div style={{
            background: "var(--bg-hover)", border: "1px solid var(--border)",
            borderRadius: 8, padding: "0.75rem", marginBottom: "1rem",
            display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap",
          }}>
            <span style={{ fontSize: "0.8rem" }}>
              Нэмэх: <strong>&ldquo;{newEntityPopover.selection}&rdquo;</strong>
            </span>
            <select
              value={newEntityGroup}
              onChange={(e) => setNewEntityGroup(e.target.value)}
              style={{
                fontSize: "0.8rem", padding: "0.2rem 0.4rem",
                background: "var(--bg-card)", border: "1px solid var(--border)",
                color: "var(--text-primary)", borderRadius: 4
              }}
            >
              {ENTITY_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
            <button className="btn btn-primary" style={{ fontSize: "0.8rem", padding: "0.25rem 0.75rem" }}
              onClick={addNewEntity}>Нэмэх</button>
            <button className="btn btn-secondary" style={{ fontSize: "0.8rem", padding: "0.25rem 0.75rem" }}
              onClick={() => setNewEntityPopover(null)}>Болих</button>
          </div>
        )}

        {/* Entity list (always shown as summary) */}
        {!hasOffsets && entities.length > 0 && (
          <div style={{ marginBottom: "1rem" }}>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "0.5rem" }}>
              Нэрлэсэн объектууд (засахын тулд текст дээр дарна уу):
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
              {entities.map((e, i) => (
                <span
                  key={i}
                  onClick={() => removeEntity(i)}
                  title="Устгах"
                  style={{
                    padding: "0.25rem 0.6rem", borderRadius: 6, cursor: "pointer",
                    background: `${color(e.entity_group)}20`,
                    border: `1px solid ${color(e.entity_group)}`,
                    color: color(e.entity_group), fontSize: "0.8rem",
                    display: "flex", alignItems: "center", gap: 4,
                  }}
                >
                  {e.word}
                  <sup style={{ fontSize: "0.6rem" }}>[{e.entity_group}]</sup>
                  <span style={{ opacity: 0.5, fontSize: "0.7rem" }}>✕</span>
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Error */}
        {error && <p style={{ color: "var(--negative)", fontSize: "0.85rem", marginBottom: "0.5rem" }}>❌ {error}</p>}

        {/* Actions */}
        <div style={{ display: "flex", gap: "0.75rem", justifyContent: "flex-end", alignItems: "center" }}>
          {saved && <span style={{ color: "var(--positive)", fontSize: "0.85rem" }}>✓ Хадгалагдлаа</span>}
          <button className="btn btn-secondary" onClick={onClose}>Болих</button>
          <button className="btn btn-primary" onClick={save} disabled={saving}>
            {saving ? "Хадгалж байна..." : "💾 Хадгалах"}
          </button>
        </div>
      </div>
    </div>
  );
}
