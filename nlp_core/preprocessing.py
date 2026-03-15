"""
preprocessing.py — Mongolian NLP preprocessing pipeline.

Two distinct modes, called from the analysis router on the SAME raw text:

  preprocess_nlp(text)  →  for NER + Sentiment Analysis
    Goal: give BERT maximum linguistic context.
    Keeps punctuation, restores capitalisation, protects name structure.
    Does NOT remove stopwords — grammar words help sentiment polarity.

  preprocess_tm(text)   →  for Topic Modeling (BERTopic)
    Goal: give BERTopic clean content-bearing tokens only.
    Aggressive: lowercase, strip all punctuation, remove stopwords.
    Keeps compound name hyphens as single tokens (бат-эрдэнэ).

Changes from the original:
  - protect_names() now handles BOTH uppercase (А.Бат) and lowercase (б.амар)
    social-media initials, and handles compound surnames with hyphens (А.Бат-Эрдэнэ)
  - clean_basic() now removes hashtags/mentions and BMP emoji (U+2000-U+2BFF etc.)
    before deep cleaning — the original passed these through to clean_deep
    where they were handled inconsistently
  - clean_deep() regex narrowed — original [А-Яа-яӨөҮүЁё-]+ allowed a trailing
    hyphen to absorb the next word. Name protection now happens in clean_basic
    (via _protect_names) so clean_deep never sees raw А.Бат forms at all
  - capitalize_for_ner() is a new function that restores sentence-start
    capitals and capitalises the initial letter in lowercase name patterns,
    fixing the core problem where б.амар wouldn't be tagged as PER
  - remove_stopwords() now also filters single-character tokens (д, т, н etc.)
  - preprocess_dual() added — returns both NLP and TM forms in one call
  - add_stopwords() added — lets main.py inject KB stopwords at startup
"""

import re
import unicodedata
from typing import List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

MONGOLIAN_PATTERN = re.compile(r"[А-Яа-яӨөҮүЁё]")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HASHTAG_MENTION = re.compile(r"[@#]\S+")

# BMP emoji/symbol blocks missed by the original \U00010000-\U0010ffff range
BMP_EMOJI = re.compile(
    r"[\u2000-\u27FF\u2900-\u2BFF\uFE00-\uFEFF\uFF00-\uFFEF]"
)
SUPPLEMENTARY_EMOJI = re.compile(r"[\U00010000-\U0010FFFF]")

# Uppercase Mongolian initial: А.Бат-Эрдэнэ, Б.Сувдаа
MN_NAME_UPPER = re.compile(
    r"\b([А-ЯӨҮЁ])\.\s*"
    r"([А-Яа-яӨөҮүЁё][а-яөүёa-z]+"
    r"(?:-[А-Яа-яӨөҮүЁё][а-яөүёa-z]+)*)"
)

# Lowercase initial: б.амар, о.батзориг  (very common in social media)
# ORIGINAL CODE HAD NO HANDLING FOR THIS — it only matched [А-ЯӨҮЁ]
MN_NAME_LOWER = re.compile(
    r"\b([а-яөүё])\.\s*"
    r"([а-яөүёa-z]+"
    r"(?:-[а-яөүёa-z]+)*)"
)

# Protected form А_Бат-Эрдэнэ — matched by restore_names()
MN_NAME_PROTECTED = re.compile(
    r"\b([А-ЯӨҮЁ])_([А-Яа-яӨөҮүЁё][а-яөүёa-z]+(?:-[А-Яа-яӨөҮүЁё][а-яөүёa-z]+)*)\b"
)

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[А-ЯӨҮЁ\u0400-\u04FF]|[A-Z])")


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

MONGOLIAN_STOPWORDS: Set[str] = {
    "ба", "бас", "бол", "бөгөөд", "байна", "байгаа", "байсан", "бсан",
    "бхаа", "бн", "бна", "байх", "юм", "биш", "бгаа", "бдаг", "байдаг",
    "бхоо", "бх",
    "энэ", "тэр", "эдгээр", "тэдгээр", "үүн", "үүнд", "үүнээс", "үүний",
    "үүнтэй", "түүн", "түүнд", "түүнээс", "түүний", "түүнтэй",
    "тийм", "ийм", "чинь", "минь", "билээ", "шүү",
    "би", "чи", "та", "бид", "тэд", "миний", "чиний", "таны", "бидний",
    "тэдний", "над", "надад", "надаас", "чамд", "чамаас", "танд", "танаас",
    "өөр", "өөрөө", "өөрийн", "өөрт", "өөрөөс",
    "гэж", "гэх", "гэсэн", "гэжээ", "гэв", "гэвч", "гээд", "гнээ",
    "гэнэ", "гээ",
    "л", "ч", "уу", "үү", "юу", "яаж", "яагаад",
    "хаана", "хэзээ", "хэн", "ямар", "ямарч", "яах", "вэ", "бэ", "бээ",
    "болон", "мөн", "эсвэл", "гэхдээ", "харин",
    "дээр", "доор", "дотор", "гадна", "хойно", "өмнө",
    "руу", "рүү", "аас", "ээс", "оос", "өөс", "тай", "тэй", "той",
    "д", "т", "нь", "аа", "ээ", "оо", "өө",
    "бай", "болно", "болох", "болсон",
    "их", "бага", "маш", "тун", "нэлээд", "шиг",
    "шд", "н", "шдэ", "шдээ", "шт", "штэ", "штээ", "ш дээ", "ш тээ",
    "бз", "биз", "дээ", "даа", "юмаа", "аан", "хө", "тэ", "тээ",
    "гш", "ммхн", "сдаа", "сда", "хаха", "кк",
    "гэх", "хийх", "авах", "өгөх", "очих", "ирэх",
    "ын", "ийн", "ний", "ийг", "ууд", "үүд",
    "та нар",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_unicode(text: str) -> str:
    """NFC-normalize so visually identical Mongolian characters match regexes."""
    return unicodedata.normalize("NFC", text)


def _remove_emoji(text: str) -> str:
    """Remove BMP symbol blocks and supplementary-plane emoji."""
    text = BMP_EMOJI.sub(" ", text)
    text = SUPPLEMENTARY_EMOJI.sub(" ", text)
    return text


def _protect_names(text: str) -> str:
    """
    Convert Mongolian name patterns to protected underscore form before
    any cleaning strips the dots or hyphens.

    WHY UNDERSCORE:
      The character-whitelist in clean_deep preserves [_] and [-], so both
      the initial-name join AND the compound-surname hyphen survive.

    WHAT CHANGED FROM ORIGINAL:
      Original clean_deep had:
        re.sub(r'\b([А-ЯӨҮЁ])\.\s*([А-Яа-яӨөҮүЁё-]+)', r'\1_\2', text)
      Problems:
        1. Only runs INSIDE clean_deep, after clean_basic already ran.
           If the text came through preprocess_nlp (which never calls
           clean_deep), name dots were never protected.
        2. [А-Яа-яӨөҮүЁё-]+ has a trailing hyphen IN the character class
           which means it matches the hyphen character anywhere, including
           at the end of a word, potentially absorbing the next token.
        3. Only matched uppercase initials — б.амар was completely missed.

      This version runs _protect_names() inside clean_basic() so it fires
      for BOTH NLP and TM pipelines, before any stripping occurs.
    """
    def _replace_upper(m: re.Match) -> str:
        initial = m.group(1)
        name = "-".join(p.capitalize() for p in m.group(2).split("-"))
        return f"{initial}_{name}"

    def _replace_lower(m: re.Match) -> str:
        # Capitalize the single-letter initial and each name part
        initial = m.group(1).upper()
        name = "-".join(p.capitalize() for p in m.group(2).split("-"))
        return f"{initial}_{name}"

    text = MN_NAME_UPPER.sub(_replace_upper, text)
    text = MN_NAME_LOWER.sub(_replace_lower, text)
    return text


def _restore_names(text: str) -> str:
    """Undo protection: А_Бат-Эрдэнэ → А.Бат-Эрдэнэ (NLP mode only)."""
    return MN_NAME_PROTECTED.sub(lambda m: f"{m.group(1)}.{m.group(2)}", text)


def _capitalize_for_ner(text: str) -> str:
    """
    Heuristic capitalisation for NER on social-media Mongolian text.

    WHY THIS IS NEEDED:
      Davlan/bert-base-multilingual-cased-ner-hrl is a CASED model — it uses
      capitalisation as a primary signal to identify proper nouns. Mongolian
      social media is frequently written entirely lowercase. Without this step,
      "монгол улсын ерөнхийлөгч х.баттулга" will not tag х.баттулга as PER
      because the model sees it as an ordinary lowercase word.

    WHAT THIS DOES:
      1. Capitalises the first word of each detected sentence.
      2. Capitalises the name component inside protected tokens:
         Б_амар → Б_Амар (the initial is already uppercase from _protect_names,
         but the name itself may still be lowercase if it came from the lower
         pattern and capitalize() didn't fire — this is a safety pass).

    WHAT THIS DOES NOT DO:
      - Does NOT blindly capitalise all words (that would confuse common nouns)
    """
    sentences = SENTENCE_BOUNDARY.split(text)
    sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
    text = " ".join(sentences)

    # Fix any protected token where name part is still lowercase
    text = re.sub(
        r"([А-ЯӨҮЁ])_([а-яөүё])([а-яөүё]*)",
        lambda m: f"{m.group(1)}_{m.group(2).upper()}{m.group(3)}",
        text,
    )
    return text


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Text preprocessing pipeline for Mongolian social media data.

    Initialise once at app startup with stopwords from the knowledge base:

        kb = KnowledgeBase()
        preprocessor = Preprocessor(extra_stopwords=kb.get_stopwords())

    Then in the analysis router, call preprocess_dual() per document:

        nlp_text, tm_text = preprocessor.preprocess_dual(raw_text)
        entities  = ner_engine.recognize(nlp_text)
        sentiment = sentiment_analyzer.analyze(nlp_text)
        tm_texts.append(tm_text)

    After all documents:
        if len(tm_texts) >= 10:  # BERTopic minimum — skip below this
            topic_results, summary = topic_modeler.fit_transform(tm_texts)
    """

    def __init__(self, extra_stopwords: Optional[List[str]] = None):
        self.stopwords: Set[str] = MONGOLIAN_STOPWORDS.copy()
        if extra_stopwords:
            self.stopwords.update(w.lower().strip() for w in extra_stopwords)

    def add_stopwords(self, words: List[str]) -> None:
        """
        Inject additional stopwords at runtime (e.g. after admin adds one).
        Call preprocessor.add_stopwords(kb.get_stopwords()) when the admin
        saves a new stopword — takes effect on the next analysis request
        without restarting the server.
        """
        self.stopwords.update(w.lower().strip() for w in words)

    def is_mongolian(self, text: str) -> bool:
        return isinstance(text, str) and bool(MONGOLIAN_PATTERN.search(text))

    # ------------------------------------------------------------------
    # clean_basic
    # ------------------------------------------------------------------

    def clean_basic(self, text: str, replace_url: bool = True) -> str:
        """
        Light surface cleaning.

        CHANGES FROM ORIGINAL:
          Original: only handled URLs and whitespace normalisation.
          Updated:
            1. Unicode NFC normalisation added (first step — must precede regex)
            2. _protect_names() called here so it fires for BOTH pipelines.
               Original had protection only inside clean_deep() which is only
               called in TM mode — NLP mode had no name protection at all.
            3. Hashtag/mention removal added. Original left @user and #tag
               in the text; in TM mode these became bare tokens like "монгол"
               (from #монгол) with artificially inflated frequency.
            4. BMP emoji removal added via _remove_emoji(). Original only
               removed supplementary-plane emoji and only inside clean_deep().

        Args:
            replace_url: True = replace with [URL] token (NLP mode needs the
                         signal that a URL was present for sentiment context).
                         False = remove entirely (TM mode — URL adds no topic).
        """
        if not isinstance(text, str):
            return ""

        text = _normalize_unicode(text)
        text = _protect_names(text)          # must be before any dot/hyphen stripping

        if replace_url:
            text = URL_PATTERN.sub("[URL]", text)
        else:
            text = URL_PATTERN.sub("", text)

        text = HASHTAG_MENTION.sub("", text)
        text = _remove_emoji(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # clean_deep
    # ------------------------------------------------------------------

    def clean_deep(self, text: str) -> str:
        """
        Aggressive symbol/punctuation removal for TM mode.

        CHANGES FROM ORIGINAL:
          Original had THREE issues inside this function:
          1. Name protection regex: r'\b([А-ЯӨҮЁ])\.\s*([А-Яа-яӨөҮүЁё-]+)'
             The character class [А-Яа-яӨөҮүЁё-]+ includes a hyphen at the
             END of the class which makes it match ANY hyphen, including
             standalone hyphens at word boundaries. This could join the
             protected name to the next word.
             FIX: Name protection is fully removed from here — it now happens
             in _protect_names() called inside clean_basic(). By the time
             clean_deep() runs, А.Бат-Эрдэнэ is already А_Бат-Эрдэнэ and
             the character whitelist below preserves both _ and -.

          2. Uppercase-only matching: the original only protected [А-ЯӨҮЁ]
             initials. Lowercase б.амар was left unprotected.
             FIX: Handled by _protect_names() as above.

          3. Emoji removal: original had re.sub(r'[\U00010000-\U0010ffff]', '', text)
             here, missing BMP symbols.
             FIX: Moved to _remove_emoji() inside clean_basic() which runs first.

        What this function now does:
          - Remove [URL] placeholder if still present
          - Apply character whitelist: keep Mongolian Cyrillic, Latin, digits,
            spaces, underscores (protected name joins), and hyphens
            (compound surname separators inside protected names)
          - Normalise whitespace
        """
        if not isinstance(text, str):
            return ""

        # Remove [URL] placeholder
        text = re.sub(r"\[URL\]", "", text)

        # Character whitelist — everything outside this becomes a space
        # _ preserved: А_Бат protected form
        # - preserved: Бат-Эрдэнэ compound name (inside protected token)
        text = re.sub(
            r"[^A-Za-zА-Яа-яӨөҮүЁё0-9\s_\-]",
            " ",
            text,
        )
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Stopword removal
    # ------------------------------------------------------------------

    def _remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from lowercased text.

        CHANGE FROM ORIGINAL:
        Added len(w) > 1 filter. Single-character Mongolian tokens (д, т, н,
        ч, л, etc.) are case inflections and particles written as separate
        words in informal text. They are effectively stopwords and pollute
        topic model vocabulary. The original code left them in.
        """
        if not isinstance(text, str):
            return ""
        words = text.split()
        return " ".join(
            w for w in words
            if len(w) > 1 and w.lower() not in self.stopwords
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess_nlp(self, text: str) -> str:
        """
        Preprocessing for NER and Sentiment Analysis.

        Pipeline: clean_basic → capitalize_for_ner → restore_names
        Intentionally skips: clean_deep, lowercasing, stopword removal
        """
        if not isinstance(text, str):
            return ""
        text = self.clean_basic(text, replace_url=True)
        text = _capitalize_for_ner(text)
        text = _restore_names(text)
        return text

    def preprocess_tm(self, text: str) -> str:
        """
        Preprocessing for Topic Modeling (BERTopic).

        Pipeline: clean_basic → clean_deep → lowercase → remove_stopwords
                  → strip initial prefix

        Why strip the initial prefix in TM mode:
          After lowercasing, А_Бат-Эрдэнэ becomes а_бат-эрдэнэ.
          The single letter а adds nothing to topic clusters — the meaningful
          token is бат-эрдэнэ (the surname, treated as one compound token).
          The regex below strips the initial and underscore, keeping the name.
        """
        if not isinstance(text, str):
            return ""
        text = self.clean_basic(text, replace_url=False)
        text = self.clean_deep(text)
        text = text.lower()
        text = self._remove_stopwords(text)
        # Strip single-letter initial prefix: а_батэрдэнэ → батэрдэнэ
        text = re.sub(
            r"\b[а-яөүё]_([а-яөүёa-z]+(?:-[а-яөүёa-z]+)*)\b",
            r"\1",
            text,
        )
        return re.sub(r"\s+", " ", text).strip()

    def preprocess_dual(self, text: str) -> Tuple[str, str]:
        """
        Return both NLP and TM forms in one call.

        Use this in the router to avoid processing the same text twice:
            nlp_text, tm_text = preprocessor.preprocess_dual(raw)
        """
        return self.preprocess_nlp(text), self.preprocess_tm(text)

    def split_sentences(self, text: str) -> List[str]:
        """
        Split NLP-preprocessed text into sentences for chunked NER.
        Useful when a document exceeds BERT's 512-token limit.
        """
        parts = SENTENCE_BOUNDARY.split(text)
        return [p.strip() for p in parts if p.strip()]

    def preprocess_batch(self, texts: List[str], mode: str = "nlp") -> List[str]:
        """
        Preprocess a list of texts in the given mode ("nlp" or "tm").
        Returns a list of the same length.
        """
        fn = self.preprocess_tm if mode == "tm" else self.preprocess_nlp
        return [fn(t) for t in texts]