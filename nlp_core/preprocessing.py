"""
Preprocessing service — ported from the user's existing ppForTopicModeling.py and preprocess.py.
Handles Mongolian text cleaning, stopword removal, and language detection.
"""

import re
from typing import List

# Mongolian Cyrillic character detection
MONGOLIAN_PATTERN = re.compile(r'[А-Яа-яӨөҮүЁё]')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')

# Mongolian stopwords (from the user's existing list)
MONGOLIAN_STOPWORDS = {
    "ба", "бас", "бол", "бөгөөд", "байна", "байгаа", "байсан", "бсан",
    "бхаа", "бн", "бна", "байх", "юм", "биш", "бгаа", "бдаг", "байдаг", "бхоо", "бх",
    "энэ", "тэр", "эдгээр", "тэдгээр", "үүн", "үүнд", "үүнээс", "үүний", "үүнтэй",
    "түүн", "түүнд", "түүнээс", "түүний", "түүнтэй",
    "тийм", "ийм", "чинь", "минь", "билээ", "шүү",
    "би", "чи", "та", "бид", "та нар", "тэд", "миний", "чиний", "таны", "бидний", "тэдний",
    "над", "надад", "надаас", "чамд", "чамаас", "танд", "танаас",
    "өөр", "өөрөө", "өөрийн", "өөрт", "өөрөөс",
    "гэж", "гэх", "гэсэн", "гэжээ", "гэв", "гэвч", "гээд", "гнээ", "гэнэ", "гээ",
    "л", "ч", "уу", "үү", "юу", "яаж", "яагаад",
    "хаана", "хэзээ", "хэн", "ямар", "ямарч", "яах", "вэ", "бэ", "бээ",
    "болон", "мөн", "эсвэл", "гэхдээ", "харин",
    "дээр", "доор", "дотор", "гадна", "хойно", "өмнө",
    "руу", "рүү", "аас", "ээс", "оос", "өөс", "тай", "тэй", "той",
    "д", "т",
    "нь", "аа", "ээ", "оо", "өө",
    "бай", "болно", "болох", "болсон",
    "их", "бага", "маш", "тун", "нэлээд", "шиг",
    "шд", "н", "шдэ", "шдээ", "шт", "штэ", "штээ", "ш дээ", "ш тээ", "бз", "биз", "дээ", "даа",
    "юмаа", "аан", "хө", "тэ", "тээ", "гш", "ммхн", "сдаа", "сда",
    "хаха", "кк",
    "гэх", "хийх", "авах", "өгөх", "очих", "ирэх",
    "ын", "ийн", "ний", "ийг", "ууд", "үүд",
}


class Preprocessor:
    """Text preprocessing pipeline for Mongolian social media data."""

    def __init__(self, extra_stopwords: List[str] = None):
        self.stopwords = MONGOLIAN_STOPWORDS.copy()
        if extra_stopwords:
            self.stopwords.update(extra_stopwords)

    def is_mongolian(self, text: str) -> bool:
        """Check if the text contains Mongolian Cyrillic characters."""
        if not isinstance(text, str):
            return False
        return bool(MONGOLIAN_PATTERN.search(text))

    def clean_basic(self, text: str) -> str:
        """Basic cleaning: remove URLs, normalize whitespace."""
        if not isinstance(text, str):
            return ""
        text = URL_PATTERN.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def clean_deep(self, text: str) -> str:
        """Deep cleaning: remove links, symbols, emojis, punctuation."""
        if not isinstance(text, str):
            return ""
        # Preserve name initials like Б.Сувдаа
        text = re.sub(r'\b([А-ЯӨҮЁ])\.?\s*([А-Яа-яӨөҮүЁё]+)', r'\1_\2', text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\[URL\]", "", text)
        text = re.sub(r"[@#]", "", text)
        text = re.sub(r"[^\w\s\u0400-\u04FF]", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def remove_stopwords(self, text: str) -> str:
        """Remove Mongolian stopwords from text."""
        if not isinstance(text, str):
            return ""
        words = text.split()
        words = [w for w in words if w.lower() not in self.stopwords]
        return " ".join(words)

    def preprocess(self, text: str, deep_clean: bool = True) -> str:
        """Full preprocessing pipeline."""
        text = self.clean_basic(text)
        if deep_clean:
            text = self.clean_deep(text)
            text = text.lower()
            text = self.remove_stopwords(text)
        return text.strip()

    def preprocess_batch(self, texts: List[str], deep_clean: bool = True) -> List[str]:
        """Preprocess a batch of texts."""
        return [self.preprocess(t, deep_clean) for t in texts]
