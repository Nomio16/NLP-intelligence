"""
check_preprocessing.py — Manual diagnostic for the Mongolian preprocessing pipeline.

Run from inside NLP-intelligence/:
    python check_preprocessing.py

Each test prints: INPUT → NLP OUTPUT | TM OUTPUT
Then flags anything that looks wrong.
"""

from nlp_core.preprocessing import Preprocessor

p = Preprocessor()

# ---------------------------------------------------------------------------
# Test cases: (label, raw_input, what_to_check)
# ---------------------------------------------------------------------------
CASES = [
    # ── Name protection ────────────────────────────────────────────────────
    ("uppercase initial",
     "Д.Гантулга УИХ-ын гишүүн байна.",
     "NLP: name should be Д.Гантулга (dot restored). TM: initial stripped → гантулга or гантулга"),

    ("lowercase initial (social media)",
     "өнөөдөр б.амар ирэхгүй байна гэсэн",
     "NLP: б.амар → Б.Амар (capitalized). TM: initial stripped, амар kept"),

    ("compound surname",
     "А.Бат-Эрдэнэ сайдаар томилогдлоо.",
     "NLP: А.Бат-Эрдэнэ stays as one token with dot. TM: бат-эрдэнэ as one hyphenated token"),

    # ── Capitalization for NER ─────────────────────────────────────────────
    ("all lowercase sentence",
     "монгол улсын ерөнхийлөгч х.баттулга өнөөдөр хэлэв",
     "NLP: 'монгол' → 'Монгол', х.баттулга → Х.Баттулга"),

    # ── Hashtags and mentions ──────────────────────────────────────────────
    ("hashtag and mention",
     "@МонголТВ #монголулс Улаанбаатар хотод мэдээ гарлаа",
     "NLP: @МонголТВ and #монголулс stripped. TM: same."),

    # ── URLs ───────────────────────────────────────────────────────────────
    ("URL handling",
     "Дэлгэрэнгүй мэдээллийг https://montsame.mn/news/123 хаягаас үзнэ үү",
     "NLP: URL → [URL] token. TM: URL removed entirely."),

    # ── Emoji ──────────────────────────────────────────────────────────────
    ("emoji sentiment markers",
     "Маш сайн байна 😊🔥 Улаанбаатар хотод ирлээ ✅",
     "NLP: 🔥→[EXCITED], unknown 😊 stripped. TM: all emoji stripped."),

    ("sarcastic laugh emoji",
     "Засгийн газрын шийдвэр маш сайн байна 😂😂",
     "NLP: 😂→[LAUGH] (ambiguous, BERT infers from context). TM: stripped."),

    ("negative emoji",
     "Энэ бол огт зөв биш 😡💔 нийтлэл байна",
     "NLP: 😡→[ANGRY] 💔→[SAD]. TM: stripped."),

    ("togrog symbol preserved",
     "Энэ бараа 50,000₮ байна — маш үнэтэй",
     "NLP: ₮ and — preserved (were wrongly removed before). TM: stripped by clean_deep."),

    # ── Stopword removal (TM only) ─────────────────────────────────────────
    ("stopword removal in TM",
     "энэ бол маш сайн санаа юм байна",
     "NLP: ALL words kept. TM: энэ бол маш юм байна removed, 'сайн санаа' should remain"),

    # ── Punctuation preservation (NLP only) ───────────────────────────────
    ("punctuation in NLP",
     "Тийм үү? Та хаанаас ирсэн бэ. Монгол улсаас.",
     "NLP: punctuation kept. TM: punctuation stripped."),

    # ── Real social media style ────────────────────────────────────────────
    ("real social media post",
     "яах вэ дээ шдэ 😂 @найз #хөгжилтэй монгол хүн л гэж бодогдоод байна",
     "NLP: slang particles kept, emoji/tags stripped. TM: шдэ, яах, вэ, дээ, л, гэж removed"),

    ("mixed mongolian english",
     "Today Монгол улсын ерөнхийлөгч made an announcement. #politics",
     "NLP: English words kept, Mongolian capitalized. TM: cleaned."),
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD  = "\033[1m"
YELLOW = "\033[33m"
CYAN  = "\033[36m"
GREEN = "\033[32m"
RED   = "\033[31m"

def run():
    print(f"\n{BOLD}=== PREPROCESSING DIAGNOSTIC ==={RESET}\n")
    issues = []

    for label, raw, hint in CASES:
        nlp_out = p.preprocess_nlp(raw)
        tm_out  = p.preprocess_tm(raw)

        print(f"{BOLD}{CYAN}[{label}]{RESET}")
        print(f"  {YELLOW}IN :{RESET}  {raw}")
        print(f"  {GREEN}NLP:{RESET}  {nlp_out}")
        print(f"  {GREEN}TM :{RESET}  {tm_out}")
        print(f"  {YELLOW}CHECK:{RESET} {hint}")

        # ── Automatic sanity checks ──────────────────────────────────────
        case_issues = []

        # NLP: should not be empty
        if not nlp_out.strip():
            case_issues.append("NLP output is EMPTY")

        # TM: should not be empty (unless all stopwords)
        if not tm_out.strip():
            case_issues.append("TM output is EMPTY (may be okay if all stopwords)")

        # NLP: URL should become [URL]
        if "https://" in raw and "[URL]" not in nlp_out:
            case_issues.append("URL not replaced with [URL] in NLP mode")

        # TM: URL should be fully removed
        if "https://" in raw and ("https://" in tm_out or "[URL]" in tm_out):
            case_issues.append("URL not fully removed in TM mode")

        # NLP: hashtag/mention should be stripped
        if "@" in nlp_out or (any(c in raw for c in "@#") and "#" in nlp_out):
            case_issues.append("Hashtag or mention still present in NLP output")

        # NLP: if input had uppercase initial name like Д.Гантулга, it should survive
        import re
        upper_names = re.findall(r"[А-ЯӨҮЁ]\.[А-Яа-яӨөҮүЁё]", raw)
        for name in upper_names:
            initial = name[0]
            if initial + "." not in nlp_out:
                case_issues.append(f"Name initial {name!r} lost dot in NLP output → got: {nlp_out}")

        # NLP: first word of sentence should be capitalized
        first_word = nlp_out.split()[0] if nlp_out.split() else ""
        if first_word and first_word[0].islower():
            case_issues.append(f"First word '{first_word}' not capitalized in NLP output")

        if case_issues:
            for issue in case_issues:
                print(f"  {RED}⚠ ISSUE: {issue}{RESET}")
            issues.extend([(label, i) for i in case_issues])
        else:
            print(f"  {GREEN}✓ No automatic issues detected{RESET}")

        print()

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"{BOLD}=== SUMMARY ==={RESET}")
    if issues:
        print(f"{RED}{len(issues)} issue(s) found:{RESET}")
        for label, issue in issues:
            print(f"  [{label}] {issue}")
    else:
        print(f"{GREEN}All automatic checks passed. Review the outputs above manually.{RESET}")

    print()

if __name__ == "__main__":
    run()
