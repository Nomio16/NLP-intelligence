"""
Generate training data in character-offset JSON format.
All output uses the same format as NER_v1.0.json.gz:
  {"text": "...", "labels": [[start, end, "LABEL"], ...]}

Sources:
  1. NER_v1.0.json.gz             → copy as-is (10,162 sentences)
  2. mongolian_abbreviations       → generate ORG sentences
  3. level_2-5.json                → generate LOC sentences
  4. personal names + suffixes     → generate PER sentences
  5. NEW: full-name-only org sentences
  6. NEW: current politicians (2024)
  7. NEW: companies, banks, media
  8. NEW: government agencies
"""

import json
import gzip
import csv
import random
import os

random.seed(42)

NER_DATASET = os.path.join(os.path.dirname(__file__), "..", "NER-dataset")
OUTPUT_DIR = os.path.dirname(__file__)


# ═══════════════════════════════════════════════
#  DATA LISTS
# ═══════════════════════════════════════════════

# Mongolian case suffixes
SUFFIXES = [
    "", "ын", "ийн", "гийн", "д", "т", "ыг", "ийг",
    "аас", "ээс", "оос", "өөс",
    "тай", "тэй", "той", "төй",
    "руу", "рүү",
    "аар", "ээр", "оор", "өөр",
]

INITIALS = list("АБВГДЕЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЭЮЯ")

# ───── COMMON MONGOLIAN NAMES (expanded) ─────
COMMON_NAMES = [
    # Classic / common names
    "Батулга", "Хүрэлсүх", "Энхбаяр", "Ганбаатар", "Цогтбаатар",
    "Баасанхүү", "Ганхуяг", "Нямбаатар", "Дэлгэрсайхан", "Мөнхбат",
    "Тогтохсүрэн", "Ганболд", "Эрдэнэбат", "Амарсайхан", "Батбаатар",
    "Сүхбаатар", "Баярсайхан", "Нарантуяа", "Оюунчимэг", "Мөнхжаргал",
    "Золжаргал", "Энхтайван", "Алтангэрэл", "Батмөнх", "Дамдинсүрэн",
    "Жамбалдорж", "Чимэддорж", "Пүрэвсүрэн", "Уранчимэг", "Болормаа",
    "Сарантуяа", "Энхтүвшин", "Цэцэгмаа", "Одонтуяа", "Мягмарсүрэн",
    "Батцэцэг", "Ундрах",
    # More male names
    "Ганзориг", "Баатарсүх", "Мөнхсайхан", "Тэмүүлэн", "Эрдэнэсайхан",
    "Нарансүх", "Батсүх", "Цэрэндорж", "Дорж", "Болд",
    "Цэндбаатар", "Эрдэнэбаатар", "Баярмаа", "Оюунболд", "Мөнхтулга",
    "Ганзул", "Батжаргал", "Ринчиндорж", "Лхагвасүрэн", "Батчулуун",
    "Мөнхболд", "Баттулга", "Ганбат", "Гантулга", "Баттүшиг",
    "Дэмбэрэл", "Нямдорж", "Цэнгэл", "Оргил", "Отгонбаяр",
    "Содномбалжир", "Тулга", "Баянмөнх", "Цогзолмаа", "Дэлгэрмаа",
    "Хүрэлбаатар", "Мөнхөө", "Энхжаргал", "Болдбаатар", "Жаргалсайхан",
    # More female names
    "Солонго", "Анужин", "Номинчимэг", "Мөнхзул", "Энхцэцэг",
    "Алтанцэцэг", "Байгалмаа", "Ариунаа", "Оюунаа", "Должинсүрэн",
    "Туяа", "Цэрмаа", "Энхмаа", "Мэдэгмаа", "Ариунзул",
    "Сүрэнхорлоо", "Мандах", "Мөнхцэцэг", "Оюунгэрэл", "Буянаа",
    # Current politicians (2024)
    "Амарбаясгалан", "Жавхлан", "Чинзориг", "Батцэцэг",
    "Мөнхбаатар", "Сайнбуян", "Пүрэвсүрэн", "Энхамгалан",
    "Нямосор", "Ганбямба", "Тэмүүжин", "Энхбат",
    "Номиндарь", "Ням-Осор", "Содбаатар", "Цэрэнпил",
]

# Compound names with hyphens
COMPOUND_NAMES = [
    "Бат-Эрдэнэ", "Оюун-Эрдэнэ", "Ану-Үжин", "Бат-Үүл", "Бат-Очир",
    "Мөнх-Эрдэнэ", "Ган-Очир", "Алтан-Очир", "Бат-Амгалан", "Отгон-Эрдэнэ",
    "Саран-Туяа", "Бямба-Очир", "Түмэн-Өлзий", "Ган-Эрдэнэ", "Бат-Хүрэл",
    "Мөнх-Оргил", "Алтан-Эрдэнэ", "Эрдэнэ-Очир", "Цэнд-Аюуш", "Бат-Билэг",
    "Энх-Амгалан", "Бат-Сайхан", "Мөнх-Амгалан", "Ган-Эрдэнэ", "Алтан-Од",
]

# ───── CURRENT POLITICIANS with specific initials (2024) ─────
KNOWN_POLITICIANS = [
    # (initial, name, position_context)
    ("У", "Хүрэлсүх", "Ерөнхийлөгч"),
    ("Л", "Оюун-Эрдэнэ", "Ерөнхий сайд"),
    ("Д", "Амарбаясгалан", "УИХ-ын дарга"),
    ("С", "Амарсайхан", "УИХ-ын дэд дарга"),
    ("Б", "Батцэцэг", "Гадаад хэргийн сайд"),
    ("Ч", "Жавхлан", "Сангийн сайд"),
    ("Д", "Чинзориг", "Хөдөлмөрийн сайд"),
    ("Л", "Энх-Амгалан", "Боловсролын сайд"),
    ("С", "Мөнхбаатар", "Зам тээврийн сайд"),
    ("Б", "Сайнбуян", "Хууль зүйн сайд"),
    ("К", "Ням-Осор", "Батлан хамгаалахын сайд"),
    ("Л", "Пүрэвсүрэн", "Эрчим хүчний сайд"),
    ("Ж", "Ганбаатар", "Хүнс хөдөө аж ахуйн сайд"),
    ("Б", "Энхбаяр", "Уул уурхайн сайд"),
    ("Н", "Ганбямба", "Байгаль орчны сайд"),
    ("С", "Тэмүүжин", "АН-ын дарга"),
    ("Д", "Ганбаатар", "ХҮН намын дарга"),
    ("Ж", "Энхбат", "ИЗНН-ын дарга"),
    # Historical / well-known
    ("Ц", "Элбэгдорж", "Ерөнхийлөгч асан"),
    ("Х", "Баттулга", "Ерөнхийлөгч асан"),
    ("Н", "Энхбаяр", "Ерөнхийлөгч асан"),
    ("С", "Баяр", "Ерөнхий сайд асан"),
    ("Н", "Алтанхуяг", "Ерөнхий сайд асан"),
    ("Ч", "Сайханбилэг", "Ерөнхий сайд асан"),
    ("Ж", "Эрдэнэбат", "Ерөнхий сайд асан"),
    ("М", "Энхболд", "УИХ-ын дарга асан"),
    ("Г", "Занданшатар", "УИХ-ын дарга асан"),
    ("Д", "Цогтбаатар", "Гадаад хэргийн сайд асан"),
    ("С", "Ганбаатар", "Хууль зүйн сайд асан"),
]

# ───── GOVERNMENT AGENCIES (abbreviation + full name + label) ─────
GOVERNMENT_AGENCIES = [
    # Ministries
    ("ГХЯ", "Гадаад хэргийн яам", "ORG"),
    ("СЯ", "Сангийн яам", "ORG"),
    ("ХЗДХЯ", "Хууль зүй дотоод хэргийн яам", "ORG"),
    ("БХЦЯ", "Батлан хамгаалахын яам", "ORG"),
    ("БШУЯ", "Боловсрол шинжлэх ухааны яам", "ORG"),
    ("ЭМЯ", "Эрүүл мэндийн яам", "ORG"),
    ("ХНХЯ", "Хөдөлмөр нийгмийн хамгааллын яам", "ORG"),
    ("ЗТХЯ", "Зам тээврийн хөгжлийн яам", "ORG"),
    ("БХБЯ", "Барилга хот байгуулалтын яам", "ORG"),
    ("УУХҮЯ", "Уул уурхай хүнд үйлдвэрийн яам", "ORG"),
    ("ХХААХҮЯ", "Хүнс хөдөө аж ахуй хөнгөн үйлдвэрийн яам", "ORG"),
    ("БОАЖЯ", "Байгаль орчин аялал жуулчлалын яам", "ORG"),
    ("ЦХХЯ", "Цахим хөгжил харилцаа холбооны яам", "ORG"),
    ("ЭХЯ", "Эрчим хүчний яам", "ORG"),
    ("ССЯ", "Соёл спортын яам", "ORG"),
    # Agencies
    ("АТГ", "Авлигатай тэмцэх газар", "ORG"),
    ("ТЕГ", "Тагнуулын ерөнхий газар", "ORG"),
    ("ЦЕГ", "Цагдаагийн ерөнхий газар", "ORG"),
    ("ОБЕГ", "Онцгой байдлын ерөнхий газар", "ORG"),
    ("УЕПГ", "Улсын ерөнхий прокурорын газар", "ORG"),
    ("ГЕГ", "Гаалийн ерөнхий газар", "ORG"),
    ("ҮСХ", "Үндэсний статистикийн хороо", "ORG"),
    ("СЗХ", "Санхүүгийн зохицуулах хороо", "ORG"),
    ("ШӨХТГ", "Шударга өрсөлдөөн хэрэглэгчийн төлөө газар", "ORG"),
    ("ХХЗХ", "Харилцаа холбооны зохицуулах хороо", "ORG"),
    ("НДЕГ", "Нийгмийн даатгалын ерөнхий газар", "ORG"),
    ("НЗДТГ", "Нийслэлийн засаг даргын тамгын газар", "ORG"),
    ("ШШГЕГ", "Шүүхийн шийдвэр гүйцэтгэх ерөнхий газар", "ORG"),
    ("ТЕГ", "Татварын ерөнхий газар", "ORG"),
    ("МХЕГ", "Мэргэжлийн хяналтын ерөнхий газар", "ORG"),
    ("ГИХАГ", "Гадаадын иргэн харьяатын асуудал эрхлэх газар", "ORG"),
]

# ───── COMPANIES, BANKS, MEDIA, MINING ─────
COMPANIES = [
    # Banks
    ("Хаан банк", "ORG"), ("Худалдаа хөгжлийн банк", "ORG"),
    ("Голомт банк", "ORG"), ("Хас банк", "ORG"), ("Төрийн банк", "ORG"),
    ("Капитрон банк", "ORG"), ("Богд банк", "ORG"), ("Чингис хаан банк", "ORG"),
    ("Ард санхүүгийн нэгдэл", "ORG"), ("Монголбанк", "ORG"),
    # Telecom
    ("Мобиком", "ORG"), ("Юнител", "ORG"), ("Скайтел", "ORG"),
    ("Жи-Мобайл", "ORG"), ("Монголиа телеком", "ORG"),
    # Mining & Energy
    ("Эрдэнэт үйлдвэр", "ORG"), ("Оюу толгой", "ORG"),
    ("Эрдэнэс Тавантолгой", "ORG"), ("Эрдэнэс Монгол", "ORG"),
    ("Тавантолгой", "ORG"), ("Монгол Алт", "ORG"),
    ("Бороо Голд", "ORG"), ("Турквоз Хилл", "ORG"),
    # Airlines & Transport
    ("МИАТ", "ORG"), ("Хүннү Эйр", "ORG"), ("Аэро Монголиа", "ORG"),
    ("Улаанбаатар төмөр зам", "ORG"),
    # Media
    ("Монцамэ", "ORG"), ("Монголын үндэсний телевиз", "ORG"),
    ("Eagle News", "ORG"), ("Зүүн хаалга медиа", "ORG"),
    # Major companies
    ("МАК", "ORG"), ("МИК", "ORG"), ("АПУ", "ORG"),
    ("Говь ХК", "ORG"), ("Талх чихэр", "ORG"),
    ("Шүүдан ХХК", "ORG"), ("Монголын цахилгаан холбоо", "ORG"),
    ("Мон-Инженеринг", "ORG"), ("Мон-Пари", "ORG"),
    ("Номин холдинг", "ORG"), ("Макс групп", "ORG"),
    ("Тэнгэр санхүүгийн групп", "ORG"), ("Бодь групп", "ORG"),
    # International orgs commonly mentioned
    ("Дэлхийн банк", "ORG"), ("Дэлхийн Эрүүл Мэндийн Байгууллага", "ORG"),
    ("Нэгдсэн Үндэстний Байгууллага", "ORG"),
    ("Олон Улсын Валютын Сан", "ORG"),
    ("ЮНЕСКО", "ORG"), ("ДЭМБ", "ORG"),
    # Political parties
    ("Монгол Ардын Нам", "ORG"), ("Ардчилсан Нам", "ORG"),
    ("ХҮН нам", "ORG"), ("Иргэний Зориг Ногоон нам", "ORG"),
    ("Зөв хүн Электорат нам", "ORG"),
    # Universities
    ("Монгол Улсын Их Сургууль", "ORG"), ("МУИС", "ORG"),
    ("Монгол Улсын Шинжлэх Ухааны Академи", "ORG"),
    ("ШУТИС", "ORG"), ("ХААИС", "ORG"), ("СЭЗИС", "ORG"),
    ("Соёлын Их Сургууль", "ORG"), ("Анагаахын Шинжлэх Ухааны Үндэсний Их Сургууль", "ORG"),
    ("АШУҮИС", "ORG"),
]

# ═══════════════════════════════════════════════
#  TEMPLATE DEFINITIONS
# ═══════════════════════════════════════════════

ORG_TEMPLATES = [
    "{name}-ын дарга мэдэгдэл хийлээ",
    "{name}-ын хурал болж байна",
    "{name}-ийн шийдвэрийг зарлалаа",
    "{name}-д хандив өргөлөө",
    "{name} шинэ бодлого баталлаа",
    "{name}-ын ажилтнууд цуглалаа",
    "{name}-ийн тайлан гарлаа",
    "Өнөөдөр {name}-ын хурал болно",
    "{name}-аас мэдэгдэл гаргалаа",
    "{name}-ын гишүүд уулзалт хийв",
    "Маргааш {name}-д уулзалт болно",
    "{name}-ын төлөөлөгч ярилцлага өгөв",
    "{name} шинэ төсөл эхлүүллээ",
    "{name}-ын удирдлагууд хуралдлаа",
    "{name} тайлангаа танилцууллаа",
]

ORG_FULLNAME_ONLY_TEMPLATES = [
    "{fullname}-ын дарга мэдэгдэл хийлээ",
    "{fullname}-ийн хурал өнөөдөр болов",
    "{fullname}-д шинэ захирал томилогдлоо",
    "{fullname} шинэ бодлого баталлаа",
    "{fullname}-аас тайлбар ирүүлэв",
    "{fullname}-ын тайлан гарлаа",
    "Өнөөдөр {fullname}-ын хурал болно",
    "{fullname} ажлаа эхлүүллээ",
    "{fullname}-ийн ажилтнууд цуглалаа",
    "Маргааш {fullname}-д уулзалт болно",
    "{fullname}-ийн удирдлагууд хуралдлаа",
    "{fullname} тайлангаа танилцууллаа",
    "{fullname}-аас мэдэгдэл гаргалаа",
    "{fullname}-ын гишүүд уулзалт хийв",
    "{fullname} шинэ төсөл эхлүүллээ",
]

COMPANY_TEMPLATES = [
    "{name}-д ажилтан авна",
    "{name}-ын хувьцааны ханш өрсөн",
    "{name} шинэ бүтээгдэхүүн гаргалаа",
    "{name}-тай гэрээ байгууллаа",
    "{name}-ын ашиг орлого нэмэгдлээ",
    "{name}-д хөрөнгө оруулалт орж ирэв",
    "{name} шинэ салбар нээлээ",
    "{name}-ын захирал ярилцлага өгөв",
    "Иргэд {name}-д гомдол гаргалаа",
    "{name} зах зээлд тэргүүлж байна",
    "{name}-ын үйлчилгээ сайжирлаа",
    "Өнөөдөр {name}-ын хурал болно",
]

POLITICIAN_TEMPLATES = [
    "{per} {position} болж томилогдлоо",
    "{position} {per} мэдэгдэл хийлээ",
    "{per} хэвлэлийн хурал зарлалаа",
    "{per} гадаад айлчлал хийв",
    "Иргэд {per} шүүмжиллээ",
    "{per} парламентад үг хэлэв",
    "{per} шинэ хууль санаачиллаа",
    "Сэтгүүлчид {per} ярилцлага авав",
    "{per} сонгогчидтой уулзав",
    "{per} төсвийн талаар мэдэгдэл хийв",
    "{per} хуралд оролцов",
    "{per} шийдвэр гаргалаа",
]

AGENCY_POLITICIAN_TEMPLATES = [
    "{per} {org}-ын даргаар томилогдлоо",
    "{org}-ын дарга {per} мэдэгдэл хийв",
    "{per} {org}-д ажиллаж байна",
    "{org}-ын {per} хэвлэлийн хурал хийлээ",
]

LOC_TEMPLATES = [
    "{name} нутгийн иргэд цуглалаа",
    "{name}-д шинэ сургууль баригдана",
    "{name}-ийн засаг дарга мэдэгдэл хийлээ",
    "Өнөөдөр {name}-д хүйтэн байна",
    "{name} нутагт бороо орно",
    "{name}-аас ирсэн мэдээ",
    "{name}-ийн иргэд гомдол гаргалаа",
    "Маргааш {name}-д арга хэмжээ болно",
    "{name} хотод шинэ зам тавина",
    "{name}-д хөрөнгө оруулалт нэмэгдлээ",
    "{name}-ийн хүн ам өсчээ",
    "Манай гэр бүл {name}-д амьдардаг",
    "{name} нь Монголын баруун хэсэгт оршдог",
    "{name}-ийн төвд шинэ эмнэлэг нээгдлээ",
    "{name} сумын малчид хүндрэл тулгарлаа",
]

LOC_MULTI_TEMPLATES = [
    "{loc1}-аас {loc2} руу нисэх тийз хямдарчээ",
    "{loc1} болон {loc2}-д цас орно",
    "{loc1}-ийн засаг дарга {loc2}-д зочиллоо",
]

PER_TEMPLATES = [
    "{name} ерөнхий сайдаар томилогдлоо",
    "{name} сонгуульд нэр дэвшив",
    "{name} хэвлэлийн хурал хийлээ",
    "{name} шүүхэд дуудагдлаа",
    "{name} парламентад үг хэллээ",
    "Ерөнхийлөгч {name} зарлиг гаргалаа",
    "{name} албан тушаалаасаа огцорлоо",
    "{name} шагнал хүртлээ",
    "{name} гадаад айлчлал хийв",
    "Иргэд {name} шүүмжиллээ",
    "{name} хуралд оролцов",
    "Сэтгүүлчид {name} асуулт тавив",
    "{name} мэдэгдэл гаргалаа",
    "{name} шинэ хууль санаачиллаа",
    "{name} сонирхол татав",
    "{name} захидал бичив",
    "{name} илтгэл тавив",
    "{name} ном бичжээ",
    "Олон нийт {name} дэмжив",
    "{name} шагнал гардуулав",
]

PER_INITIAL_TEMPLATES = [
    "{initial}.{name} ерөнхий сайдаар томилогдлоо",
    "{initial}.{name} хэвлэлийн хурал хийлээ",
    "{initial}.{name} шүүхэд дуудагдлаа",
    "Ерөнхийлөгч {initial}.{name} зарлиг гаргалаа",
    "{initial}.{name} албан тушаалаасаа огцорлоо",
    "Сайд {initial}.{name} тайлбар хийв",
    "Гишүүн {initial}.{name} саналаа илэрхийллээ",
    "{initial}.{name} сонгуульд ялалт байгууллаа",
    "Иргэд {initial}.{name} дэмжив",
    "{initial}.{name} албан ёсоор мэдэгдлээ",
    "{initial}.{name} ярилцлага өгөв",
    "Дарга {initial}.{name} тушаал гаргав",
    "{initial}.{name} тэмцээнд оролцов",
    "{initial}.{name} шагнал хүртлээ",
    "{initial}.{name} хуралд оролцов",
]


# ═══════════════════════════════════════════════
#  DATA LOADING FUNCTIONS
# ═══════════════════════════════════════════════

def load_ner_v1():
    path = os.path.join(NER_DATASET, "NER_v1.0.json.gz")
    entries = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("text") and entry.get("labels"):
                entries.append(entry)
    print(f"[NER_v1.0] Loaded {len(entries)} sentences")
    return entries


def load_abbreviations():
    path = os.path.join(NER_DATASET, "mongolian_abbreviations.csv")
    abbrevs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "|" in line:
                parts = line.split("|")
                abbrev = parts[0].strip()
                fullname = parts[1].strip() if len(parts) > 1 else ""
                if abbrev and len(abbrev) >= 2:
                    abbrevs.append((abbrev, fullname))
    print(f"[Abbreviations] Loaded {len(abbrevs)} entries")
    return abbrevs


def load_locations():
    locations = []
    for level_file in ["level_2.json", "level_3.json"]:
        path = os.path.join(NER_DATASET, level_file)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    name = item.get("canonical", "")
                    if name:
                        locations.append(name)

    districts_path = os.path.join(NER_DATASET, "districts.csv")
    if os.path.exists(districts_path):
        with open(districts_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                for cell in row:
                    cell = cell.strip()
                    if cell and len(cell) > 2 and cell not in locations:
                        locations.append(cell)

    countries_path = os.path.join(NER_DATASET, "countries.csv")
    if os.path.exists(countries_path):
        with open(countries_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 1:
                    country = parts[0].strip()
                    if country and len(country) > 1:
                        locations.append(country)

    # Notable locations not in files
    extra_locs = [
        "Чингис хаан олон улсын нисэх буудал", "Шинэ яармаг",
        "Майдар хот", "Зайсан", "Нарантуул", "Хурд",
    ]
    locations.extend(extra_locs)

    locations = list(dict.fromkeys(locations))
    print(f"[Locations] Loaded {len(locations)} location names")
    return locations


# ═══════════════════════════════════════════════
#  GENERATION FUNCTIONS
# ═══════════════════════════════════════════════

LOC_ABBREVS = {
    "АНУ", "БНСУ", "ОХУ", "БНХАУ", "БНАСАУ", "СХД", "ХУД",
    "БЗД", "БГД", "СБД", "ЧД", "УБ",
}


def _make_entry(text, entity_text, label, start_hint=None):
    """Helper to create a labeled entry, finding entity position in text."""
    try:
        start = text.index(entity_text) if start_hint is None else start_hint
        end = start + len(entity_text)
        return {"text": text, "labels": [[start, end, label]]}
    except ValueError:
        return None


def generate_org_sentences(abbrevs):
    """Generate sentences from abbreviations — both abbrev-only and full-name-only."""
    entries = []

    for abbrev, fullname in abbrevs:
        label = "LOC" if abbrev in LOC_ABBREVS else "ORG"

        # --- Abbreviation-only sentences (МАН-ын дарга...) ---
        for template in random.sample(ORG_TEMPLATES, min(3, len(ORG_TEMPLATES))):
            text = template.format(name=abbrev)
            e = _make_entry(text, abbrev, label)
            if e: entries.append(e)

        # --- Full-name-only sentences (Монгол Ардын Нам шинэ бодлого баталлаа) ---
        if fullname and len(fullname) > 3:
            for template in random.sample(ORG_FULLNAME_ONLY_TEMPLATES, min(3, len(ORG_FULLNAME_ONLY_TEMPLATES))):
                text = template.format(fullname=fullname)
                e = _make_entry(text, fullname, label)
                if e: entries.append(e)

        # --- Both together (Монгол Ардын Нам буюу МАН) ---
        if fullname and random.random() < 0.5:
            templates = [
                f"{fullname} буюу {abbrev} шинэ бодлого баталлаа",
                f"{abbrev} ({fullname}) хуралдлаа",
                f"{fullname} ({abbrev})-ын дарга мэдэгдэл хийлээ",
            ]
            text = random.choice(templates)
            labels = []
            try:
                idx_a = text.index(abbrev)
                labels.append([idx_a, idx_a + len(abbrev), label])
                idx_f = text.index(fullname)
                labels.append([idx_f, idx_f + len(fullname), label])
                entries.append({"text": text, "labels": labels})
            except ValueError:
                pass

    print(f"[ORG Abbreviations] Generated {len(entries)} sentences")
    return entries


def generate_agency_sentences():
    """Generate sentences from government agencies — both abbrev and full name."""
    entries = []

    for abbrev, fullname, label in GOVERNMENT_AGENCIES:
        # Abbreviation-only
        for template in random.sample(ORG_TEMPLATES, min(3, len(ORG_TEMPLATES))):
            text = template.format(name=abbrev)
            e = _make_entry(text, abbrev, label)
            if e: entries.append(e)

        # Full-name-only
        for template in random.sample(ORG_FULLNAME_ONLY_TEMPLATES, min(3, len(ORG_FULLNAME_ONLY_TEMPLATES))):
            text = template.format(fullname=fullname)
            e = _make_entry(text, fullname, label)
            if e: entries.append(e)

        # Both together
        text = f"{fullname} ({abbrev})-ын дарга мэдэгдэл хийлээ"
        labels = []
        try:
            idx_a = text.index(abbrev)
            labels.append([idx_a, idx_a + len(abbrev), label])
            idx_f = text.index(fullname)
            labels.append([idx_f, idx_f + len(fullname), label])
            entries.append({"text": text, "labels": labels})
        except ValueError:
            pass

        # With politician
        for init, name, position in random.sample(KNOWN_POLITICIANS, min(2, len(KNOWN_POLITICIANS))):
            per_name = f"{init}.{name}"
            template = random.choice(AGENCY_POLITICIAN_TEMPLATES)
            text = template.format(per=per_name, org=abbrev)
            labels = []
            try:
                idx_p = text.index(per_name)
                labels.append([idx_p, idx_p + len(per_name), "PER"])
                idx_o = text.index(abbrev)
                labels.append([idx_o, idx_o + len(abbrev), label])
                entries.append({"text": text, "labels": labels})
            except ValueError:
                pass

    print(f"[Government Agencies] Generated {len(entries)} sentences")
    return entries


def generate_company_sentences():
    """Generate sentences from companies, banks, media."""
    entries = []

    for company_name, label in COMPANIES:
        for template in random.sample(COMPANY_TEMPLATES, min(3, len(COMPANY_TEMPLATES))):
            text = template.format(name=company_name)
            e = _make_entry(text, company_name, label)
            if e: entries.append(e)

    print(f"[Companies] Generated {len(entries)} sentences")
    return entries


def generate_loc_sentences(locations):
    entries = []

    for loc_name in locations:
        templates = random.sample(LOC_TEMPLATES, min(2, len(LOC_TEMPLATES)))
        for template in templates:
            text = template.format(name=loc_name)
            e = _make_entry(text, loc_name, "LOC")
            if e: entries.append(e)

    if len(locations) >= 2:
        for _ in range(min(500, len(locations))):
            loc1, loc2 = random.sample(locations, 2)
            template = random.choice(LOC_MULTI_TEMPLATES)
            text = template.format(loc1=loc1, loc2=loc2)
            labels = []
            try:
                idx1 = text.index(loc1)
                labels.append([idx1, idx1 + len(loc1), "LOC"])
                idx2 = text.index(loc2)
                labels.append([idx2, idx2 + len(loc2), "LOC"])
                entries.append({"text": text, "labels": labels})
            except ValueError:
                pass

    print(f"[LOC Locations] Generated {len(entries)} sentences")
    return entries


def generate_per_sentences():
    entries = []
    all_names = COMMON_NAMES + COMPOUND_NAMES

    for name in all_names:
        # With various suffixes
        for suffix in random.sample(SUFFIXES, min(5, len(SUFFIXES))):
            suffixed = name + suffix
            template = random.choice(PER_TEMPLATES)
            text = template.format(name=suffixed)
            e = _make_entry(text, suffixed, "PER")
            if e: entries.append(e)

        # With initial (Д.Батулга format) + optional suffix
        initial = random.choice(INITIALS)
        for _ in range(4):
            template = random.choice(PER_INITIAL_TEMPLATES)
            full_name = f"{initial}.{name}"
            suffix = random.choice(["", "", "ын", "ийн", "д", "ыг", "аас", "тай"])
            suffixed_name = f"{initial}.{name}{suffix}" if suffix else full_name
            text = template.format(initial=initial, name=name)
            if suffix:
                text = text.replace(full_name, suffixed_name)
            e = _make_entry(text, suffixed_name, "PER")
            if e: entries.append(e)

    # Multi-person sentences
    for _ in range(800):
        name1, name2 = random.sample(all_names, 2)
        init1, init2 = random.choice(INITIALS), random.choice(INITIALS)
        full1, full2 = f"{init1}.{name1}", f"{init2}.{name2}"

        templates = [
            f"{full1} болон {full2} уулзалт хийв",
            f"Сайд {full1} гишүүн {full2} нартай уулзав",
            f"{full1} {full2} хоёр хамтран ажиллана",
            f"{full1} болон {full2} хуралд оролцов",
        ]
        text = random.choice(templates)
        labels = []
        try:
            idx1 = text.index(full1)
            labels.append([idx1, idx1 + len(full1), "PER"])
            idx2 = text.index(full2)
            labels.append([idx2, idx2 + len(full2), "PER"])
            entries.append({"text": text, "labels": labels})
        except ValueError:
            pass

    print(f"[PER Names] Generated {len(entries)} sentences")
    return entries


def generate_politician_sentences():
    """Generate sentences with real politician names in context."""
    entries = []

    for init, name, position in KNOWN_POLITICIANS:
        per_name = f"{init}.{name}"

        # With position context
        for template in random.sample(POLITICIAN_TEMPLATES, min(5, len(POLITICIAN_TEMPLATES))):
            text = template.format(per=per_name, position=position)
            e = _make_entry(text, per_name, "PER")
            if e: entries.append(e)

        # With suffixes
        for suffix in random.sample(SUFFIXES[:8], min(4, 8)):
            suffixed = f"{init}.{name}{suffix}"
            template = random.choice(PER_TEMPLATES)
            text = template.format(name=suffixed)
            e = _make_entry(text, suffixed, "PER")
            if e: entries.append(e)

    # Politician + org combinations
    for init, name, position in KNOWN_POLITICIANS:
        per_name = f"{init}.{name}"
        for abbrev, fullname, label in random.sample(GOVERNMENT_AGENCIES, min(2, len(GOVERNMENT_AGENCIES))):
            text = f"{per_name} {fullname}-д ажиллаж байна"
            labels = []
            try:
                idx_p = text.index(per_name)
                labels.append([idx_p, idx_p + len(per_name), "PER"])
                idx_o = text.index(fullname)
                labels.append([idx_o, idx_o + len(fullname), "ORG"])
                entries.append({"text": text, "labels": labels})
            except ValueError:
                pass

    print(f"[Politicians] Generated {len(entries)} sentences")
    return entries


def generate_mixed_sentences(abbrevs, locations):
    """Sentences with multiple entity types together."""
    entries = []
    all_names = COMMON_NAMES + COMPOUND_NAMES

    mixed_templates = [
        ("Сайд {per} {loc}-д ажиллав", ["PER", "LOC"]),
        ("{per} {org}-ын хурлаар үг хэлэв", ["PER", "ORG"]),
        ("{org}-ын дарга {per} {loc} хотод зочиллоо", ["ORG", "PER", "LOC"]),
        ("{per} {loc}-аас {loc2} руу нүүлээ", ["PER", "LOC", "LOC"]),
        ("{org} {loc}-д шинэ салбар нээлээ", ["ORG", "LOC"]),
        ("{per} {org}-д ажилд орлоо", ["PER", "ORG"]),
        ("{loc}-ийн {org}-ын дарга {per} мэдэгдэл хийв", ["LOC", "ORG", "PER"]),
        ("{per} {org}-ын хуралд {loc}-д оролцов", ["PER", "ORG", "LOC"]),
        ("{org} {per} нарыг {loc}-д томиллоо", ["ORG", "PER", "LOC"]),
    ]

    org_names = [a[0] for a in abbrevs[:100]] + [c[0] for c in COMPANIES[:30]]

    for _ in range(3000):
        template_text, label_types = random.choice(mixed_templates)

        per = f"{random.choice(INITIALS)}.{random.choice(all_names)}"
        org = random.choice(org_names) if org_names else "МАН"
        loc = random.choice(locations[:100]) if locations else "Улаанбаатар"
        loc2 = random.choice(locations[:100]) if locations else "Дархан"

        try:
            text = template_text.format(per=per, org=org, loc=loc, loc2=loc2)
        except (KeyError, IndexError):
            continue

        labels = []
        for entity_text, label_type in [
            (per, "PER"), (org, "ORG"), (loc, "LOC"), (loc2, "LOC")
        ]:
            try:
                idx = text.index(entity_text)
                if not any(l[0] == idx for l in labels):
                    labels.append([idx, idx + len(entity_text), label_type])
            except ValueError:
                continue

        if labels:
            entries.append({"text": text, "labels": labels})

    print(f"[Mixed] Generated {len(entries)} sentences")
    return entries


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Generating NER Training Data v2.1")
    print("=" * 60)

    # Load sources
    ner_v1 = load_ner_v1()
    abbrevs = load_abbreviations()
    locations = load_locations()

    # Generate all categories
    org_data = generate_org_sentences(abbrevs)
    agency_data = generate_agency_sentences()
    company_data = generate_company_sentences()
    loc_data = generate_loc_sentences(locations)
    per_data = generate_per_sentences()
    politician_data = generate_politician_sentences()
    mixed_data = generate_mixed_sentences(abbrevs, locations)

    # Save individual files
    datasets = {
        "ner_v1_base.jsonl": ner_v1,
        "org_abbreviations.jsonl": org_data,
        "org_agencies.jsonl": agency_data,
        "org_companies.jsonl": company_data,
        "loc_locations.jsonl": loc_data,
        "per_names.jsonl": per_data,
        "per_politicians.jsonl": politician_data,
        "mixed_entities.jsonl": mixed_data,
    }

    for filename, data in datasets.items():
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  Saved {filename}: {len(data)} entries")

    # Merge all into one file
    all_data = []
    for data in datasets.values():
        all_data.extend(data)
    random.shuffle(all_data)

    merged_path = os.path.join(OUTPUT_DIR, "train_v2_merged.jsonl")
    with open(merged_path, "w", encoding="utf-8") as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(all_data)} training sentences")
    print(f"Saved to: {merged_path}")
    print(f"{'=' * 60}")

    # Statistics
    from collections import Counter
    label_counts = Counter()
    for entry in all_data:
        for _, _, label in entry["labels"]:
            label_counts[label] += 1

    print(f"\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")

    # Show breakdown
    print(f"\nDataset breakdown:")
    for filename, data in datasets.items():
        print(f"  {filename}: {len(data)} sentences")


if __name__ == "__main__":
    main()
