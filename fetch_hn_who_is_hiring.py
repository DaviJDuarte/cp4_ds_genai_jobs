"""Baixa e classifica comentários de nível superior do Hacker News 'Who is hiring?'.

O script usa a API pública de busca do HN fornecida pelo Algolia. Cada comentário
de nível superior é tratado como uma publicação de empregador, mesmo quando
anuncia várias funções. As marcações por palavras-chave são descritivas e não
devem ser interpretadas como um censo de vagas ou candidatos.
"""

from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

START_MONTH = pd.Timestamp("2022-11-01")
END_MONTH = pd.Timestamp("2025-05-01")
SEARCH_ENDPOINT = "https://hn.algolia.com/api/v1/search_by_date"
ITEM_ENDPOINT = "https://hn.algolia.com/api/v1/items/{thread_id}"
THREAD_TITLE = re.compile(r"^Ask HN: Who is hiring\? \(([A-Za-z]+ 20\d{2})\)$")

SOFTWARE_PATTERN = re.compile(
    r"\b(?:software|developer|devops|site reliability|sre|backend|back-end|front[- ]?end|"
    r"full[- ]?stack|web engineer|mobile engineer|ios engineer|android engineer|"
    r"data engineer|machine learning engineer|ml engineer|ai engineer|platform engineer|"
    r"cloud engineer|infrastructure engineer|security engineer|qa engineer|test automation|"
    r"embedded engineer|firmware engineer|engineering manager|technical lead|tech lead|"
    r"solutions architect)\b",
    re.IGNORECASE,
)
JUNIOR_PATTERN = re.compile(
    r"\b(?:junior|jr\.?|intern|internship|new grad(?:uate)?|graduate role|entry[- ]level|"
    r"early career|apprentice(?:ship)?|university hire|0[-– ]2 years|1[-– ]2 years)\b",
    re.IGNORECASE,
)
SENIOR_PATTERN = re.compile(
    r"\b(?:senior|sr\.?|staff|principal|distinguished|engineering manager|director of engineering|"
    r"head of engineering|software architect|solutions architect|technical lead|tech lead|team lead|"
    r"lead (?:software|backend|back-end|front[- ]?end|full[- ]?stack|platform|data|ml|ai|mobile|"
    r"security|infrastructure|cloud) engineer|(?:5|6|7|8|9|10)\+? years(?: of)? experience)\b",
    re.IGNORECASE,
)


def fetch_json(url: str, attempts: int = 4) -> dict:
    request = Request(url, headers={"User-Agent": "FIAP-CP4-educational-analysis/1.0"})
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=45) as response:
                return json.load(response)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Ramo de nova tentativa inalcançável")


def clean_html(value: str | None) -> str:
    if not value:
        return ""
    plain = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return re.sub(r"\s+", " ", plain).strip()


def find_threads() -> list[dict]:
    query = urlencode({"query": "Who is hiring", "tags": "story", "hitsPerPage": 1000})
    response = fetch_json(f"{SEARCH_ENDPOINT}?{query}")
    threads = []
    for hit in response.get("hits", []):
        match = THREAD_TITLE.match(hit.get("title") or "")
        if not match:
            continue
        month = pd.Timestamp(datetime.strptime(match.group(1), "%B %Y"))
        if START_MONTH <= month <= END_MONTH:
            threads.append({
                "thread_id": int(hit["objectID"]),
                "thread_title": hit["title"],
                "thread_month": month,
                "thread_created_at": hit.get("created_at"),
                "thread_total_comments_including_replies": hit.get("num_comments"),
            })
    threads.sort(key=lambda row: row["thread_month"])
    expected = len(pd.period_range(START_MONTH, END_MONTH, freq="M"))
    if len(threads) != expected:
        found = [row["thread_month"].strftime("%Y-%m") for row in threads]
        raise RuntimeError(f"Esperadas {expected} discussões mensais, mas foram encontradas {len(threads)}: {found}")
    return threads


def classify_comment(thread: dict, comment: dict) -> dict:
    text = clean_html(comment.get("text"))
    software_related = bool(SOFTWARE_PATTERN.search(text))
    mentions_junior = bool(JUNIOR_PATTERN.search(text)) if software_related else False
    mentions_senior = bool(SENIOR_PATTERN.search(text)) if software_related else False
    if mentions_junior and mentions_senior:
        group = "mixed"
    elif mentions_junior:
        group = "junior_or_intern"
    elif mentions_senior:
        group = "senior"
    elif software_related:
        group = "unspecified"
    else:
        group = "not_software"
    return {
        **thread,
        "comment_id": int(comment["id"]),
        "author": comment.get("author"),
        "comment_created_at": comment.get("created_at"),
        "text": text,
        "software_related": software_related,
        "mentions_junior_or_intern": mentions_junior,
        "mentions_senior": mentions_senior,
        "seniority_group": group,
    }


def aggregate_monthly(comments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, frame in comments.groupby("thread_month", sort=True):
        software = frame.loc[frame.software_related]
        total_software = len(software)
        junior = int(software.mentions_junior_or_intern.sum())
        senior = int(software.mentions_senior.sum())
        mixed = int((software.mentions_junior_or_intern & software.mentions_senior).sum())
        rows.append({
            "month_start": month,
            "thread_id": int(frame.thread_id.iloc[0]),
            "thread_title": frame.thread_title.iloc[0],
            "top_level_comments": int(len(frame)),
            "software_posts": int(total_software),
            "junior_intern_mentions": junior,
            "senior_mentions": senior,
            "mixed_mentions": mixed,
            "unspecified_seniority": int(
                ((software.seniority_group == "unspecified")).sum()
            ),
            "junior_intern_share": junior / total_software if total_software else 0.0,
            "senior_share": senior / total_software if total_software else 0.0,
        })
    monthly = pd.DataFrame(rows).sort_values("month_start").reset_index(drop=True)
    monthly["junior_intern_share_3m"] = monthly.junior_intern_share.rolling(3, min_periods=1).mean()
    monthly["senior_share_3m"] = monthly.senior_share.rolling(3, min_periods=1).mean()
    return monthly


def main():
    threads = find_threads()
    records = []
    for position, thread in enumerate(threads, start=1):
        item = fetch_json(ITEM_ENDPOINT.format(thread_id=thread["thread_id"]))
        for comment in item.get("children", []):
            if comment.get("id") and comment.get("text"):
                records.append(classify_comment(thread, comment))
        print(f"[{position:02d}/{len(threads)}] {thread['thread_month']:%Y-%m}: {len(item.get('children', []))} comentários de nível superior")

    comments = pd.DataFrame(records)
    comments["thread_month"] = pd.to_datetime(comments.thread_month)
    comments.to_csv(RAW / "hn_who_is_hiring_top_level.csv", index=False)

    monthly = aggregate_monthly(comments)
    monthly.to_csv(PROCESSED / "hn_seniority_monthly.csv", index=False)

    early = monthly.head(6)
    late = monthly.tail(6)
    summary = {
        "source": "API de busca do Hacker News fornecida pelo Algolia",
        "source_api": "https://hn.algolia.com/api",
        "official_hn_item_documentation": "https://github.com/HackerNews/API",
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage": {"start": str(START_MONTH.date()), "end": str(END_MONTH.date())},
        "unit": "Um comentário de nível superior corresponde a uma publicação de empregador; uma publicação pode anunciar várias funções.",
        "classification": {
            "software_pattern": SOFTWARE_PATTERN.pattern,
            "junior_pattern": JUNIOR_PATTERN.pattern,
            "senior_pattern": SENIOR_PATTERN.pattern,
            "mixed_rule": "Uma publicação de software que corresponde aos termos de início de carreira e sênior é marcada nas duas contagens de menções e rotulada como mista.",
        },
        "rows": int(len(comments)),
        "software_posts": int(comments.software_related.sum()),
        "early_six_month_average": {
            "junior_intern_posts": float(early.junior_intern_mentions.mean()),
            "senior_posts": float(early.senior_mentions.mean()),
            "junior_intern_share": float(early.junior_intern_share.mean()),
            "senior_share": float(early.senior_share.mean()),
        },
        "late_six_month_average": {
            "junior_intern_posts": float(late.junior_intern_mentions.mean()),
            "senior_posts": float(late.senior_mentions.mean()),
            "junior_intern_share": float(late.junior_intern_share.mean()),
            "senior_share": float(late.senior_share.mean()),
        },
        "limitations": [
            "O Hacker News é uma comunidade de tecnologia autoselecionada, não um portal de vagas representativo.",
            "A classificação por palavras-chave mede menções, não o número de vagas distintas.",
            "Um comentário de empregador de nível superior pode conter várias funções e níveis de senioridade.",
            "Comentários excluídos e funções descritas sem os termos correspondentes podem não ser identificados.",
        ],
    }
    (PROCESSED / "hn_seniority_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
