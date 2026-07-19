#!/usr/bin/env python3
"""从 DBLP 爬取 CCF A 会议中与音频处理相关的论文。"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


DBLP_BASES = (
    "https://dblp.uni-trier.de",
    "https://dblp.org",
)

# 这里选的是音频/语音/多模态论文更可能出现的 CCF A 会议子集。
DEFAULT_VENUES = (
    ("ACM MM", "Multimedia", "mm"),
    ("ACL", "Natural Language Processing", "acl"),
    ("AAAI", "Artificial Intelligence", "aaai"),
    ("CVPR", "Computer Vision", "cvpr"),
    ("ICCV", "Computer Vision", "iccv"),
    ("ECCV", "Computer Vision", "eccv"),
    ("ICML", "Machine Learning", "icml"),
    ("ICLR", "Machine Learning", "iclr"),
    ("IJCAI", "Artificial Intelligence", "ijcai"),
    ("KDD", "Data Mining", "kdd"),
    ("NeurIPS", "Machine Learning", "nips"),
    ("SIGIR", "Information Retrieval", "sigir"),
    ("WWW", "Web", "www"),
    ("CHI", "Human-Computer Interaction", "chi"),
)

DEFAULT_KEYWORDS = (
    "acoustic",
    "audio",
    "auditory",
    "aural",
    "binaural",
    "clap",
    "codec",
    "music",
    "singing",
    "song",
    "sound",
    "speaker",
    "speech",
    "voice",
    "vocal",
    "wav",
    "waveform",
)

DEFAULT_EXCLUDE_PATTERNS = (
    "hate speech",
    "free speech",
    "offensive speech",
    "counter speech",
    "counterspeech",
)


@dataclass(frozen=True)
class Venue:
    name: str
    area: str
    dblp_path: str


def request_text(url: str, timeout: int, retries: int, sleep: float) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Codex CCF-A audio literature crawler; mailto:research@example.com)"
    }
    req = urllib.request.Request(url, headers=headers)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            return data.decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep * (attempt + 1))
    raise RuntimeError(f"请求失败: {url}: {last_error}")


def fetch_from_bases(path: str, timeout: int, retries: int, sleep: float) -> tuple[str, str]:
    errors = []
    for base in DBLP_BASES:
        url = f"{base}{path}"
        try:
            return request_text(url, timeout=timeout, retries=retries, sleep=sleep), url
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError("; ".join(errors))


def parse_xml(text: str, source: str) -> ET.Element:
    try:
        return ET.fromstring(text)
    except ET.ParseError as exc:
        raise RuntimeError(f"XML 解析失败: {source}: {exc}") from exc


def text_of(elem: ET.Element, tag: str) -> str:
    child = elem.find(tag)
    if child is None:
        return ""
    value = "".join(child.itertext())
    return " ".join(html.unescape(value).split())


def all_text(elem: ET.Element, tag: str) -> list[str]:
    values = []
    for child in elem.findall(tag):
        value = " ".join(html.unescape("".join(child.itertext())).split())
        if value:
            values.append(value)
    return values


def discover_proceedings(venue: Venue, years: set[int], timeout: int, retries: int, sleep: float) -> list[dict[str, str]]:
    path = f"/db/conf/{venue.dblp_path}/index.xml"
    text, source_url = fetch_from_bases(path, timeout=timeout, retries=retries, sleep=sleep)
    root = parse_xml(text, source_url)
    proceedings = []
    for elem in root.iter("proceedings"):
        year_text = text_of(elem, "year")
        if not year_text.isdigit() or int(year_text) not in years:
            continue
        url = text_of(elem, "url")
        title = text_of(elem, "title")
        if not url:
            continue
        page_path = "/" + re.sub(r"\.html(#.*)?$", ".xml", url)
        proceedings.append(
            {
                "venue": venue.name,
                "area": venue.area,
                "year": year_text,
                "title": title,
                "path": page_path,
            }
        )
    return proceedings


def keyword_hits(text: str, keywords: Iterable[str]) -> list[str]:
    lowered = text.lower()
    hits = []
    for keyword in keywords:
        pattern = r"(?<![a-z0-9])" + re.escape(keyword.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            hits.append(keyword)
    return hits


def is_excluded(text: str, patterns: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def doi_from_ees(ees: list[str]) -> str:
    for ee in ees:
        match = re.search(r"10\.\d{4,9}/\S+", ee, re.I)
        if match:
            return match.group(0).rstrip(".,;)")
    return ""


def crawl_proceeding(
    proc: dict[str, str],
    keywords: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
    timeout: int,
    retries: int,
    sleep: float,
) -> list[dict[str, str]]:
    text, source_url = fetch_from_bases(proc["path"], timeout=timeout, retries=retries, sleep=sleep)
    root = parse_xml(text, source_url)
    rows = []
    for elem in root.iter("inproceedings"):
        title = text_of(elem, "title")
        if is_excluded(title, exclude_patterns):
            continue
        hits = keyword_hits(title, keywords)
        if not hits:
            continue
        authors = all_text(elem, "author")
        ees = all_text(elem, "ee")
        url = text_of(elem, "url")
        rows.append(
            {
                "year": text_of(elem, "year") or proc["year"],
                "venue": proc["venue"],
                "ccf_area": proc["area"],
                "title": title,
                "authors": "; ".join(authors),
                "booktitle": text_of(elem, "booktitle") or proc["venue"],
                "pages": text_of(elem, "pages"),
                "doi": doi_from_ees(ees),
                "ee": "; ".join(ees),
                "dblp_url": f"https://dblp.org/{url}" if url else source_url,
                "matched_keywords": "; ".join(hits),
                "proceedings": proc["title"],
            }
        )
    return rows


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    result = []
    for row in rows:
        key = row["doi"].lower() if row["doi"] else re.sub(r"\W+", "", row["title"].lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "year",
        "venue",
        "ccf_area",
        "title",
        "authors",
        "booktitle",
        "pages",
        "doi",
        "ee",
        "dblp_url",
        "matched_keywords",
        "proceedings",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    rows: list[dict[str, str]],
    years: list[int],
    keywords: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_venue: dict[str, int] = {}
    for row in rows:
        by_venue[row["venue"]] = by_venue.get(row["venue"], 0) + 1

    lines = [
        "# CCF A 音频处理相关会议论文",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 年份范围：{min(years)}-{max(years)}",
        f"- 关键词：{', '.join(keywords)}",
        f"- 排除词组：{', '.join(exclude_patterns) if exclude_patterns else '无'}",
        f"- 命中论文数：{len(rows)}",
        "",
        "## 会议命中统计",
        "",
        "| 会议 | 数量 |",
        "| --- | ---: |",
    ]
    for venue, count in sorted(by_venue.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {venue} | {count} |")

    lines.extend(
        [
            "",
            "## 论文列表",
            "",
            "| 年份 | 会议 | 题名 | DOI/链接 | 命中词 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        link = row["doi"] or row["dblp_url"]
        title = row["title"].replace("|", "\\|")
        lines.append(
            f"| {row['year']} | {row['venue']} | {title} | {link} | {row['matched_keywords']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_start = datetime.now().year - 5
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=default_start)
    parser.add_argument("--end-year", type=int, default=datetime.now().year)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/literature"))
    parser.add_argument("--keywords", nargs="*", default=list(DEFAULT_KEYWORDS))
    parser.add_argument("--exclude-patterns", nargs="*", default=list(DEFAULT_EXCLUDE_PATTERNS))
    parser.add_argument("--venues", nargs="*", help="只抓取指定会议简称，例如 ACM MM ACL CVPR")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.8)
    parser.add_argument("--save-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    years = list(range(args.start_year, args.end_year + 1))
    venues = [Venue(*item) for item in DEFAULT_VENUES]
    if args.venues:
        wanted = {item.lower() for item in args.venues}
        venues = [venue for venue in venues if venue.name.lower() in wanted]
        if not venues:
            known = ", ".join(item[0] for item in DEFAULT_VENUES)
            raise SystemExit(f"没有匹配的会议简称。可选值: {known}")
    keywords = tuple(dict.fromkeys(keyword.lower() for keyword in args.keywords))
    exclude_patterns = tuple(dict.fromkeys(pattern.lower() for pattern in args.exclude_patterns))

    all_rows: list[dict[str, str]] = []
    errors: list[str] = []
    for venue in venues:
        try:
            proceedings = discover_proceedings(
                venue, set(years), timeout=args.timeout, retries=args.retries, sleep=args.sleep
            )
        except RuntimeError as exc:
            errors.append(f"{venue.name}: {exc}")
            continue
        print(f"{venue.name}: 发现 {len(proceedings)} 个 proceedings", file=sys.stderr)
        for proc in proceedings:
            try:
                rows = crawl_proceeding(
                    proc,
                    keywords,
                    exclude_patterns,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep=args.sleep,
                )
                all_rows.extend(rows)
                print(f"  {proc['year']} {proc['path']}: 命中 {len(rows)} 篇", file=sys.stderr)
                time.sleep(args.sleep)
            except RuntimeError as exc:
                errors.append(f"{venue.name} {proc['year']} {proc['path']}: {exc}")

    rows = dedupe_rows(all_rows)
    rows.sort(key=lambda item: (int(item["year"] or 0), item["venue"], item["title"]), reverse=True)

    stamp = f"{min(years)}_{max(years)}"
    csv_path = args.output_dir / f"ccf_a_audio_papers_{stamp}.csv"
    md_path = args.output_dir / f"ccf_a_audio_papers_{stamp}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, years, keywords, exclude_patterns)
    if args.save_json:
        json_path = args.output_dir / f"ccf_a_audio_papers_{stamp}.json"
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if errors:
        error_path = args.output_dir / f"ccf_a_audio_papers_{stamp}.errors.txt"
        error_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"部分会议/卷册抓取失败，详见: {error_path}", file=sys.stderr)

    print(f"写入: {csv_path}")
    print(f"写入: {md_path}")
    print(f"共命中 {len(rows)} 篇")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
