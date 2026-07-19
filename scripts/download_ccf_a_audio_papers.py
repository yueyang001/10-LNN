#!/usr/bin/env python3
"""下载 CCF A 音频处理论文的公开 PDF。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import mimetypes
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


USER_AGENT = "Mozilla/5.0 (Codex open PDF downloader; mailto:research@example.com)"


def safe_filename(text: str, max_len: int = 150) -> str:
    text = html.unescape(text)
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text[:max_len].strip(" .") or "paper"


def request(url: str, timeout: int) -> Tuple[bytes, str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        final_url = resp.geturl()
        content_type = resp.headers.get("Content-Type", "")
    return data, final_url, content_type


def is_pdf(data: bytes, content_type: str, url: str) -> bool:
    return (
        data.startswith(b"%PDF")
        or "application/pdf" in content_type.lower()
        or urllib.parse.urlparse(url).path.lower().endswith(".pdf")
    )


def absolutize(url: str, base: str) -> str:
    url = html.unescape(url)
    return urllib.parse.urljoin(base, url)


def split_links(value: str) -> List[str]:
    links = []
    for part in (value or "").split(";"):
        part = part.strip()
        if part.startswith("http"):
            links.append(part)
    return links


def openalex_pdf_candidates(row: Dict[str, str], timeout: int) -> List[str]:
    doi = (row.get("doi") or "").strip()
    if not doi:
        return []
    encoded = urllib.parse.quote(f"doi:{doi}", safe=":")
    url = f"https://api.openalex.org/works/{encoded}?select=best_oa_location,locations,open_access"
    try:
        data, _, _ = request(url, timeout)
        obj = json.loads(data.decode("utf-8", errors="replace"))
    except Exception:
        return []

    candidates = []
    for location in [obj.get("best_oa_location") or {}] + (obj.get("locations") or []):
        pdf_url = location.get("pdf_url")
        if pdf_url:
            candidates.append(pdf_url)
        landing = location.get("landing_page_url")
        if landing and landing.lower().endswith(".pdf"):
            candidates.append(landing)
    return list(dict.fromkeys(candidates))


def openalex_batch_candidates(rows: List[Dict[str, str]], timeout: int) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    dois = [row.get("doi", "").strip().lower() for row in rows if row.get("doi", "").strip()]
    for offset in range(0, len(dois), 25):
        chunk = dois[offset : offset + 25]
        query = urllib.parse.urlencode(
            {
                "filter": "doi:" + "|".join(chunk),
                "per-page": len(chunk),
                "select": "doi,best_oa_location,locations",
            }
        )
        try:
            data, _, _ = request(f"https://api.openalex.org/works?{query}", timeout)
            works = json.loads(data.decode("utf-8", errors="replace")).get("results", [])
        except Exception as exc:
            print(console_text(f"OpenAlex 批量查询失败: {exc}"), flush=True)
            continue
        for work in works:
            doi = (work.get("doi") or "").lower().replace("https://doi.org/", "")
            urls = []
            for location in [work.get("best_oa_location") or {}] + (work.get("locations") or []):
                pdf_url = location.get("pdf_url")
                if pdf_url:
                    urls.append(pdf_url)
                landing = location.get("landing_page_url")
                if landing and landing.lower().endswith(".pdf"):
                    urls.append(landing)
            result[doi] = list(dict.fromkeys(urls))
    return result


def page_pdf_candidates(url: str, timeout: int) -> List[str]:
    try:
        data, final_url, content_type = request(url, timeout)
    except Exception:
        return []
    if is_pdf(data, content_type, final_url):
        return [final_url]

    text = data.decode("utf-8", errors="replace")
    candidates = []

    for match in re.finditer(
        r"""<meta[^>]+(?:name|property)=["'](?:citation_pdf_url|og:pdf)["'][^>]+content=["']([^"']+)["']""",
        text,
        re.I,
    ):
        candidates.append(absolutize(match.group(1), final_url))

    # 常见会议页面里 PDF 链接一般在 href 中。
    for match in re.finditer(r"""href=["']([^"']+)["']""", text, re.I):
        href = absolutize(match.group(1), final_url)
        lowered = href.lower()
        if ".pdf" in lowered or "/download/" in lowered:
            candidates.append(href)

    # NeurIPS abstract 页面可由 Abstract 链接推导 Paper PDF。
    if "papers.nips.cc" in final_url or "proceedings.neurips.cc" in final_url:
        parsed = urllib.parse.urlparse(final_url)
        paper_path = re.sub(r"-Abstract(-[^/]+)?\.html$", r"-Paper\1.pdf", parsed.path)
        if paper_path != parsed.path:
            candidates.append(urllib.parse.urlunparse(parsed._replace(path=paper_path)))
        paper_path = paper_path.replace("/hash/", "/file/")
        candidates.append(urllib.parse.urlunparse(parsed._replace(path=paper_path)))

    return list(dict.fromkeys(candidates))


def deterministic_pdf_candidates(url: str) -> List[str]:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path
    candidates = []

    if path.lower().endswith(".pdf"):
        candidates.append(url)

    if "aclanthology.org" in host and not path.lower().endswith(".pdf"):
        candidates.append(urllib.parse.urlunparse(parsed._replace(path=path.rstrip("/") + ".pdf")))

    if "proceedings.mlr.press" in host and path.lower().endswith(".html"):
        stem = Path(path).stem
        parent = str(Path(path).parent).replace("\\", "/")
        volume = Path(path).parent.name
        candidates.append(f"https://raw.githubusercontent.com/mlresearch/{volume}/main/assets/{stem}/{stem}.pdf")
        candidates.append(urllib.parse.urlunparse(parsed._replace(path=f"{parent}/{stem}/{stem}.pdf")))

    if "openreview.net" in host and parsed.path == "/forum":
        query = urllib.parse.parse_qs(parsed.query)
        paper_id = (query.get("id") or [""])[0]
        if paper_id:
            candidates.append(f"https://openreview.net/pdf?id={paper_id}")

    if "ijcai.org" in host:
        match = re.search(r"/proceedings/(\d{4})/(\d+)$", path)
        if match:
            year, number = match.groups()
            candidates.append(f"https://www.ijcai.org/proceedings/{year}/{int(number):04d}.pdf")

    if "arxiv.org" in host and path.startswith("/abs/"):
        candidates.append(urllib.parse.urlunparse(parsed._replace(path=path.replace("/abs/", "/pdf/", 1))))

    if "papers.nips.cc" in host or "proceedings.neurips.cc" in host:
        paper_path = re.sub(r"-Abstract(-[^/]+)?\.html$", r"-Paper\1.pdf", path)
        if paper_path != path:
            candidates.append(urllib.parse.urlunparse(parsed._replace(path=paper_path)))
            candidates.append(urllib.parse.urlunparse(parsed._replace(path=paper_path.replace("/hash/", "/file/"))))

    return list(dict.fromkeys(candidates))


def row_candidates(row: Dict[str, str], timeout: int, fast: bool) -> List[str]:
    candidates = []
    candidates.extend(split_links(row.get("ee", "")))
    doi = (row.get("doi") or "").strip()
    if doi and not fast:
        candidates.append(f"https://doi.org/{doi}")
    if not fast:
        candidates.extend(openalex_pdf_candidates(row, timeout))

    expanded = []
    for url in candidates:
        if not url.startswith("http"):
            continue
        expanded.extend(deterministic_pdf_candidates(url))
        if not fast:
            expanded.extend(page_pdf_candidates(url, timeout))
    return list(dict.fromkeys(expanded))


def download_pdf_with_curl(url: str, dest: Path, timeout: int) -> Tuple[bool, str]:
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        "curl.exe",
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--max-time",
        str(timeout),
        "--connect-timeout",
        str(min(timeout, 5)),
        "--user-agent",
        USER_AGENT,
        "--output",
        str(tmp),
        url,
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout + 5)
    except subprocess.TimeoutExpired:
        if tmp.exists():
            tmp.unlink()
        return False, "curl 超时"
    if proc.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        return False, proc.stderr.decode("utf-8", errors="replace").strip() or f"curl {proc.returncode}"
    data = tmp.read_bytes()[:8]
    if not data.startswith(b"%PDF"):
        tmp.unlink()
        return False, "不是 PDF"
    tmp.replace(dest)
    return True, url


def download_pdf(url: str, dest: Path, timeout: int, use_curl: bool) -> Tuple[bool, str]:
    if use_curl:
        return download_pdf_with_curl(url, dest, timeout)
    try:
        data, final_url, content_type = request(url, timeout)
        if not is_pdf(data, content_type, final_url):
            return False, f"不是 PDF: {content_type or final_url}"
        dest.write_bytes(data)
        return True, final_url
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, str(exc)


def paper_id(row: Dict[str, str]) -> str:
    key = row.get("doi") or row.get("dblp_url") or row.get("title") or ""
    return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:10]


def console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def read_existing_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        return {row["paper_id"]: row for row in csv.DictReader(fh)}


def write_manifest(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    fields = [
        "paper_id",
        "status",
        "year",
        "venue",
        "title",
        "doi",
        "pdf_path",
        "source_url",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("docs/literature/ccf_a_audio_papers_2024_2024.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/literature/pdfs_2024_2024"))
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0, help="调试用：只处理前 N 篇，0 表示全部")
    parser.add_argument("--retry-failed", action="store_true", help="重新尝试之前失败的条目")
    parser.add_argument("--fast", action="store_true", help="只尝试稳定可推导的公开 PDF 链接")
    parser.add_argument("--max-candidates", type=int, default=3, help="每篇最多尝试的 PDF 候选链接数")
    parser.add_argument("--use-curl", action="store_true", help="用 curl.exe 下载，超时控制更可靠")
    parser.add_argument("--start-index", type=int, default=1, help="从第 N 条开始处理（从 1 计数）")
    parser.add_argument("--end-index", type=int, default=0, help="处理到第 N 条，0 表示末尾")
    parser.add_argument("--scan-only", action="store_true", help="只扫描磁盘并重建清单")
    parser.add_argument("--openalex-batch", action="store_true", help="批量查询开放获取 PDF")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "download_manifest.csv"
    existing = read_existing_manifest(manifest_path)

    with args.csv.open("r", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    start_index = max(args.start_index, 1)
    end_index = args.end_index if args.end_index > 0 else len(rows)
    rows = rows[start_index - 1 : end_index]

    manifest: List[Dict[str, str]] = []
    downloaded = skipped = failed = 0

    for index, row in enumerate(rows, start=1):
        pid = paper_id(row)
        old = existing.get(pid)
        if old and old.get("status") == "downloaded" and Path(old.get("pdf_path", "")).exists():
            manifest.append(old)
            skipped += 1
            continue
        if old and old.get("status") == "failed" and not args.retry_failed:
            manifest.append(old)
            skipped += 1
            continue

        prefix = f"{row.get('year', '')}_{row.get('venue', '')}_{pid}"
        dest = args.output_dir / f"{safe_filename(prefix + '_' + row.get('title', ''))}.pdf"
        line = f"[{index}/{len(rows)}] {row.get('venue')} - {row.get('title')[:80]}"
        print(console_text(line), flush=True)

        status = "failed"
        source_url = ""
        message = "未找到公开 PDF"
        candidates = row_candidates(row, args.timeout, args.fast)
        if args.max_candidates > 0:
            candidates = candidates[: args.max_candidates]
        for candidate in candidates:
            ok, msg = download_pdf(candidate, dest, args.timeout, args.use_curl)
            source_url = candidate
            if ok:
                status = "downloaded"
                message = msg
                downloaded += 1
                break
            message = msg
            time.sleep(args.sleep)
        if status != "downloaded":
            failed += 1

        manifest.append(
            {
                "paper_id": pid,
                "status": status,
                "year": row.get("year", ""),
                "venue": row.get("venue", ""),
                "title": row.get("title", ""),
                "doi": row.get("doi", ""),
                "pdf_path": str(dest if status == "downloaded" else ""),
                "source_url": source_url,
                "message": message,
            }
        )
        write_manifest(manifest_path, manifest)
        time.sleep(args.sleep)

    # 保留本轮未处理但已有记录的条目。
    seen = {row["paper_id"] for row in manifest}
    manifest.extend(row for pid, row in existing.items() if pid not in seen)
    write_manifest(manifest_path, manifest)

    print(f"下载完成: downloaded={downloaded}, skipped={skipped}, failed={failed}")
    print(f"清单: {manifest_path}")
    return 0


def resumable_main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "download_manifest.csv"
    manifest_by_id = read_existing_manifest(manifest_path)

    with args.csv.open("r", newline="", encoding="utf-8-sig") as fh:
        all_rows = list(csv.DictReader(fh))
    if args.limit:
        all_rows = all_rows[: args.limit]

    start_index = max(args.start_index, 1)
    end_index = args.end_index if args.end_index > 0 else len(all_rows)
    selected = [
        (index, row)
        for index, row in enumerate(all_rows, start=1)
        if start_index <= index <= end_index
    ]
    oa_candidates = (
        openalex_batch_candidates([row for _, row in selected], args.timeout)
        if args.openalex_batch
        else {}
    )
    downloaded = skipped = failed = 0

    for index, row in selected:
        pid = paper_id(row)
        prefix = f"{row.get('year', '')}_{row.get('venue', '')}_{pid}"
        dest = args.output_dir / f"{safe_filename(prefix + '_' + row.get('title', ''))}.pdf"
        old = manifest_by_id.get(pid, {})

        if dest.exists():
            manifest_by_id[pid] = {
                "paper_id": pid,
                "status": "downloaded",
                "year": row.get("year", ""),
                "venue": row.get("venue", ""),
                "title": row.get("title", ""),
                "doi": row.get("doi", ""),
                "pdf_path": str(dest),
                "source_url": old.get("source_url", ""),
                "message": old.get("message", "已在磁盘中"),
            }
            skipped += 1
            continue
        if args.scan_only:
            continue
        if old.get("status") == "failed" and not args.retry_failed:
            skipped += 1
            continue

        line = f"[{index}/{len(all_rows)}] {row.get('venue')} - {row.get('title')[:80]}"
        print(console_text(line), flush=True)
        status = "failed"
        source_url = ""
        message = "未找到公开 PDF"
        if args.openalex_batch:
            doi = row.get("doi", "").strip().lower()
            candidates = oa_candidates.get(doi, []) + row_candidates(row, args.timeout, True)
            candidates = list(dict.fromkeys(candidates))
        else:
            candidates = row_candidates(row, args.timeout, args.fast)
        if args.max_candidates > 0:
            candidates = candidates[: args.max_candidates]
        for candidate in candidates:
            ok, message = download_pdf(candidate, dest, args.timeout, args.use_curl)
            source_url = candidate
            if ok:
                status = "downloaded"
                downloaded += 1
                break
            time.sleep(args.sleep)
        if status != "downloaded":
            failed += 1

        manifest_by_id[pid] = {
            "paper_id": pid,
            "status": status,
            "year": row.get("year", ""),
            "venue": row.get("venue", ""),
            "title": row.get("title", ""),
            "doi": row.get("doi", ""),
            "pdf_path": str(dest if status == "downloaded" else ""),
            "source_url": source_url,
            "message": message,
        }
        write_manifest(manifest_path, manifest_by_id.values())
        time.sleep(args.sleep)

    write_manifest(manifest_path, manifest_by_id.values())
    print(f"下载完成: downloaded={downloaded}, skipped={skipped}, failed={failed}")
    print(f"清单: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(resumable_main())
