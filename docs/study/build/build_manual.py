#!/usr/bin/env python3
"""Compile docs/study/[0-9]*.md into one print-first HTML file (no markdown library).

Usage:  python docs/study/build/build_manual.py  [--out docs/study/build/manual.html]

Supported markdown subset (exactly what the study manual uses):
  ATX headings (#..######) with anchors, fenced code (```lang), pipe tables,
  nested ordered/unordered lists (indentation-based), blockquotes (recursive),
  horizontal rules, paragraphs, and inline **bold**, *italic*, _italic_,
  `code` (protected from all other inline formatting), [links](url).
Files whose names start with "_" (e.g. _gaps.md) are excluded.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import html
import os
import re
import sys
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)

# ----------------------------------------------------------------------------- inline
_CODE_RE = re.compile(r"(`+)(.+?)\1")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
_ITAL_STAR_RE = re.compile(r"(?<![\*\w])\*(?=[^\s\*])(.+?)(?<=[^\s\*])\*(?![\*\w])")
_ITAL_US_RE = re.compile(r"(?<![\w])_(?=[^\s_])(.+?)(?<=[^\s_])_(?![\w])")


def inline(text: str) -> str:
    """Render inline markdown. Code spans are extracted first and restored last,
    so nothing inside them is ever bold/italic/link-processed."""
    slots: list[str] = []

    def stash(fragment: str) -> str:
        slots.append(fragment)
        return f"\x00{len(slots) - 1}\x00"

    text = _CODE_RE.sub(lambda m: stash(f"<code>{html.escape(m.group(2).strip() if m.group(1) != '`' else m.group(2))}</code>"), text)
    text = html.escape(text, quote=False)

    def link(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        return stash(f'<a href="{html.escape(url, quote=True)}">{label}</a>')

    text = _LINK_RE.sub(link, text)
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITAL_STAR_RE.sub(r"<em>\1</em>", text)
    text = _ITAL_US_RE.sub(r"<em>\1</em>", text)
    # restore (links may contain stashed code; loop until stable)
    for _ in range(3):
        text = re.sub(r"\x00(\d+)\x00", lambda m: slots[int(m.group(1))], text)
    return text


def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_]+", "-", text).strip("-")[:60] or "section"


# ----------------------------------------------------------------------------- blocks
_HEAD_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^(\s*)```(.*)$")
_HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")

PAGE_WIDTH_PT = 510.0      # A4 210mm minus 2×15mm margins ≈ 180mm ≈ 510pt
MONO_ADVANCE = 0.6         # JetBrains Mono advance width (em)


def split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells, cur, in_code, i = [], [], False, 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            cur.append("|"); i += 2; continue
        if c == "`":
            in_code = not in_code
        if c == "|" and not in_code:
            cells.append("".join(cur).strip()); cur = []
        else:
            cur.append(c)
        i += 1
    cells.append("".join(cur).strip())
    return cells


class Renderer:
    def __init__(self, module_key: str):
        self.module_key = module_key
        self.ids: set[str] = set()
        self.headings: list[tuple[int, str, str]] = []   # (level, id, html)
        self.stats = {"pre": 0, "table": 0, "h": [0] * 7}

    def unique_id(self, text: str) -> str:
        base = f"{self.module_key}-{slugify(text)}"
        hid, n = base, 2
        while hid in self.ids:
            hid = f"{base}-{n}"; n += 1
        self.ids.add(hid)
        return hid

    def render(self, lines: list[str]) -> str:
        out: list[str] = []
        i = 0
        para: list[str] = []

        def flush_para():
            if para:
                out.append("<p>" + inline(" ".join(x.strip() for x in para)) + "</p>")
                para.clear()

        while i < len(lines):
            line = lines[i]
            if not line.strip():
                flush_para(); i += 1; continue

            m = _FENCE_RE.match(line)
            if m:
                flush_para()
                indent, lang = len(m.group(1)), m.group(2).strip()
                body = []
                i += 1
                while i < len(lines) and not _FENCE_RE.match(lines[i]):
                    body.append(lines[i][indent:] if lines[i][:indent].strip() == "" else lines[i])
                    i += 1
                i += 1  # closing fence
                width = max((len(b.rstrip()) for b in body), default=1)
                size = min(8.6, 0.97 * PAGE_WIDTH_PT / (max(width, 1) * MONO_ADVANCE))
                cls = f' class="lang-{html.escape(lang)}"' if lang else ""
                out.append(f'<pre style="font-size:{size:.2f}pt"><code{cls}>'
                           + html.escape("\n".join(b.rstrip() for b in body)) + "</code></pre>")
                self.stats["pre"] += 1
                continue

            m = _HEAD_RE.match(line)
            if m:
                flush_para()
                level, text = len(m.group(1)), m.group(2)
                hid = self.unique_id(text)
                rendered = inline(text)
                self.headings.append((level, hid, rendered))
                self.stats["h"][level] += 1
                out.append(f'<h{level} id="{hid}">{rendered}</h{level}>')
                i += 1; continue

            if _HR_RE.match(line):
                flush_para(); out.append("<hr>"); i += 1; continue

            if line.lstrip().startswith("|") and i + 1 < len(lines) and _TABLE_SEP_RE.match(lines[i + 1]):
                flush_para()
                header = split_row(line)
                aligns = []
                for c in split_row(lines[i + 1]):
                    c = c.strip()
                    aligns.append("right" if c.endswith(":") and not c.startswith(":")
                                  else "center" if c.startswith(":") and c.endswith(":") else "")
                i += 2
                rows = []
                while i < len(lines) and lines[i].lstrip().startswith("|"):
                    rows.append(split_row(lines[i])); i += 1
                def cell(tag, txt, j):
                    a = aligns[j] if j < len(aligns) and aligns[j] else ""
                    st = f' style="text-align:{a}"' if a else ""
                    return f"<{tag}{st}>{inline(txt)}</{tag}>"
                h = "".join(cell("th", c, j) for j, c in enumerate(header))
                b = "".join("<tr>" + "".join(cell("td", c, j) for j, c in enumerate(r)) + "</tr>" for r in rows)
                out.append(f'<div class="tbl"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>')
                self.stats["table"] += 1
                continue

            if line.lstrip().startswith(">"):
                flush_para()
                quoted = []
                while i < len(lines) and lines[i].lstrip().startswith(">"):
                    q = lines[i].lstrip()[1:]
                    quoted.append(q[1:] if q.startswith(" ") else q)
                    i += 1
                out.append("<blockquote>" + self.render(quoted) + "</blockquote>")
                continue

            if _LIST_RE.match(line):
                flush_para()
                html_list, i = self.parse_list(lines, i)
                out.append(html_list)
                continue

            para.append(line); i += 1
        flush_para()
        return "\n".join(out)

    def parse_list(self, lines: list[str], i: int) -> tuple[str, int]:
        m = _LIST_RE.match(lines[i])
        base_indent = len(m.group(1))
        ordered = m.group(2)[0].isdigit()
        start = int(re.match(r"\d+", m.group(2)).group()) if ordered else 1
        items: list[list[str]] = []
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                # blank: continue list only if next non-blank belongs to it
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    nm = _LIST_RE.match(lines[j])
                    ind = len(lines[j]) - len(lines[j].lstrip())
                    if (nm and len(nm.group(1)) == base_indent
                            and nm.group(2)[0].isdigit() == ordered) or ind > base_indent:
                        if items:
                            items[-1].append("")
                        i = j; continue
                break
            m = _LIST_RE.match(line)
            ind = len(line) - len(line.lstrip())
            if m and len(m.group(1)) == base_indent:
                if m.group(2)[0].isdigit() != ordered:
                    break
                content_col = len(m.group(1)) + len(m.group(2)) + 1
                items.append([m.group(3)])
                items[-1].append(f"\x01{content_col}")  # marker of content column
                i += 1; continue
            if ind > base_indent and items:
                items[-1].append(line); i += 1; continue
            if m and len(m.group(1)) < base_indent:
                break
            if not m and items and ind <= base_indent and not _HEAD_RE.match(line) \
                    and not _FENCE_RE.match(line) and not line.lstrip().startswith(("|", ">")):
                items[-1].append(" " * (base_indent + 2) + line.strip()); i += 1; continue  # lazy continuation
            break
        tag = "ol" if ordered else "ul"
        start_attr = f' start="{start}"' if ordered and start != 1 else ""
        parts = [f"<{tag}{start_attr}>"]
        for it in items:
            first = it[0]
            col = int(it[1][1:])
            rest = [l[col:] if len(l) - len(l.lstrip()) >= col else l.lstrip() for l in it[2:]]
            if any(r.strip() for r in rest):
                body = self.render([first] + rest)
                # unwrap a single leading paragraph for tight look
                body = re.sub(r"^<p>(.*?)</p>", r"\1", body, count=1, flags=re.S)
            else:
                body = inline(first)
            parts.append(f"<li>{body}</li>")
        parts.append(f"</{tag}>")
        return "".join(parts), i


# ----------------------------------------------------------------------------- page
CSS = r"""
:root{
  --bg:#fbfaf7; --fg:#1d1d1b; --muted:#5f5f5a; --rule:#d9d6cf; --accent:#1f5fa8;
  --code-bg:#f2f0eb; --code-fg:#1d1d1b; --th-bg:#ece9e2; --quote:#7a5a1e; --quote-bg:#faf4e6;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#16171a; --fg:#e7e5df; --muted:#a3a19a; --rule:#3a3b40; --accent:#7fb2ee;
         --code-bg:#202227; --code-fg:#e7e5df; --th-bg:#26282d; --quote:#e3c27a; --quote-bg:#25221b; }
}
*{box-sizing:border-box}
html{font-size:10.5pt}
body{margin:0;background:var(--bg);color:var(--fg);
  font-family:"Source Serif 4","Source Serif Pro",Georgia,serif;line-height:1.5;
  font-variant-numeric:oldstyle-nums proportional-nums;hyphens:auto;}
main{max-width:180mm;margin:0 auto;padding:12mm 0}
h1,h2,h3,h4,h5,h6{font-family:"Inter",system-ui,sans-serif;line-height:1.25;color:var(--fg);
  font-variant-numeric:lining-nums;break-after:avoid;page-break-after:avoid;break-inside:avoid}
h1{font-size:20pt;font-weight:800;letter-spacing:-.01em;margin:0 0 6mm;padding-bottom:3mm;border-bottom:2px solid var(--fg)}
h2{font-size:13.5pt;font-weight:700;margin:8mm 0 3mm;padding-bottom:1mm;border-bottom:1px solid var(--rule)}
h3{font-size:11.5pt;font-weight:700;margin:5mm 0 2mm}
h4{font-size:10.5pt;font-weight:700;margin:4mm 0 1.5mm}
p{margin:0 0 2.6mm;orphans:3;widows:3}
li{margin:0 0 1mm;orphans:2;widows:2}
ul,ol{margin:0 0 3mm;padding-left:6mm}
a{color:var(--accent);text-decoration:none}
strong{font-weight:700}
code{font-family:"JetBrains Mono","DejaVu Sans Mono",ui-monospace,monospace;font-size:.86em;background:var(--code-bg);
  color:var(--code-fg);padding:.05em .25em;border-radius:2px;font-variant-ligatures:none;word-break:break-word}
pre{font-family:"JetBrains Mono","DejaVu Sans Mono",ui-monospace,monospace;background:var(--code-bg);color:var(--code-fg);
  padding:2.4mm 3mm;border-radius:3px;border:1px solid var(--rule);overflow:hidden;white-space:pre;
  line-height:1.32;margin:0 0 3.4mm;break-inside:avoid;page-break-inside:avoid;font-variant-ligatures:none}
pre code{background:none;padding:0;font-size:1em;word-break:normal}
blockquote{margin:0 0 3.4mm;padding:2mm 4mm;border-left:3px solid var(--quote);background:var(--quote-bg);
  break-inside:avoid;page-break-inside:avoid}
blockquote p:last-child{margin-bottom:0}
hr{border:0;border-top:1px solid var(--rule);margin:5mm 0}
.tbl{margin:0 0 3.6mm;break-inside:avoid;page-break-inside:avoid}
table{border-collapse:collapse;width:100%;font-size:8.3pt;line-height:1.35;font-family:"Inter",system-ui,sans-serif;
  font-variant-numeric:lining-nums tabular-nums}
th,td{border:1px solid var(--rule);padding:1.2mm 1.6mm;vertical-align:top;text-align:left;word-break:break-word}
th{background:var(--th-bg);font-weight:700}
td code,th code{font-size:.92em}
tr{break-inside:avoid;page-break-inside:avoid}
section.module{break-before:page;page-break-before:always}
section.module h1:not(:first-child){break-before:page;page-break-before:always}
.cover{height:257mm;display:flex;flex-direction:column;justify-content:center;padding:0 6mm}
.cover .kicker{font-family:"Inter",sans-serif;letter-spacing:.18em;text-transform:uppercase;font-size:9pt;color:var(--muted)}
.cover h1{font-size:34pt;border:0;margin:4mm 0 2mm;line-height:1.05}
.cover .sub{font-size:14pt;color:var(--muted);margin-bottom:14mm;font-style:italic}
.cover dl{font-family:"Inter",sans-serif;font-size:9.5pt;display:grid;grid-template-columns:38mm 1fr;gap:1.5mm 4mm;margin:0}
.cover dt{color:var(--muted)} .cover dd{margin:0}
.toc{break-before:page;page-break-before:always}
.toc h1{font-size:20pt}
.toc ol{list-style:none;padding:0;margin:0}
.toc > ol > li{margin:0 0 2.2mm;break-inside:avoid;font-family:"Inter",sans-serif;font-weight:700;font-size:10pt}
.toc ol ol{margin:.6mm 0 0 7mm;font-weight:400;font-size:8.4pt;columns:2;column-gap:8mm}
.toc ol ol li{margin:0 0 .4mm;break-inside:avoid}
@page{size:A4;margin:15mm 15mm 16mm 15mm;
  @bottom-center{content:counter(page);font-family:"Inter",sans-serif;font-size:8pt;color:#666}}
@page :first{@bottom-center{content:none}}
@media print{
  :root{--bg:#fff;--fg:#000;--muted:#333;--rule:#9a9a9a;--accent:#000;--code-bg:#f3f3f3;--code-fg:#000;
        --th-bg:#e9e9e9;--quote:#555;--quote-bg:#f6f6f6}
  html{font-size:10pt} body{background:#fff;color:#000} main{max-width:none;padding:0}
  a{color:#000}
}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Inter:wght@400;700;800&family=JetBrains+Mono:wght@400;700&'
         'family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,700;1,8..60,400&display=block">')


def build(out_path: str) -> dict:
    files = sorted(f for f in glob.glob(os.path.join(STUDY, "[0-9]*.md")))
    sections, toc, totals = [], [], {"pre": 0, "table": 0, "h1": 0, "h2": 0, "md_fences": 0, "md_tables": 0}
    for f in files:
        key = "m" + os.path.basename(f).split("-")[0]
        src = open(f, encoding="utf-8").read().splitlines()
        # source-side counts for verification
        fences = sum(1 for l in src if _FENCE_RE.match(l)) // 2
        tables = sum(1 for a, b in zip(src, src[1:]) if a.lstrip().startswith("|") and _TABLE_SEP_RE.match(b))
        r = Renderer(key)
        body = r.render(src)
        sections.append(f'<section class="module" id="{key}">{body}</section>')
        h1 = [h for h in r.headings if h[0] == 1]
        h2 = [h for h in r.headings if h[0] == 2]
        sub = "".join(f'<li><a href="#{hid}">{txt}</a></li>' for _, hid, txt in h2)
        toc.append(f'<li><a href="#{h1[0][1]}">{h1[0][2]}</a><ol>{sub}</ol></li>' if h1 else "")
        totals["pre"] += r.stats["pre"]; totals["table"] += r.stats["table"]
        totals["h1"] += r.stats["h"][1]; totals["h2"] += r.stats["h"][2]
        totals["md_fences"] += fences; totals["md_tables"] += tables
    totals["modules"] = len(files)
    today = dt.date.today().isoformat()
    cover = f"""<div class="cover">
<div class="kicker">Study manual · derived from source</div>
<h1>vla-finetune</h1>
<div class="sub">Every design decision, from the absolute basics to the end — defended from the code, not the docs.</div>
<dl><dt>Repository</dt><dd>AdityaKishore0698/VLA-finetune</dd>
<dt>Package</dt><dd>model_training · CLI <code>mtrain</code></dd>
<dt>Modules</dt><dd>{len(files)} (00 Orientation – 18 Data Provenance &amp; Demo)</dd>
<dt>Method</dt><dd>source read file by file; docs treated as claims; engine (rde) not available — engine-side behaviour marked inferred</dd>
<dt>Compiled</dt><dd>{today}</dd></dl></div>"""
    toc_html = f'<nav class="toc"><h1 id="contents">Contents</h1><ol>{"".join(toc)}</ol></nav>'
    doc = (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>vla-finetune — Study Manual</title>{FONTS}<style>{CSS}</style></head>'
           f'<body><main>{cover}{toc_html}{"".join(sections)}</main></body></html>')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return totals


# ----------------------------------------------------------------------------- verify
VOID = {"meta", "link", "br", "hr", "img", "input"}


class Checker(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.errors, self.text, self.skip = [], [], [], 0
        self.counts: dict[str, int] = {}

    def handle_starttag(self, tag, attrs):
        self.counts[tag] = self.counts.get(tag, 0) + 1
        if tag in VOID:
            return
        self.stack.append(tag)
        if tag in ("pre", "code", "style", "script"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}> (open: {self.stack[-3:]})")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
            return
        self.stack.pop()
        if tag in ("pre", "code", "style", "script"):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.text.append(data)


def verify(path: str, totals: dict) -> list[str]:
    c = Checker()
    c.feed(open(path, encoding="utf-8").read())
    problems = list(c.errors)
    if c.stack:
        problems.append(f"unclosed tags at end: {c.stack}")
    prose = "".join(c.text)
    for pat, name in [(r"\*\*", "bold markers"), (r"```", "code fences"), (r"\]\(", "link syntax"),
                      (r"(?m)^\s*#{1,6}\s", "heading markers"), (r"(?m)^\s*\|?\s*-{3,}\s*\|", "table separators"),
                      (r"`", "backticks")]:
        hits = re.findall(pat, prose)
        if hits:
            problems.append(f"leftover {name}: {len(hits)}")
    if c.counts.get("pre", 0) != totals["md_fences"]:
        problems.append(f"<pre> {c.counts.get('pre', 0)} != fenced blocks {totals['md_fences']}")
    if c.counts.get("table", 0) != totals["md_tables"]:
        problems.append(f"<table> {c.counts.get('table', 0)} != md tables {totals['md_tables']}")
    totals["html_counts"] = {k: c.counts.get(k, 0) for k in ("section", "h1", "h2", "h3", "table", "pre", "li", "blockquote")}
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "manual.html"))
    a = ap.parse_args()
    t = build(a.out)
    probs = verify(a.out, t)
    print(f"wrote {a.out}")
    print("source:", {k: t[k] for k in ("modules", "md_fences", "md_tables", "h1", "h2")})
    print("html:  ", t["html_counts"])
    if probs:
        print("PROBLEMS:"); [print("  -", p) for p in probs]; sys.exit(1)
    print("verification: OK (balanced tags, no leftover markdown, counts match)")
