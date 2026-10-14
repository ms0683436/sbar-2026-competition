"""Amendment 7 — stats.cpbl.com.tw public player pages (identities and velocity tables).

The site documents an AI access path (llms.txt, /.well-known/agent-skills/cpbl-navigation):
public pages return Markdown when requested with `Accept: text/markdown`, and `/api/` must not
be called (robots.txt). Each player page shows ONE default view (the player's current level
and the current season); other seasons or levels need the page's filters, which this module
does not touch. Pages are cached under data/raw/cpbl_site/pages/ (git-ignored, not
redistributed). The site's terms ask for written authorization for any use; Johnny decided on
2026-10-14 to use it without one for the backtest only (reports/limitations.md L10).

Usage:  python -m src.cpbl_site    (fetch missing pages, sequential, ~1 request per second)
"""
from __future__ import annotations

import re
import time
import urllib.request

import pandas as pd

from .common import raw

SITE = "https://stats.cpbl.com.tw"
UA = "sbar-2026-research (public pages as Markdown)"


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/markdown", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def player_urls() -> list[str]:
    f = raw("cpbl_site", "sitemap.xml")
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(_get(f"{SITE}/sitemap.xml"), encoding="utf-8")
    return re.findall(r"<loc>(https://stats\.cpbl\.com\.tw/players/\d+)</loc>", f.read_text(encoding="utf-8"))


def fetch_pages(pause: float = 0.7) -> None:
    d = raw("cpbl_site", "pages")
    d.mkdir(parents=True, exist_ok=True)
    for u in player_urls():
        f = d / f"{u.rsplit('/', 1)[1]}.md"
        if f.exists() and f.stat().st_size:
            continue
        f.write_text(_get(u), encoding="utf-8")
        time.sleep(pause)


def _field(t: str, label: str) -> str | None:
    m = re.search(rf"{label}\n\n(.*)\n", t)
    return m.group(1).strip() if m else None


def identities() -> pd.DataFrame:
    """Name, team, position, hand, birth date, original name and default view of every page."""
    rows = []
    for f in sorted(raw("cpbl_site", "pages").glob("*.md")):
        t = f.read_text(encoding="utf-8")
        h = re.search(r"\\# (\S*) \| (.*?) \| (.*)\n", t)
        pr = t.split("## 中職百分位排名(PR)")[1] if "## 中職百分位排名(PR)" in t else ""
        sel = re.search(r"一軍例行賽一軍季後挑戰賽一軍總冠軍賽二軍例行賽二軍總冠軍賽\n\n(.*)\n\n(\d{4}|暫無資料)", pr)
        name = re.search(r"\n## (.*)\n\n\\#", t)
        rows.append({"site_id": f.stem, "site_name": clean_name(name.group(1)) if name else None,
                     "team": h.group(2) if h else None, "position": h.group(3) if h else None,
                     "throws": _field(t, "投球"), "birth": _field(t, "生日"), "original_name": _field(t, "原名"),
                     "default_game_type": sel.group(1) if sel else None, "default_season": sel.group(2) if sel else None})
    return pd.DataFrame(rows)


def clean_name(s: str) -> str:
    return re.sub(r"^[\\*#]+", "", str(s)).strip()


def velocity() -> pd.DataFrame:
    """Rows of each page's 球種x球速轉速 table (default view only): season, hand, pitch group,
    pitches, max / avg velocity (km/h), max / avg spin."""
    out = []
    ident = identities().set_index("site_id")
    for f in sorted(raw("cpbl_site", "pages").glob("*.md")):
        t = f.read_text(encoding="utf-8")
        for m in re.finditer(r"\|?\s*(\d{4}) \| ([左右]) \| (\S+) \| (\d+) \| ([\d.]+|-) \| ([\d.]+|-) \| ([\d.]+|-) \| ([\d.]+|-) \|", t):
            y, hand, grp, n, vmax, vavg, smax, savg = m.groups()
            num = lambda x: float(x) if x != "-" else None
            out.append({"site_id": f.stem, "site_name": ident.site_name.get(f.stem),
                        "default_game_type": ident.default_game_type.get(f.stem), "season": int(y), "hand": hand,
                        "pitch_group": grp, "pitches": int(n), "velo_max_kmh": num(vmax), "velo_avg_kmh": num(vavg),
                        "spin_max": num(smax), "spin_avg": num(savg)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    fetch_pages()
    print(identities().default_game_type.value_counts())
