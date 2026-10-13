"""Download / rebuild every raw input. Idempotent; skips files already present.

Usage:  python -m src.fetch_data            (everything)
        python -m src.fetch_data --only rebas milb retro

Sources (all public, see data/sources.md for licences):
  * REBAS 野球革命 Open Data (ODC-By 1.0) — CPBL 2024/2025 pitch-by-pitch
  * armstjc/milb-data-repository GitHub releases — mirror of MLB Stats API
    MiLB (AAA) pitch-level feed, season and game stats
  * chadwickbureau/retrosheet — MLB regular-season event files 2024/2025
  * chadwickbureau/register — MLBAM <-> Retrosheet ID crosswalk + birth years
  * chadwick 0.10.0 tools (cwdaily) — Retrosheet event files -> player-game lines

Network note: the official sites (stats.cpbl.com.tw, baseballsavant.mlb.com,
statsapi.mlb.com) were blocked by the execution environment's egress policy at
analysis time; this script only uses the GitHub-hosted mirrors above.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import urllib.parse
from pathlib import Path

import requests

from .common import ROOT, config, path, raw

CHUNK = 1 << 20


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=1200) as r:
        if r.status_code != 200:
            print(f"  [{r.status_code}] {url}")
            return False
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
        tmp.rename(dest)
    print(f"  ok {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
    return True


def fetch_rebas() -> None:
    src = config()["sources"]
    for year, tag in src["rebas_seasons"].items():
        name = f"CPBL-{year}-OpenData.zip"
        _download(f"{src['rebas_release']}/{tag}/{urllib.parse.quote(name)}", raw("rebas", name))


def fetch_milb() -> None:
    src = config()["sources"]
    base = src["milb_release"]
    for year, months in src["milb_pbp_months"].items():
        for m in months:
            f = f"{year}_{m}_aaa_pbp.csv"
            _download(f"{base}/pbp/{f}", raw("milb_pbp", f))
    for year, months in src["milb_game_months"].items():
        for m in months:
            f = f"{year}_{m}_aaa_player_game_stats.csv"
            _download(f"{base}/game_player_stats/{f}", raw("milb_game", f))
    for year in src["milb_season_years"]:
        for kind, tag in (("batting", "season_player_batting"), ("pitching", "season_player_pitching")):
            f = f"{year}_aaa_season_{kind}_stats.csv"
            _download(f"{base}/{tag}/{f}", raw("milb_season", f))


def _git_sparse(url: str, dest: Path, paths: list[str] | None) -> None:
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    if not (dest / ".git").exists():
        args = ["git", "clone", "--depth", "1", "--filter=blob:none"]
        if paths:
            args.append("--sparse")
        subprocess.run(args + [url, str(dest)], check=True, env=env)
    if paths:
        subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", *paths], check=True, env=env)


def chadwick_bin() -> Path:
    return path("external") / "cw" / "install" / "bin"


def fetch_retro() -> None:
    ext = path("external")
    seasons = [f"seasons/{y}" for y in config()["eligibility"]["seasons"]]
    _git_sparse(config()["sources"]["retrosheet_repo"], ext / "retrosheet", seasons)
    _git_sparse(config()["sources"]["register_repo"], ext / "register", None)
    # chadwick tools
    if not (chadwick_bin() / "cwdaily").exists():
        cw = ext / "cw"
        tar = cw / "chadwick-0.10.0.tar.gz"
        _download(config()["sources"]["chadwick_tarball"], tar)
        subprocess.run(["tar", "xzf", str(tar), "-C", str(cw)], check=True)
        srcdir = cw / "chadwick-0.10.0"
        subprocess.run(["./configure", f"--prefix={cw/'install'}"], cwd=srcdir, check=True,
                       stdout=subprocess.DEVNULL)
        subprocess.run(["make", "-j8"], cwd=srcdir, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["make", "install"], cwd=srcdir, check=True, stdout=subprocess.DEVNULL)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["rebas", "milb", "retro"])
    a = ap.parse_args(argv)
    if "rebas" in a.only:
        print("REBAS open data"); fetch_rebas()
    if "milb" in a.only:
        print("MiLB AAA mirror"); fetch_milb()
    if "retro" in a.only:
        print("Retrosheet + Chadwick"); fetch_retro()
    print("done; ROOT =", ROOT)


if __name__ == "__main__":
    main()
