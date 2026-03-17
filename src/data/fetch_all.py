"""
Master data fetch script — runs all data sources in sequence.

Usage:
    python src/data/fetch_all.py --kenpom-email EMAIL --kenpom-password PASS

    # Skip KenPom (if you want to run other sources only):
    python src/data/fetch_all.py --skip-kenpom

    # Run everything:
    python src/data/fetch_all.py --kenpom-email you@email.com --kenpom-password pass123
"""

import argparse
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

STEPS = [
    ("Bart Torvik / T-Rank",       "fetch_torvik.py",          []),
    ("FiveThirtyEight / Elo",       "fetch_538.py",             []),
    ("WarrenNolan NET/RPI",         "fetch_warrennolan.py",     []),
    # Sports-Reference is slow (~42 years × 2 pages × 3s = ~4 min); run last
    ("Sports-Reference CBB",        "fetch_sports_reference.py", []),
]


def run_script(script_name: str, extra_args: list[str]) -> bool:
    script = SRC_DIR / script_name
    cmd = [sys.executable, str(script)] + extra_args
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(SRC_DIR.parent.parent))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Fetch all external data sources")
    parser.add_argument("--kenpom-email",    default=None)
    parser.add_argument("--kenpom-password", default=None)
    parser.add_argument("--skip-kenpom",     action="store_true")
    parser.add_argument("--skip-sportsref",  action="store_true",
                        help="Skip Sports-Reference (slow, ~4 minutes)")
    parser.add_argument("--only",            default=None,
                        help="Run only this script (e.g. --only fetch_torvik.py)")
    args = parser.parse_args()

    steps = STEPS.copy()

    # Optionally skip slow sources
    if args.skip_sportsref:
        steps = [(n, s, a) for n, s, a in steps if "sports_reference" not in s]

    # Run specific script only
    if args.only:
        steps = [(n, s, a) for n, s, a in steps if args.only in s]

    results = {}

    for name, script, extra in steps:
        ok = run_script(script, extra)
        results[name] = "OK" if ok else "FAILED"

    # KenPom — requires credentials
    if not args.skip_kenpom:
        if args.kenpom_email and args.kenpom_password:
            ok = run_script("fetch_kenpom.py", [
                "--email",    args.kenpom_email,
                "--password", args.kenpom_password,
            ])
            results["KenPom"] = "OK" if ok else "FAILED"
        else:
            print("\n[KenPom] Skipped — provide --kenpom-email and --kenpom-password to fetch")
            results["KenPom"] = "SKIPPED (no credentials)"

    print(f"\n{'='*60}")
    print("FETCH SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        icon = "✓" if status == "OK" else ("~" if "SKIPPED" in status else "✗")
        print(f"  {icon}  {name:35s}  {status}")

    print(f"\nAll raw data saved to: data/raw/")
    print("Next step: python src/02_feature_engineering_v2.py")


if __name__ == "__main__":
    main()
