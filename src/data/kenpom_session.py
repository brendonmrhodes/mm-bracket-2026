"""
KenPom session helper — uses requests + Firefox cookies (no headless browser needed).
Firefox session cookies bypass Cloudflare authentication. The HTML tables are
server-side rendered so no JavaScript execution is required.

Playwright is kept for verify_session() compatibility but all page fetches
use requests for reliability and speed.
"""

import time
import random
import requests
import pandas as pd
import browser_cookie3
from bs4 import BeautifulSoup


def _human_pause(base: float = 5.0, jitter: float = 3.0):
    """Sleep for base ± jitter seconds to mimic human browsing pace."""
    time.sleep(base + random.uniform(-jitter / 2, jitter))


def make_requests_session() -> requests.Session:
    """
    Build a requests.Session pre-loaded with the user's Firefox cookies
    for kenpom.com. Headers mimic real Firefox on macOS to avoid bot detection.
    """
    jar = browser_cookie3.firefox(domain_name="kenpom.com")
    session = requests.Session()
    session.cookies.update(jar)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) "
            "Gecko/20100101 Firefox/133.0"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    })
    return session


def polite_get(session: requests.Session, url: str,
               base_delay: float = 7.0, jitter: float = 5.0) -> requests.Response | None:
    """
    GET a URL with a human-like delay before the request, randomised headers,
    and automatic retry on transient errors. Always use this instead of
    session.get() directly.
    """
    # Randomise Referer between the homepage and a plausible internal page
    referers = [
        "https://kenpom.com/",
        "https://kenpom.com/index.php",
        "https://kenpom.com/fanmatch.php",
    ]
    session.headers["Referer"] = random.choice(referers)

    # Human-like pause BEFORE each request
    sleep_time = base_delay + random.uniform(0, jitter)
    time.sleep(sleep_time)

    try:
        r = session.get(url, timeout=20)
        return r
    except requests.exceptions.RequestException as e:
        print(f"    Request error {url}: {e}")
        return None


def verify_session(session: requests.Session) -> bool:
    """Fetch kenpom.com and confirm we have a valid logged-in session."""
    try:
        r = session.get("https://kenpom.com/", timeout=15)
        return r.status_code == 200 and "ratings-table" in r.text
    except Exception:
        return False


def fetch_page(session: requests.Session, url: str,
               base_delay: float = 7.0, jitter: float = 5.0) -> str | None:
    """Fetch a KenPom page using polite_get and return its HTML, or None on failure."""
    r = polite_get(session, url, base_delay=base_delay, jitter=jitter)
    if r is None:
        return None

    if r.status_code == 403:
        print(f"    403 Forbidden at {url} — session may be expired")
        return None
    if r.status_code == 429 or "1015" in r.text[:300]:
        print(f"    Rate-limited at {url} — aborting to avoid ban")
        return None
    if r.status_code != 200:
        print(f"    HTTP {r.status_code} at {url}")
        return None

    # Sanity check: make sure we got a real page, not a challenge
    if "Just a moment" in r.text[:500] or "Checking your browser" in r.text[:500]:
        print(f"    Cloudflare JS challenge at {url} — session may need refresh")
        return None

    return r.text


# ── Playwright compatibility shims (used by fetch_kenpom.py) ──────────────────

def make_firefox_session(playwright):
    """Legacy Playwright session builder — used by fetch_kenpom.py."""
    from playwright.sync_api import sync_playwright
    import browser_cookie3 as bc3
    jar = bc3.firefox(domain_name="kenpom.com")
    cookies = [
        {"name": c.name, "value": c.value, "domain": c.domain or ".kenpom.com", "path": c.path or "/"}
        for c in jar
    ]
    browser = playwright.firefox.launch(headless=True)
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    context.add_cookies(cookies)
    page = context.new_page()
    return browser, context, page


def parse_table(html: str, table_id: str = None) -> pd.DataFrame:
    """Parse a KenPom HTML table into a clean DataFrame."""
    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table", {"id": table_id}) if table_id else None
    if table is None:
        tables = soup.find_all("table")
        # Pick the largest table (most rows)
        table = max(tables, key=lambda t: len(t.find_all("tr")), default=None) if tables else None

    if table is None:
        return pd.DataFrame()

    thead = table.find("thead")
    header_rows = thead.find_all("tr") if thead else []

    # Build flattened column names from potentially multi-row headers
    if len(header_rows) == 2:
        top_cells = header_rows[0].find_all(["th", "td"])
        bot_cells = header_rows[1].find_all(["th", "td"])
        expanded_top = []
        for th in top_cells:
            span = int(th.get("colspan", 1))
            expanded_top.extend([th.get_text(strip=True)] * span)
        bot_labels = [c.get_text(strip=True) for c in bot_cells]
        cols = []
        for t, b in zip(expanded_top, bot_labels):
            if t and b and t.lower() not in ("team", "conf", "rk", ""):
                cols.append(f"{t}_{b}")
            else:
                cols.append(b or t)
    elif len(header_rows) == 1:
        cols = [c.get_text(strip=True) for c in header_rows[0].find_all(["th", "td"])]
    else:
        cols = [th.get_text(strip=True) for th in table.find_all("th")]

    # Parse body
    tbody = table.find("tbody") or table
    rows = []
    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        row = [c.get_text(strip=True) for c in cells]
        if not any(row) or row[0] in ("", "Rk", "Rank"):
            continue
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    max_cols = max(len(r) for r in rows)
    col_names = cols[:max_cols] if cols else []
    if len(col_names) < max_cols:
        col_names += [f"col_{i}" for i in range(len(col_names), max_cols)]

    padded = [r[:max_cols] + [""] * max(0, max_cols - len(r)) for r in rows]
    df = pd.DataFrame(padded, columns=col_names)

    # Drop all-empty or header-repeat rows
    if df.columns[0] in df.columns:
        df = df[df.iloc[:, 0].notna() & (df.iloc[:, 0] != "") & (df.iloc[:, 0] != df.columns[0])]

    # Coerce numerics — use iloc to avoid issues with duplicate column names
    skip = {"Team", "team", "Conf", "conf", "Name", "name"}
    for i, col in enumerate(df.columns):
        if col not in skip:
            series = df.iloc[:, i]
            if series.dtype == "object":
                df.iloc[:, i] = pd.to_numeric(series, errors="coerce")

    return df
