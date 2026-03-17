"""
Shared Playwright Firefox session for KenPom scraping.
Firefox is not detected by Cloudflare. Used by both fetch_kenpom.py
and fetch_kenpom_extended.py.
"""

import time
import random
import pandas as pd
import browser_cookie3
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


def make_firefox_session(playwright):
    """
    Launch a Playwright Firefox browser pre-loaded with the user's
    existing Firefox cookies for kenpom.com. No login needed.
    """
    jar = browser_cookie3.firefox(domain_name="kenpom.com")
    cookies = [
        {"name": c.name, "value": c.value, "domain": c.domain or ".kenpom.com", "path": c.path or "/"}
        for c in jar
    ]

    browser = playwright.firefox.launch(headless=True)
    context = browser.new_context(viewport={"width": 1280, "height": 800})

    # Inject cookies before any navigation
    context.add_cookies(cookies)
    page = context.new_page()
    return browser, context, page


def verify_session(page) -> bool:
    """Navigate to kenpom.com and confirm we have a valid logged-in session."""
    page.goto("https://kenpom.com/", wait_until="domcontentloaded", timeout=20000)
    try:
        page.wait_for_selector("#ratings-table", timeout=8000)
        return True
    except PWTimeout:
        return False


def _human_pause(base: float = 5.0, jitter: float = 3.0):
    """Sleep for base ± jitter seconds to mimic human browsing pace."""
    time.sleep(base + random.uniform(-jitter / 2, jitter))


def fetch_page(page, url: str, table_selector: str = "table", timeout: int = 20000) -> str | None:
    """Navigate to a URL and return page HTML once the table is loaded.
    Automatically waits out Cloudflare 'Just a moment...' challenges."""
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
    except PWTimeout:
        print(f"    Timeout on navigation to {url}")
        return None
    except Exception as e:
        print(f"    Nav error {url}: {e}")
        return None

    # Wait out any Cloudflare JS challenge (up to 20s)
    for _ in range(10):
        title = page.title()
        if "1015" in page.content()[:200]:
            print(f"    Rate-limited (1015) at {url} — aborting")
            return None
        if "Just a moment" in title or "Checking your browser" in title:
            time.sleep(2)
        else:
            break
    else:
        print(f"    Cloudflare challenge did not clear for {url}")
        return None

    # Simulate brief human scroll before extracting
    try:
        page.evaluate("window.scrollBy(0, Math.random() * 300 + 100)")
    except Exception:
        pass

    # Now wait for the actual table
    try:
        page.wait_for_selector(table_selector, timeout=timeout)
        return page.content()
    except PWTimeout:
        print(f"    Timeout waiting for table at {url}")
        return None


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
