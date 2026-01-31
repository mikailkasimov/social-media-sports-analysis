
# run_perspective.py
# Scores 4chan + reddit posts/comments with Perspective API in chunks
# - Cleans 4chan HTML
# - Skips rows already scored (using row_key)
# - Repairs/normalizes existing output CSVs even if they are headerless/shifted
# - Safe temp-file commits (won’t crash if tmp not written)

#!/usr/bin/env python3
import os
import sys
import time
import json
import math
import hashlib
import logging
import sqlite3
import warnings
from typing import Dict, Optional, Tuple, List

import pandas as pd
import requests

# Optional: BeautifulSoup only needed for 4chan HTML cleanup
try:
    from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False

# Optional tqdm
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


ATTRIBUTES = [
    "TOXICITY",
    "SEVERE_TOXICITY",
    "INSULT",
    "PROFANITY",
    "THREAT",
    "IDENTITY_ATTACK",
    "SEXUALLY_EXPLICIT",
]

CHAN_OUT_COLS = ["thread_number", "post_number", "text_body"] + [a.lower() for a in ATTRIBUTES]
REDDIT_COMMENTS_OUT_COLS = ["link_id", "text"] + [a.lower() for a in ATTRIBUTES]
REDDIT_POSTS_OUT_COLS = ["post_id", "title", "text"] + [a.lower() for a in ATTRIBUTES]


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def clean_4chan_text(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    # If bs4 isn't available, do a minimal cleanup.
    if not _HAS_BS4:
        return s.replace("<br>", "\n").strip()

    # If this doesn't look like HTML at all, avoid parsing (also avoids URL warnings).
    looks_like_html = ("<" in s) or ("&" in s)
    if not looks_like_html:
        return s.strip()

    soup = BeautifulSoup(s, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text(" ", strip=True)
    # Unescape HTML entities without importing html explicitly
    text = bytes(text, "utf-8").decode("unicode_escape", errors="ignore")
    return text.strip()


def detect_has_header(path: str, expected_first_col: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
        return expected_first_col in first
    except Exception:
        return False


class PerspectiveClient:
    def __init__(
        self,
        api_key: str,
        sleep_s: float = 1.05,
        max_retries: int = 5,
        timeout_s: int = 30,
        truncate_chars: int = 3000,
    ):
        self.api_key = api_key
        self.sleep_s = sleep_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.truncate_chars = truncate_chars
        self.session = requests.Session()
        self.url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"

        self.requested_attributes = {a: {} for a in ATTRIBUTES}

    def analyze(self, text: str) -> Optional[Dict[str, float]]:
        if not isinstance(text, str):
            return None
        text = text.strip()
        if not text:
            return None

        # Truncate to avoid API errors on very long text
        if self.truncate_chars and len(text) > self.truncate_chars:
            text = text[: self.truncate_chars]

        payload = {
            "comment": {"text": text},
            "languages": ["en"],
            "requestedAttributes": self.requested_attributes,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.post(self.url, json=payload, timeout=self.timeout_s)

                # Rate limit / quota backoff
                if r.status_code in (429, 503):
                    backoff = self.sleep_s * (2 ** (attempt - 1))
                    jitter = 0.1 * backoff
                    time.sleep(backoff + jitter)
                    continue

                # Other errors
                r.raise_for_status()
                data = r.json()

                out: Dict[str, float] = {}
                scores = data.get("attributeScores", {})
                for a in ATTRIBUTES:
                    v = scores.get(a, {}).get("summaryScore", {}).get("value", None)
                    out[a.lower()] = v

                # IMPORTANT: sleep ONCE per request (not per attribute)
                time.sleep(self.sleep_s)
                return out

            except Exception as e:
                if attempt == self.max_retries:
                    logging.warning(f"Perspective API failed after {self.max_retries} tries: {e}")
                    return None
                backoff = self.sleep_s * (2 ** (attempt - 1))
                time.sleep(backoff)

        return None


class StateDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

        self.conn.execute("CREATE TABLE IF NOT EXISTS chan_keys (k TEXT PRIMARY KEY);")
        self.conn.execute("CREATE TABLE IF NOT EXISTS reddit_comment_keys (k TEXT PRIMARY KEY);")
        self.conn.execute("CREATE TABLE IF NOT EXISTS reddit_post_keys (k TEXT PRIMARY KEY);")
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

    def ensure_loaded_from_csv(self, table: str, csv_path: str, key_builder, expected_first_col: str) -> None:
        if not os.path.exists(csv_path):
            logging.info(f"[STATE] {csv_path} does not exist yet; nothing to preload for {table}.")
            return

        # If table already has keys, assume it’s already preloaded
        cur = self.conn.execute(f"SELECT COUNT(1) FROM {table};")
        n = cur.fetchone()[0]
        if n and n > 0:
            logging.info(f"[STATE] {table} already has {n:,} keys; skipping preload from {csv_path}.")
            return

        has_header = detect_has_header(csv_path, expected_first_col)
        logging.info(f"[STATE] Preloading keys for {table} from {csv_path} (header={has_header})...")

        inserted = 0
        chunksize = 200_000  # preload in big chunks
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, header=0 if has_header else None):
            # If no header, assign based on file type by inferring from expected_first_col
            # We only need to build keys, so we can just access by position safely.
            keys = []
            for _, row in chunk.iterrows():
                k = key_builder(row, header=has_header)
                if k:
                    keys.append((k,))
            if keys:
                self.conn.executemany(f"INSERT OR IGNORE INTO {table}(k) VALUES (?);", keys)
                inserted += len(keys)
                self.conn.commit()

        logging.info(f"[STATE] Preload done for {table}. Insert attempted: {inserted:,} rows.")

    def keys_exist(self, table: str, keys: List[str]) -> set:
        if not keys:
            return set()
        placeholders = ",".join(["?"] * len(keys))
        cur = self.conn.execute(f"SELECT k FROM {table} WHERE k IN ({placeholders});", keys)
        return {r[0] for r in cur.fetchall()}

    def add_keys(self, table: str, keys: List[str]) -> None:
        if not keys:
            return
        self.conn.executemany(f"INSERT OR IGNORE INTO {table}(k) VALUES (?);", [(k,) for k in keys])
        self.conn.commit()


def process_chan_posts(
    in_path: str,
    out_path: str,
    state: StateDB,
    client: PerspectiveClient,
    chunksize: int = 100,
):
    logging.info(f"[CHAN] Input: {in_path}")
    logging.info(f"[CHAN] Output: {out_path}")

    def key_from_output_row(row, header: bool) -> Optional[str]:
        # If headerless, columns are numeric; thread_number at col0, post_number at col1
        try:
            t = row["thread_number"] if header else row.iloc[0]
            p = row["post_number"] if header else row.iloc[1]
            return f"{int(t)}:{int(p)}"
        except Exception:
            return None

    state.ensure_loaded_from_csv(
        table="chan_keys",
        csv_path=out_path,
        key_builder=key_from_output_row,
        expected_first_col="thread_number",
    )

    if not os.path.exists(in_path):
        logging.warning(f"[CHAN] Missing input file: {in_path} (skipping)")
        return

    # If output exists but has no header, keep writing headerless to avoid mixing
    out_has_header = detect_has_header(out_path, "thread_number") if os.path.exists(out_path) else True
    write_header = (not os.path.exists(out_path)) and out_has_header

    processed_total = 0
    skipped_total = 0

    reader = pd.read_csv(in_path, chunksize=chunksize, usecols=["thread_number", "post_number", "text_body"])
    for i, chunk in enumerate(reader, start=1):
        chunk = chunk.copy()
        chunk["text_body"] = chunk["text_body"].apply(clean_4chan_text)

        keys = [f"{int(r.thread_number)}:{int(r.post_number)}" for r in chunk.itertuples(index=False)]
        already = state.keys_exist("chan_keys", keys)
        mask = [k not in already for k in keys]
        todo = chunk.loc[mask].reset_index(drop=True)

        skipped = len(chunk) - len(todo)
        skipped_total += skipped
        if len(todo) == 0:
            logging.info(f"[CHAN] Chunk {i}: skipped {skipped} (all already processed)")
            continue

        scores_rows = []
        done_keys = []
        iterable = todo["text_body"]
        if tqdm is not None:
            iterable = tqdm(iterable, total=len(todo), desc=f"[CHAN] chunk {i}")

        for idx, text in enumerate(iterable):
            if not text:
                scores_rows.append({a.lower(): None for a in ATTRIBUTES})
            else:
                scores = client.analyze(text)
                if scores is None:
                    scores_rows.append({a.lower(): None for a in ATTRIBUTES})
                else:
                    scores_rows.append(scores)

        scores_df = pd.DataFrame(scores_rows)
        out_df = pd.concat([todo[["thread_number", "post_number", "text_body"]], scores_df], axis=1)

        # Write
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        out_df.to_csv(out_path, mode="a", header=write_header, index=False)

        # Update state keys
        done_keys = [f"{int(r.thread_number)}:{int(r.post_number)}" for r in todo.itertuples(index=False)]
        state.add_keys("chan_keys", done_keys)

        processed_total += len(todo)
        logging.info(f"[CHAN] Chunk {i}: wrote {len(todo)} rows, skipped {skipped}")

    logging.info(f"[CHAN] Done. Total wrote={processed_total:,}, skipped={skipped_total:,}")


def process_reddit_comments(
    in_path: str,
    out_path: str,
    state: StateDB,
    client: PerspectiveClient,
    chunksize: int = 100,
):
    logging.info(f"[RCOMMENTS] Input: {in_path}")
    logging.info(f"[RCOMMENTS] Output: {out_path}")

    def key_from_output_row(row, header: bool) -> Optional[str]:
        # If headerless: link_id at col0, text at col1
        try:
            link_id = row["link_id"] if header else row.iloc[0]
            text = row["text"] if header else row.iloc[1]
            return f"{str(link_id)}:{sha1_hex(str(text))}"
        except Exception:
            return None

    state.ensure_loaded_from_csv(
        table="reddit_comment_keys",
        csv_path=out_path,
        key_builder=key_from_output_row,
        expected_first_col="link_id",
    )

    if not os.path.exists(in_path):
        logging.warning(f"[RCOMMENTS] Missing input file: {in_path} (skipping)")
        return

    out_has_header = detect_has_header(out_path, "link_id") if os.path.exists(out_path) else True
    write_header = (not os.path.exists(out_path)) and out_has_header

    processed_total = 0
    skipped_total = 0

    reader = pd.read_csv(in_path, chunksize=chunksize, usecols=["link_id", "text"])
    for i, chunk in enumerate(reader, start=1):
        chunk = chunk.copy()

        keys = [f"{str(r.link_id)}:{sha1_hex(str(r.text))}" for r in chunk.itertuples(index=False)]
        already = state.keys_exist("reddit_comment_keys", keys)
        mask = [k not in already for k in keys]
        todo = chunk.loc[mask].reset_index(drop=True)

        skipped = len(chunk) - len(todo)
        skipped_total += skipped
        if len(todo) == 0:
            logging.info(f"[RCOMMENTS] Chunk {i}: skipped {skipped} (all already processed)")
            continue

        scores_rows = []
        iterable = todo["text"]
        if tqdm is not None:
            iterable = tqdm(iterable, total=len(todo), desc=f"[RCOMMENTS] chunk {i}")

        for text in iterable:
            text = "" if not isinstance(text, str) else text.strip()
            if not text:
                scores_rows.append({a.lower(): None for a in ATTRIBUTES})
            else:
                scores = client.analyze(text)
                scores_rows.append(scores if scores is not None else {a.lower(): None for a in ATTRIBUTES})

        scores_df = pd.DataFrame(scores_rows)
        out_df = pd.concat([todo[["link_id", "text"]], scores_df], axis=1)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        out_df.to_csv(out_path, mode="a", header=write_header, index=False)

        done_keys = [f"{str(r.link_id)}:{sha1_hex(str(r.text))}" for r in todo.itertuples(index=False)]
        state.add_keys("reddit_comment_keys", done_keys)

        processed_total += len(todo)
        logging.info(f"[RCOMMENTS] Chunk {i}: wrote {len(todo)} rows, skipped {skipped}")

    logging.info(f"[RCOMMENTS] Done. Total wrote={processed_total:,}, skipped={skipped_total:,}")


def _pick_reddit_posts_cols(cols: List[str]) -> Tuple[str, str, str]:
    cols_set = set(cols)

    # ID
    if "name" in cols_set:
        id_col = "name"
    elif "id" in cols_set:
        id_col = "id"
    elif "post_id" in cols_set:
        id_col = "post_id"
    else:
        id_col = ""  # fallback to hash key only

    # title
    if "title" in cols_set:
        title_col = "title"
    else:
        title_col = ""

    # text/body
    for c in ["text", "selftext", "body", "text_body"]:
        if c in cols_set:
            text_col = c
            break
    else:
        text_col = ""

    return id_col, title_col, text_col


def process_reddit_posts(
    in_path: str,
    out_path: str,
    state: StateDB,
    client: PerspectiveClient,
    chunksize: int = 100,
):
    logging.info(f"[RPOSTS] Input: {in_path}")
    logging.info(f"[RPOSTS] Output: {out_path}")

    def key_from_output_row(row, header: bool) -> Optional[str]:
        # If headerless: post_id at col0, title at col1, text at col2
        try:
            post_id = row["post_id"] if header else row.iloc[0]
            title = row["title"] if header else row.iloc[1]
            text = row["text"] if header else row.iloc[2]
            combo = f"{title}\n\n{text}".strip()
            # Prefer stable id, but include hash too so it stays robust
            return f"{str(post_id)}:{sha1_hex(combo)}"
        except Exception:
            return None

    state.ensure_loaded_from_csv(
        table="reddit_post_keys",
        csv_path=out_path,
        key_builder=key_from_output_row,
        expected_first_col="post_id",
    )

    if not os.path.exists(in_path):
        logging.warning(f"[RPOSTS] Missing input file: {in_path} (skipping)")
        return

    # Determine columns dynamically
    peek = pd.read_csv(in_path, nrows=1)
    id_col, title_col, text_col = _pick_reddit_posts_cols(list(peek.columns))

    if not title_col and not text_col:
        logging.error(f"[RPOSTS] Could not find title/text columns in {in_path}. Columns={list(peek.columns)}")
        return

    usecols = [c for c in [id_col, title_col, text_col] if c]
    logging.info(f"[RPOSTS] Using columns: {usecols}")

    out_has_header = detect_has_header(out_path, "post_id") if os.path.exists(out_path) else True
    write_header = (not os.path.exists(out_path)) and out_has_header

    processed_total = 0
    skipped_total = 0

    reader = pd.read_csv(in_path, chunksize=chunksize, usecols=usecols)
    for i, chunk in enumerate(reader, start=1):
        chunk = chunk.copy()

        # Build normalized output columns
        if id_col:
            chunk["post_id"] = chunk[id_col].astype(str)
        else:
            # No ID column -> create a deterministic ID from content
            # (Not perfect, but stable for resume)
            tmp_title = chunk[title_col].astype(str) if title_col else ""
            tmp_text = chunk[text_col].astype(str) if text_col else ""
            chunk["post_id"] = (tmp_title + "\n\n" + tmp_text).map(lambda s: sha1_hex(str(s))[:16])

        chunk["title"] = chunk[title_col].astype(str) if title_col else ""
        chunk["text"] = chunk[text_col].astype(str) if text_col else ""

        # key includes id + hash(combo)
        combos = (chunk["title"].fillna("").astype(str) + "\n\n" + chunk["text"].fillna("").astype(str)).map(lambda s: s.strip())
        keys = [f"{pid}:{sha1_hex(combo)}" for pid, combo in zip(chunk["post_id"].astype(str), combos)]

        already = state.keys_exist("reddit_post_keys", keys)
        mask = [k not in already for k in keys]
        todo = chunk.loc[mask, ["post_id", "title", "text"]].reset_index(drop=True)
        todo_combos = combos.loc[mask].reset_index(drop=True)

        skipped = len(chunk) - len(todo)
        skipped_total += skipped
        if len(todo) == 0:
            logging.info(f"[RPOSTS] Chunk {i}: skipped {skipped} (all already processed)")
            continue

        scores_rows = []
        iterable = list(todo_combos)
        if tqdm is not None:
            iterable = tqdm(iterable, total=len(todo), desc=f"[RPOSTS] chunk {i}")

        for combo_text in iterable:
            combo_text = "" if not isinstance(combo_text, str) else combo_text.strip()
            if not combo_text:
                scores_rows.append({a.lower(): None for a in ATTRIBUTES})
            else:
                scores = client.analyze(combo_text)
                scores_rows.append(scores if scores is not None else {a.lower(): None for a in ATTRIBUTES})

        scores_df = pd.DataFrame(scores_rows)
        out_df = pd.concat([todo, scores_df], axis=1)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        out_df.to_csv(out_path, mode="a", header=write_header, index=False)

        done_keys = [f"{pid}:{sha1_hex(combo)}" for pid, combo in zip(todo["post_id"].astype(str), todo_combos)]
        state.add_keys("reddit_post_keys", done_keys)

        processed_total += len(todo)
        logging.info(f"[RPOSTS] Chunk {i}: wrote {len(todo)} rows, skipped {skipped}")

    logging.info(f"[RPOSTS] Done. Total wrote={processed_total:,}, skipped={skipped_total:,}")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--base", default=".", help="Base directory for inputs/outputs/db")
    p.add_argument("--api-key", default=os.getenv("PERSPECTIVE_API_KEY", ""), help="Perspective API key (or set env PERSPECTIVE_API_KEY)")

    p.add_argument("--chan-in", default="chan_posts.csv")
    p.add_argument("--reddit-comments-in", default="reddit_comments.csv")
    p.add_argument("--reddit-posts-in", default="reddit_posts.csv")

    p.add_argument("--chan-out", default="chan_posts_perspective.csv")
    p.add_argument("--reddit-comments-out", default="reddit_comments_perspective.csv")
    p.add_argument("--reddit-posts-out", default="reddit_posts_perspective.csv")

    p.add_argument("--db", default="perspective_state.sqlite")
    p.add_argument("--chunksize", type=int, default=100)
    p.add_argument("--sleep", type=float, default=1.05)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--truncate", type=int, default=3000)

    p.add_argument("--run-chan", action="store_true")
    p.add_argument("--run-reddit-comments", action="store_true")
    p.add_argument("--run-reddit-posts", action="store_true")
    p.add_argument("--log-level", default="INFO")

    args = p.parse_args()
    setup_logging(args.log_level)

    if not args.api_key:
        logging.error("Missing API key. Pass --api-key or set env var PERSPECTIVE_API_KEY.")
        sys.exit(2)

    base = args.base
    chan_in = os.path.join(base, args.chan_in)
    rcom_in = os.path.join(base, args.reddit_comments_in)
    rpos_in = os.path.join(base, args.reddit_posts_in)

    chan_out = os.path.join(base, args.chan_out)
    rcom_out = os.path.join(base, args.reddit_comments_out)
    rpos_out = os.path.join(base, args.reddit_posts_out)

    db_path = os.path.join(base, args.db)

    client = PerspectiveClient(
        api_key=args.api_key,
        sleep_s=args.sleep,
        max_retries=args.retries,
        timeout_s=args.timeout,
        truncate_chars=args.truncate,
    )

    state = StateDB(db_path)

    # If user didn't specify any run flags, run all three
    run_any = args.run_chan or args.run_reddit_comments or args.run_reddit_posts
    run_chan = args.run_chan or (not run_any)
    run_rcom = args.run_reddit_comments or (not run_any)
    run_rpos = args.run_reddit_posts or (not run_any)

    try:
        if run_chan:
            process_chan_posts(chan_in, chan_out, state, client, chunksize=args.chunksize)
        if run_rcom:
            process_reddit_comments(rcom_in, rcom_out, state, client, chunksize=args.chunksize)
        if run_rpos:
            process_reddit_posts(rpos_in, rpos_out, state, client, chunksize=args.chunksize)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user (KeyboardInterrupt). Progress is saved; re-run to resume.")
    finally:
        state.close()


if __name__ == "__main__":
    main()
