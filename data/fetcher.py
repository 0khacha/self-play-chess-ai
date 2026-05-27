"""
Fetch games from Chess.com archives.

Downloads games from the Chess.com public API, caches raw JSON responses
to disk, and returns a flat list of game dicts containing PGN strings.
"""
import json
import os
import time
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

import config
from utils.helpers import setup_logging

logger = setup_logging("fetcher", config.LOGS_DIR)

# ---------------------------------------------
# Retry / network constants
# ---------------------------------------------
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # exponential back-off base (seconds)
_REQUEST_TIMEOUT = 30  # seconds


def _cache_path_for_url(url: str) -> str:
    """Derive a deterministic cache file path from an archive URL.

    Example URL: https://api.chess.com/pub/player/0khacha/games/2025/04
    Cache file:  <RAW_DATA_DIR>/0khacha_2025_04.json
    """
    parts = urlparse(url).path.rstrip("/").split("/")
    # Typical path: /pub/player/<user>/games/<year>/<month>
    if len(parts) >= 6:
        user = parts[3]
        year = parts[5]
        month = parts[6] if len(parts) > 6 else "00"
    else:
        # Fallback: hash the whole URL
        user = "unknown"
        year = "0000"
        month = "00"
    filename = f"{user}_{year}_{month}.json"
    return os.path.join(config.RAW_DATA_DIR, filename)


def _fetch_url_with_retries(url: str, session: requests.Session) -> dict:
    """GET *url* with retries and exponential back-off.

    Returns the parsed JSON dict on success.
    Raises ``requests.RequestException`` after exhausting retries.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF ** attempt
                logger.warning(
                    "Request to %s failed (attempt %d/%d): %s  retrying in %.1fs",
                    url, attempt, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Request to %s failed after %d attempts: %s",
                    url, _MAX_RETRIES, exc,
                )
    raise last_exc  # type: ignore[misc]


def _load_cache(cache_file: str) -> Optional[dict]:
    """Load a cached JSON file.  Returns ``None`` if cache is missing or empty."""
    if not os.path.isfile(cache_file):
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Treat empty objects / missing games list as invalid cache
        if not data or not data.get("games"):
            return None
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Corrupt cache file %s  will re-fetch: %s", cache_file, exc)
        return None


def _save_cache(cache_file: str, data: dict) -> None:
    """Persist a JSON response to disk."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    logger.debug("Cached response to %s", cache_file)


def _normalize_game(game_dict: dict) -> Optional[dict]:
    """Ensure a game dict has the minimum required fields.

    Returns a cleaned dict with at least 'pgn', 'white', 'black' keys,
    or ``None`` if the record is unusable.
    """
    pgn = game_dict.get("pgn")
    if not pgn:
        return None

    white_info = game_dict.get("white", {})
    black_info = game_dict.get("black", {})

    # Chess.com nests username under 'username' inside the colour dict
    white_name = (
        white_info.get("username", "")
        if isinstance(white_info, dict)
        else str(white_info)
    )
    black_name = (
        black_info.get("username", "")
        if isinstance(black_info, dict)
        else str(black_info)
    )

    return {
        "pgn": pgn,
        "white": white_name,
        "black": black_name,
        # Preserve any extra metadata the caller might want later
        "url": game_dict.get("url", ""),
        "time_control": game_dict.get("time_control", ""),
        "rated": game_dict.get("rated", False),
    }


# ---------------------------------------------
# Public API
# ---------------------------------------------
def fetch_all_games(
    archives: Optional[list[str]] = None,
    username: Optional[str] = None,
) -> list[dict]:
    """Fetch all games across the given Chess.com archive URLs.

    Parameters
    ----------
    archives : list[str], optional
        List of Chess.com archive API URLs.  Defaults to
        ``config.CHESS_COM_ARCHIVES``.
    username : str, optional
        Chess.com username (used only for logging context).
        Defaults to ``config.CHESS_COM_USERNAME``.

    Returns
    -------
    list[dict]
        Each dict contains at least ``'pgn'``, ``'white'``, and ``'black'``
        keys.  Additional metadata (url, time_control, rated) is preserved
        when available.
    """
    if archives is None:
        archives = config.CHESS_COM_ARCHIVES
    if username is None:
        username = config.CHESS_COM_USERNAME

    logger.info(
        "Fetching games for '%s' from %d archive(s)", username, len(archives)
    )

    session = requests.Session()
    session.headers.update({"User-Agent": config.API_USER_AGENT})

    all_games: list[dict] = []
    fetched_count = 0
    cached_count = 0

    for url in tqdm(archives, desc="Archives", unit="archive"):
        cache_file = _cache_path_for_url(url)
        cached = _load_cache(cache_file)

        if cached is not None:
            raw_games = cached.get("games", [])
            cached_count += 1
            logger.debug("Loaded %d games from cache: %s", len(raw_games), cache_file)
        else:
            try:
                data = _fetch_url_with_retries(url, session)
            except requests.RequestException:
                logger.error("Skipping archive %s due to fetch failure", url)
                continue

            raw_games = data.get("games", [])
            fetched_count += 1

            # Cache only if there are games (don't overwrite with empty)
            if raw_games:
                _save_cache(cache_file, data)

            # Rate-limit between network requests
            time.sleep(config.API_RATE_LIMIT_SECONDS)

        for g in raw_games:
            normalized = _normalize_game(g)
            if normalized is not None:
                all_games.append(normalized)

    logger.info(
        "Done  %d total games collected (%d archives fetched, %d from cache)",
        len(all_games), fetched_count, cached_count,
    )
    return all_games
