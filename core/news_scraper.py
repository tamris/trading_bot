import logging
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple, Optional
import xml.etree.ElementTree as ET
import feedparser
import requests

from config.settings import settings

logger = logging.getLogger("ai_trading_bot.news")


class NewsScraper:
    """Fetches real-time financial headlines and manages economic calendar blackout periods."""

    CALENDAR_XML_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    CALENDAR_JSON_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    RSS_FEEDS = [
        {"name": "ForexLive", "url": "https://www.forexlive.com/feed/news"},
        {"name": "FXStreet", "url": "https://www.fxstreet.com/rss/news"},
    ]

    KEYWORDS = [
        "USD", "EUR", "FED", "POWELL", "ECB", "LAGARDE", "GOLD", "XAU",
        "INFLATION", "CPI", "RATES", "TREASURY", "DOLLAR", "CENTRAL BANK",
        "NFP", "EMPLOYMENT", "YIELDS", "TARIFF", "GEOPOLITICAL"
    ]

    def __init__(self):
        self.blackout_minutes = settings.NEWS_PROTECTION_MINUTES
        self._cached_events: List[Dict[str, Any]] = []
        self._last_calendar_fetch: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=2)
        self._cache_file = settings.LOGS_DIR.parent / "data" / "calendar_cache.json"
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_disk_cache()

    def _load_disk_cache(self):
        """Load calendar from disk if fresh."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._cached_events = data.get("events", [])
                    ts = data.get("timestamp")
                    if ts:
                        self._last_calendar_fetch = datetime.fromisoformat(ts)
                logger.info(f"Loaded {len(self._cached_events)} calendar events from disk cache.")
            except Exception as e:
                logger.warning(f"Failed to read calendar cache: {e}")

    def _save_disk_cache(self):
        """Save calendar events to disk."""
        try:
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "events": self._cached_events
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save calendar cache: {e}")

    def refresh_calendar(self, force: bool = False) -> None:
        """Fetch and cache weekly economic calendar events (XML preferred, JSON fallback)."""
        now = datetime.now(timezone.utc)
        if not force and self._last_calendar_fetch and (now - self._last_calendar_fetch < self._cache_ttl):
            return

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        # Try XML feed first (rarely rate-limited)
        try:
            resp = requests.get(self.CALENDAR_XML_URL, headers=headers, timeout=8)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                events = []
                for elem in root.findall("event"):
                    title = elem.findtext("title", "")
                    country = elem.findtext("country", "").upper()
                    date_str = elem.findtext("date", "")
                    time_str = elem.findtext("time", "")
                    impact = elem.findtext("impact", "")

                    # Convert to ISO datetime if valid
                    iso_dt = self._parse_xml_datetime(date_str, time_str)
                    events.append({
                        "title": title,
                        "country": country,
                        "impact": impact,
                        "date": iso_dt
                    })

                self._cached_events = events
                self._last_calendar_fetch = now
                self._save_disk_cache()
                logger.info(f"Economic calendar (XML) refreshed: {len(events)} events loaded.")
                return
        except Exception as e:
            logger.warning(f"Error fetching calendar XML: {e}")

        # Fallback to JSON feed
        try:
            resp = requests.get(self.CALENDAR_JSON_URL, headers=headers, timeout=8)
            if resp.status_code == 200:
                self._cached_events = resp.json()
                self._last_calendar_fetch = now
                self._save_disk_cache()
                logger.info(f"Economic calendar (JSON) refreshed: {len(self._cached_events)} events loaded.")
        except Exception as e:
            logger.warning(f"Error fetching calendar JSON: {e}")

    @staticmethod
    def _parse_xml_datetime(date_str: str, time_str: str) -> str:
        """Parse 'MM-DD-YYYY' and '10:30pm' (Forex Factory XML format) to UTC ISO string."""
        try:
            if not time_str or "all day" in time_str.lower() or "day" in time_str.lower():
                dt = datetime.strptime(date_str, "%m-%d-%Y").replace(tzinfo=timezone.utc)
                return dt.isoformat()

            # Format: '11-08-2026' and '10:30pm' (Eastern US time: -05:00 standard, -04:00 daylight)
            full_str = f"{date_str} {time_str.strip().lower()}"
            dt_est = datetime.strptime(full_str, "%m-%d-%Y %I:%M%p")
            # Forex Factory XML is in EST/EDT (-05:00 UTC)
            dt_utc = dt_est.replace(tzinfo=timezone(timedelta(hours=-5))).astimezone(timezone.utc)
            return dt_utc.isoformat()
        except Exception:
            return ""

    def is_news_blackout(self, symbol: str, check_time: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Check if current time falls within +/- NEWS_PROTECTION_MINUTES of a High Impact event
        for the relevant currency (USD for XAUUSD, USD & EUR for EURUSD).
        """
        self.refresh_calendar()
        if not self._cached_events:
            return False, "Calendar data unavailable, proceeding with caution"

        now = check_time or datetime.now(timezone.utc)
        relevant_currencies = ["USD"]
        if "EUR" in symbol.upper():
            relevant_currencies.append("EUR")
        if "JPY" in symbol.upper():
            relevant_currencies.append("JPY")

        blackout_delta = timedelta(minutes=self.blackout_minutes)

        for event in self._cached_events:
            impact = event.get("impact", "")
            country = event.get("country", "").upper()

            if impact == "High" and country in relevant_currencies:
                date_str = event.get("date", "")
                if not date_str:
                    continue
                try:
                    event_dt_utc = datetime.fromisoformat(date_str)
                    if event_dt_utc.tzinfo is None:
                        event_dt_utc = event_dt_utc.replace(tzinfo=timezone.utc)
                    else:
                        event_dt_utc = event_dt_utc.astimezone(timezone.utc)

                    time_diff = abs(now - event_dt_utc)
                    if time_diff <= blackout_delta:
                        diff_minutes = time_diff.total_seconds() / 60.0
                        msg = (
                            f"NEWS BLACKOUT ACTIVE: High Impact '{event.get('title')}' ({country}) "
                            f"at {event_dt_utc.strftime('%H:%M UTC')}. Time delta: {diff_minutes:.1f}m "
                            f"(limit: {self.blackout_minutes}m)."
                        )
                        logger.warning(msg)
                        return True, msg
                except Exception:
                    continue

        return False, "No high-impact news blackout active"

    def fetch_latest_news(self, limit: int = 8) -> List[Dict[str, str]]:
        """
        Scrape financial RSS feeds and filter for USD/EUR/Gold/Macro relevant headlines.
        Returns top `limit` headlines.
        """
        headlines = []
        seen_titles = set()

        for feed_info in self.RSS_FEEDS:
            try:
                parsed = feedparser.parse(feed_info["url"])
                for entry in parsed.entries[:25]:
                    title = getattr(entry, "title", "").strip()
                    if not title or title.lower() in seen_titles:
                        continue

                    title_upper = title.upper()
                    summary = getattr(entry, "summary", "")
                    is_relevant = any(kw in title_upper for kw in self.KEYWORDS)

                    if is_relevant:
                        seen_titles.add(title.lower())
                        published = getattr(entry, "published", "")
                        headlines.append({
                            "title": title,
                            "source": feed_info["name"],
                            "published": published,
                            "summary": summary[:160] if summary else ""
                        })

                        if len(headlines) >= limit:
                            break
            except Exception as e:
                logger.warning(f"Error parsing feed {feed_info['name']}: {e}")

            if len(headlines) >= limit:
                break

        # Safe fallback headlines if network is disconnected
        if not headlines:
            headlines = [
                {
                    "title": "US Dollar index steadies ahead of Federal Reserve monetary policy guidance.",
                    "source": "MarketSummary",
                    "published": datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT"),
                    "summary": "Gold prices consolidate while currency markets assess yields and inflation signals."
                }
            ]

        logger.info(f"Retrieved {len(headlines)} relevant financial headlines.")
        return headlines
