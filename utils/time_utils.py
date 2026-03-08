"""Date and time helper functions for collision records."""

from datetime import date, datetime, time
from typing import Optional


_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S.%f")
_TIME_FORMATS = ("%H:%M", "%H:%M:%S")


def parse_crash_date(value: str) -> Optional[date]:
    """Parse crash date text into a date object.

    Args:
        value: Date text.

    Returns:
        Optional[date]: Parsed date or None if parsing fails.

    Raises:
        None
    """
    text = (value or "").strip()
    if not text:
        return None

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    # Try slicing ISO-like string as fallback.
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None


def parse_crash_time(value: str) -> Optional[time]:
    """Parse crash time text into a time object.

    Args:
        value: Time text.

    Returns:
        Optional[time]: Parsed time or None if parsing fails.

    Raises:
        None
    """
    text = (value or "").strip()
    if not text:
        return None

    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return None


def combine_date_time(date_text: str, time_text: str) -> Optional[datetime]:
    """Combine crash date and crash time text into one datetime.

    Args:
        date_text: Date text.
        time_text: Time text.

    Returns:
        Optional[datetime]: Combined datetime, or None when parsing fails.

    Raises:
        None
    """
    parsed_date = parse_crash_date(date_text)
    if not parsed_date:
        return None

    parsed_time = parse_crash_time(time_text) or time(hour=0, minute=0)
    return datetime.combine(parsed_date, parsed_time)


def is_weekend(dt: datetime) -> int:
    """Return weekend flag from datetime.

    Args:
        dt: Input datetime.

    Returns:
        int: 1 for Saturday/Sunday, else 0.

    Raises:
        ValueError: If dt is None.
    """
    if dt is None:
        raise ValueError("Datetime value cannot be None.")
    return 1 if dt.weekday() >= 5 else 0


def pandemic_phase(dt: datetime) -> str:
    """Return 2020 NYC phase label based on crash datetime.

    Args:
        dt: Input datetime.

    Returns:
        str: Pandemic phase label.

    Raises:
        ValueError: If dt is None.
    """
    if dt is None:
        raise ValueError("Datetime value cannot be None.")

    day = dt.date()

    if day < date(2020, 3, 22):
        return "PRE_PAUSE"
    if day <= date(2020, 6, 7):
        return "PAUSE"
    if day <= date(2020, 6, 21):
        return "REOPEN_PHASE_1"
    if day <= date(2020, 7, 5):
        return "REOPEN_PHASE_2"
    if day <= date(2020, 7, 19):
        return "REOPEN_PHASE_3"
    return "REOPEN_PHASE_4_PLUS"
