import re


STATE_CODES = frozenset(
    "AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP "
    "MZ NL OD OR PB PY RJ SK TN TR TS TG UK UP UA WB".split()
)
REGISTRATION = re.compile(r"([A-Z]{2})[0-9]{1,2}[A-Z]{1,3}[0-9]{4}")
BH_REGISTRATION = re.compile(r"[0-9]{2}BH[0-9]{4}[A-HJ-NP-Z]{1,2}")
VINTAGE_REGISTRATION = re.compile(r"([A-Z]{2})VA[A-Z]{2}(?!0000)[0-9]{4}")


def validate_registration(raw_text: str) -> str:
    """Return a normalized standard registration, or an empty string."""
    text = re.sub(r"[\s.\-]+", "", raw_text).upper()
    if BH_REGISTRATION.fullmatch(text):
        return text
    match = REGISTRATION.fullmatch(text) or VINTAGE_REGISTRATION.fullmatch(text)
    return text if match and match[1] in STATE_CODES else ""
