"""Rank observed OCR strings without substituting individual characters."""

from collections import defaultdict
import re

from .plate_validator import validate_registration


def normalize_text(text):
    return re.sub(r"[\s.\-]+", "", text).upper()


def candidate_text(text):
    text = normalize_text(text)
    if re.fullmatch(r"[A-Z0-9]{5,12}", text) and re.search("[A-Z]", text) and re.search("[0-9]", text):
        return text
    return ""


def engine_report(result):
    rows = [
        {"variant": variant, "raw_text": read.raw_text, "confidence": read.confidence,
         "normalized_text": normalize_text(read.raw_text),
         "format_valid": bool(validate_registration(read.raw_text))}
        for variant, reads in result.readings.items() for read in reads
    ]
    best = max((row for row in rows if row["raw_text"].strip()), key=lambda row: row["confidence"], default=None)
    return {"elapsed_ms": result.elapsed_ms, "error": result.error, "best": best, "readings": rows}


def rank_candidates(observations):
    """Each frame contributes at most one weight to a given character or exact string."""
    by_text = defaultdict(dict)
    for text, plate, read in observations:
        current = by_text[text].get(plate.frame)
        if current is None or read.confidence > current[1].confidence:
            by_text[text][plate.frame] = (plate, read)
    if not by_text:
        return None

    def weight(plate, read):
        return max(1, plate.rank) * read.confidence

    ranked = []
    for text, matches in by_text.items():
        comparable = {}
        for spelling, frames in by_text.items():
            if len(spelling) != len(text):
                continue
            for frame, (plate, read) in frames.items():
                score = weight(plate, read)
                if frame not in comparable or score > comparable[frame][1]:
                    comparable[frame] = (spelling, score)
        total = sum(score for _, score in comparable.values())
        agreement = min(sum(score for spelling, score in comparable.values()
                            if spelling[i] == character) / total
                        for i, character in enumerate(text))
        support = sum(weight(plate, read) for plate, read in matches.values())
        consistent = len(matches) >= 2 and agreement > 0.5
        ranked.append((consistent, support, agreement, text, matches))
    return max(ranked, key=lambda item: item[:3])
