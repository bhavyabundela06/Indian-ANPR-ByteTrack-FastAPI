"""
plate_rules.py  —  Indian license-plate domain rules.

Uses the KNOWN STRUCTURE of Indian plates to validate and CORRECT OCR reads:
  format:  SS DD  L(1-3)  NNNN
           SS   = 2-letter state/UT code (must be a real one)
           DD   = 1-2 digit RTO district code
           L    = 1-3 letter series
           NNNN = 4-digit unique number  (ALWAYS digits — never letters)

Two things rules CAN do:
  1. Reject impossible reads (fake state code, letters where digits must be).
  2. Auto-correct common OCR confusions IN THE RIGHT POSITION
     (O<->0, I<->1, S<->5, etc.) then re-check.
What rules CANNOT do: decide between two reads that are BOTH valid
(e.g. MH03DY5705 vs MH03DZ5705) — that still needs voting.
"""

import re

# Every valid Indian state / UT registration code.
STATE_CODES = {
    "AP","AR","AS","BR","CG","GA","GJ","HR","HP","JH","JK","KA","KL","MP",
    "MH","MN","ML","MZ","NL","OD","OR","PB","RJ","SK","TN","TS","TR","UP",
    "UK","UA","WB","AN","CH","DN","DD","DL","LD","PY","LA",
    # BH = new "Bharat" series (nationwide)
    "BH",
}

# Full standard plate: SS DD L(1-3) NNNN
PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

# OCR confusions. Two maps because the fix depends on POSITION:
# in a LETTER slot a '0' should become 'O'; in a DIGIT slot an 'O' -> '0'.
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
            "S": "5", "B": "8", "G": "6", "T": "7"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}
# common whole-state-code fixes (first char misread)
STATE_FIX = {"WH": "MH", "NH": "MH", "0L": "DL", "0D": "OD", "HH": "MH"}


def _clean(text):
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _split(plate):
    """Break a candidate into (state, district, series, number) by the regex."""
    m = re.match(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{4})$", plate)
    return m.groups() if m else None


def is_valid_plate(text):
    """Structurally valid AND real state code."""
    p = _clean(text)
    if not PLATE_RE.match(p):
        return False
    return p[:2] in STATE_CODES


def correct_plate(text):
    """Try to fix a near-miss read using position-aware character rules.
    Returns a corrected valid plate, or None if it can't be salvaged."""
    p = _clean(text)

    # already good?
    if is_valid_plate(p):
        return p

    # obvious whole-state-code fix (e.g. WH -> MH)
    if p[:2] in STATE_FIX:
        p2 = STATE_FIX[p[:2]] + p[2:]
        if is_valid_plate(p2):
            return p2

    # Need the right overall shape to correct by position. Total length 9-10.
    if not (9 <= len(p) <= 10):
        return None

    # Layout for a 10-char plate: [0:2]=state letters, [2:4]=district digits,
    # [4:len-4]=series letters, [len-4:]=number digits.
    n = len(p)
    state = list(p[0:2])
    district = list(p[2:4]) if n == 10 else list(p[2:3])
    num = list(p[n-4:n])
    series = list(p[4 if n == 10 else 3 : n-4])

    # state -> must be LETTERS
    state = [TO_LETTER.get(c, c) if c.isdigit() else c for c in state]
    # district -> must be DIGITS
    district = [TO_DIGIT.get(c, c) if c.isalpha() else c for c in district]
    # series -> must be LETTERS
    series = [TO_LETTER.get(c, c) if c.isdigit() else c for c in series]
    # number -> must be DIGITS  (this is your rule: last 4 are ALWAYS digits)
    num = [TO_DIGIT.get(c, c) if c.isalpha() else c for c in num]

    fixed = "".join(state + district + series + num)
    return fixed if is_valid_plate(fixed) else None


def validate_or_correct(text):
    """Main entry: return a trustworthy plate string, or None to reject.
    Accepts a valid read as-is, tries to correct a near-miss, else rejects."""
    p = _clean(text)
    if is_valid_plate(p):
        return p
    return correct_plate(p)


if __name__ == "__main__":
    tests = ["MH03DY5705","MH03DYS705","MH03DYS7O5","WH03DY5705",
             "MH03D5705","MH02EK4082","MARUTI","KL07AB12O5","0L8CAF1234"]
    for t in tests:
        print(f"{t:14} -> {validate_or_correct(t)}")