"""
src/extraction.py - Step and answer extraction utilities.

Reference key used in comments:
  [L<n>]   = lecture "008-Inference-time Improvement and Reasoning", slide n
  [PA3 §x] = assignment handout PA3.pdf, section x
"""
import re  # Python's built-in regular-expression library (the only tool the handout allows here) [PA3 §1.3]


def parse_steps(completion_text: str) -> list[str]:
    """
    Parses completion text and returns a list of individual step strings.
    Must tolerate markdown decoration around labels (e.g. '**Step 1:**', '### Step 2:')
    and strip leftover '*' or '#' from captured text.

    (Provided by the starter code, unchanged.) Splitting the chain of thought into
    separate steps is what lets the PRM judge check each step on its own [PA3 §2.2].
    """
    if not completion_text or not isinstance(completion_text, str):  # empty / non-string input -> no steps
        return []  # never crash on garbage [PA3 §4.3]

    # Pattern matching Step markers with optional markdown decoration (#, *, `, :, .)
    pattern = r"(?:^|\n)\s*(?:[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)(.*?)(?=(?:\n\s*[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)|(?:\n\s*[#*`\s]*Final Answer)|$)"
    matches = re.findall(pattern, completion_text, flags=re.DOTALL | re.IGNORECASE)  # text between consecutive step labels
    cleaned = []  # final list of clean step strings
    for m in matches:  # look at every captured step body
        text = m.strip().strip("*#`").strip()  # remove whitespace and leftover markdown symbols
        if text:  # skip empty captures
            cleaned.append(text)  # keep the real step text
    return cleaned  # e.g. ["Janet starts with 16 eggs.", "She uses 3 + 4 = 7 eggs.", ...]


# Finds every "Final Answer" label, tolerating bold/backticks around it, case and a missing colon.
# Group 1 captures the rest of THAT line only ('.' does not cross newlines without DOTALL).
_FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*[*_`]*\s*:?\s*[*_`]*\s*(.*)", re.IGNORECASE)  # [PA3 §4.3 table]

# A number at the very start of the captured text: optional minus, digits, optional decimal part.
_LEADING_NUMBER_RE = re.compile(r"(-?\d+)(?:\.(\d+))?")  # group 1 = integer part, group 2 = decimals


def extract_answer(completion_text: str) -> int | None:
    """
    Extracts the final numeric integer from the completion.

    Required Behaviors:
      - 'Final Answer: 18' -> 18
      - 'Final Answer: $1,234' -> 1234
      - 'Final Answer: 18.00' -> 18
      - 'Final Answer: 18.5' -> None (non-integer)
      - 'Final Answer: -5' -> -5
      - 'Final Answer: 18 dollars.' -> 18
      - '**Final Answer:** 18' -> 18
      - 'final answer 18' (case-insensitive, colon optional) -> 18
      - Multiple 'Final Answer' lines -> use the LAST one
      - No 'Final Answer' present -> None
      - Empty string or None-like garbage -> None, and never raise

    Why this matters: majority voting and accuracy both need ONE clean number per sample.
    We only ask the model for the format through the prompt (not constrained decoding [L26]),
    so format failures are possible and must surface as None [PA3 §4.2 "Format Compliance Warning"].
    """
    try:  # the parser must NEVER raise, whatever it is given [PA3 §4.3]
        if not isinstance(completion_text, str) or not completion_text:  # None, numbers, "" ...
            return None  # nothing to parse

        matches = list(_FINAL_ANSWER_RE.finditer(completion_text))  # every "Final Answer" occurrence, in order
        if not matches:  # no delimiter at all
            return None  # no fallback guessing of "some integer in the text" [PA3 §4.3]

        rest = matches[-1].group(1)  # use the LAST Final Answer line (earlier ones are scratch work)
        rest = rest.strip().lstrip("*_` ")  # drop spaces and leftover markdown such as '** 18'
        rest = rest.replace("$", "").replace(",", "")  # '$1,234' -> '1234' (currency sign and thousands commas)
        rest = rest.lstrip()  # '$ 5' -> '5' after the '$' removal

        num = _LEADING_NUMBER_RE.match(rest)  # the answer must START with a number ('18 dollars.' is fine)
        if num is None:  # e.g. 'None', '[object Object]', '\boxed{18}'
            return None  # not a parseable integer

        int_part, decimals = num.group(1), num.group(2)  # split '18.00' into '18' and '00'
        if decimals is not None and decimals.strip("0"):  # any non-zero digit after the point, e.g. '18.5'
            return None  # GSM8K answers are integers, so a real fraction is a failed extraction
        return int(int_part)  # '18', '18.00', '-5' -> 18, 18, -5
    except Exception:  # absolutely any unexpected problem
        return None  # still return None instead of crashing the whole benchmark run
