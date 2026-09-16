"""Versioned lexical-only cleanup; never applied to cached encoder inputs."""

import html
import re

LEXICAL_CLEANING = "html-entities-v2"
_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_ORPHAN_NUMERIC_ENTITY = re.compile(r"(?<![&A-Za-z0-9;])#(?:[xX][0-9A-Fa-f]+|[0-9]+);")
_ORPHAN_NAMED_ENTITY = re.compile(r"(?<![&A-Za-z0-9;])(quot|apos|amp|lt|gt);")


def clean_lexical_text(text):
    # Restore only syntactically recognizable references whose leading ampersand
    # was lost upstream (for example "#39;s" and "quot;word"). Keep ordinary
    # numbers and words; do not blacklist tokens observed in a particular topic.
    text = _ORPHAN_NUMERIC_ENTITY.sub(lambda match: "&" + match.group(0), text)
    text = _ORPHAN_NAMED_ENTITY.sub(lambda match: "&" + match.group(0), text)
    for _ in range(3):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    return _TAG.sub(" ", text)
