"""C3 adapter compatibility layer for frozen selector-line metadata.

The C3 decision uses only line_id and text. Existing selector metadata is
accepted at the boundary and stripped before the frozen C3 v1 decision.
"""
from __future__ import annotations

from typing import Any

import v12_c3_structured_adapter as v1


DuplicateKeyError = v1.DuplicateKeyError
strict_json_loads = v1.strict_json_loads
parse_batch_response = v1.parse_batch_response
occurrences = v1.occurrences

REQUIRED_LINE_FIELDS = {"line_id", "text"}
OPTIONAL_SELECTOR_FIELDS = {"lexical_score", "subtitle_score", "selection"}
ALLOWED_LINE_FIELDS = REQUIRED_LINE_FIELDS | OPTIONAL_SELECTOR_FIELDS


def decide(qwen: Any, selected_lines: Any, assessment: Any, *, c1r, core, a1, p3, p4):
    if not isinstance(selected_lines, list):
        return v1.decide(qwen, selected_lines, assessment, c1r=c1r, core=core, a1=a1, p3=p3, p4=p4)
    cleaned = []
    for line in selected_lines:
        if (not isinstance(line, dict) or not REQUIRED_LINE_FIELDS <= set(line)
                or not set(line) <= ALLOWED_LINE_FIELDS):
            return v1.decide(qwen, selected_lines, assessment, c1r=c1r, core=core, a1=a1, p3=p3, p4=p4)
        cleaned.append({"line_id": line["line_id"], "text": line["text"]})
    return v1.decide(qwen, cleaned, assessment, c1r=c1r, core=core, a1=a1, p3=p3, p4=p4)

