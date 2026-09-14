"""Utilities for aligning saved generations with their tokenized prompts.

The released ReProbe trajectory datasets already contain ``input_ids`` and the
generated ``reply``.  Re-tokenizing a prompt template is unsafe because the
template used to create the dataset may differ from the current repository
copy.  This module instead locates the saved reply inside the saved token IDs.
"""

from __future__ import annotations

from collections.abc import Sequence


def _subsequence_starts(sequence: Sequence[int], pattern: Sequence[int]) -> list[int]:
    if not pattern or len(pattern) > len(sequence):
        return []
    width = len(pattern)
    return [
        start
        for start in range(len(sequence) - width + 1)
        if list(sequence[start : start + width]) == list(pattern)
    ]


def locate_reply_start(
    input_ids: Sequence[int],
    reply: str,
    tokenizer,
    *,
    fallback_prefix_tokens: int = 12,
) -> int:
    """Return the token index where ``reply`` starts in ``input_ids``.

    Exact full-reply matching is attempted first.  One released GSM8K sample
    was tokenized with a slightly different tokenizer build and differs at one
    token, so a sufficiently long reply prefix is used as a documented
    fallback.  The final occurrence is selected to avoid matching the example
    response format embedded in the user prompt.
    """

    saved_ids = [int(token_id) for token_id in input_ids]
    reply_ids = tokenizer(reply, add_special_tokens=False)["input_ids"]
    exact_starts = _subsequence_starts(saved_ids, reply_ids)
    if exact_starts:
        return exact_starts[-1]

    prefix_width = min(int(fallback_prefix_tokens), len(reply_ids))
    if prefix_width <= 0:
        raise ValueError("Cannot align an empty reply")
    prefix_starts = _subsequence_starts(saved_ids, reply_ids[:prefix_width])
    if not prefix_starts:
        raise ValueError(
            "Could not locate the saved reply in input_ids, including the "
            f"{prefix_width}-token fallback prefix"
        )
    return prefix_starts[-1]


def claim_token_positions(
    input_ids: Sequence[int],
    reply_start: int,
    aligned_token_ids: Sequence[int],
    special_token_ids: Sequence[int],
) -> list[int]:
    """Map reply-relative non-special-token positions to full-sequence indices."""

    special_ids = set(int(token_id) for token_id in special_token_ids)
    reply_mapping = [
        reply_start + offset
        for offset, token_id in enumerate(input_ids[reply_start:])
        if int(token_id) not in special_ids
    ]
    positions = []
    for aligned_id in aligned_token_ids:
        aligned_id = int(aligned_id)
        if aligned_id < 0 or aligned_id >= len(reply_mapping):
            raise IndexError(
                f"Claim token index {aligned_id} is outside the reply mapping "
                f"of length {len(reply_mapping)}"
            )
        positions.append(reply_mapping[aligned_id])
    return positions
