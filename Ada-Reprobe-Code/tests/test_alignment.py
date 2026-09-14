from luh.alignment import claim_token_positions, locate_reply_start


class TinyTokenizer:
    all_special_ids = [99]

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        mapping = {
            "reply": [1, 2, 3, 4],
            "drifted": [1, 2, 8, 4],
        }
        return {"input_ids": mapping[text]}


def test_locate_exact_reply_uses_final_occurrence():
    tokenizer = TinyTokenizer()
    assert locate_reply_start([1, 2, 3, 4, 7, 1, 2, 3, 4, 99], "reply", tokenizer) == 5


def test_locate_reply_prefix_fallback():
    tokenizer = TinyTokenizer()
    assert (
        locate_reply_start(
            [7, 1, 2, 3, 4, 99],
            "drifted",
            tokenizer,
            fallback_prefix_tokens=2,
        )
        == 1
    )


def test_claim_token_positions_skip_special_tokens():
    assert claim_token_positions([7, 1, 99, 2, 3], 1, [0, 1, 2], [99]) == [1, 3, 4]
