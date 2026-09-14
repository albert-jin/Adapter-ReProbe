from types import SimpleNamespace

import torch
from torch import nn

from luh.feature_extractors.basic_hidden_states import FeatureExtractorBasicHiddenStates
from luh.heads.uncertainty_head_claim import UncertaintyHeadClaim
from luh.heads.uncertainty_head_claim import LowRankResidualBridge


class DummyModel:
    config = SimpleNamespace(hidden_size=1, num_hidden_layers=1)


class DummyExtractor:
    def feature_dim(self):
        return 4

    def output_attention(self):
        return False


class CaptureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.padding_mask = None

    def forward(self, src, src_key_padding_mask=None, **kwargs):
        self.padding_mask = src_key_padding_mask.detach().clone()
        return src


def test_hidden_state_current_and_previous_alignment():
    hidden = torch.arange(4.0).reshape(1, 4, 1)
    outputs = {"hidden_states": (hidden, hidden + 10)}
    previous = FeatureExtractorBasicHiddenStates(
        DummyModel(), layer_nums=[-1], token_alignment="previous"
    )
    current = FeatureExtractorBasicHiddenStates(
        DummyModel(), layer_nums=[-1], token_alignment="current"
    )
    assert previous({}, outputs).flatten().tolist() == [10.0, 11.0, 12.0]
    assert current({}, outputs).flatten().tolist() == [11.0, 12.0, 13.0]


def test_future_tokens_are_added_to_padding_mask():
    head = UncertaintyHeadClaim(
        DummyExtractor(),
        head_dim=4,
        n_layers=1,
        n_heads=1,
        dropout=0.0,
        mask_future_tokens=True,
    )
    capture = CaptureEncoder()
    head.transformer_encoder = capture
    features = torch.zeros(1, 5, 4)
    attention_mask = torch.ones(1, 5, dtype=torch.long)
    claims = [torch.tensor([[0, 1, 1, 0, 0]], dtype=torch.long)]
    head._compute_tensors({"claims": claims}, features, attention_mask)
    assert capture.padding_mask.tolist() == [[False, False, False, True, True]]


def test_low_rank_bridge_starts_as_identity_including_bias():
    bridge = LowRankResidualBridge(feature_dim=7, rank=3, alpha=3)
    features = torch.randn(2, 5, 7)
    assert torch.equal(bridge(features), features)
