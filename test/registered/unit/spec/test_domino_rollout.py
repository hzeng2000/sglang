import unittest

import torch
from torch import nn

from sglang.srt.speculative.domino_utils import domino_greedy_rollout


class TestDominoRollout(unittest.TestCase):
    def test_full_vocabulary_matches_sequential_reference(self):
        torch.manual_seed(13)
        hidden = torch.randn(2, 5, 8)
        anchors = torch.tensor([2, 7])
        embedding = nn.Embedding(32, 8)
        lm_head = nn.Linear(8, 32, bias=False)
        gru = nn.GRU(8, 4, batch_first=True, bias=False)
        projector = nn.Sequential(
            nn.Linear(12, 4, bias=False), nn.SiLU(), nn.Linear(4, 32, bias=False)
        )
        for shifted in (False, True):
            with self.subTest(shifted=shifted), torch.no_grad():
                z = hidden[:, :4] if shifted else hidden[:, 1:]
                first = lm_head(z[:, 0]).argmax(-1)
                expected = [first]
                _, state = gru(embedding(torch.stack((anchors, first), dim=1)))
                for index in range(1, 4):
                    correction = projector(torch.cat((z[:, index], state[0]), -1))
                    token = (lm_head(z[:, index]) + correction).argmax(-1)
                    expected.append(token)
                    _, state = gru(embedding(token[:, None]), state)
                actual = domino_greedy_rollout(
                    draft_hidden=hidden,
                    verified_ids=anchors,
                    target_embedding=embedding,
                    lm_head_weight=lm_head.weight,
                    prefix_gru=gru,
                    embed_proj=projector,
                    vocab_size=32,
                    shift_label=shifted,
                    candidate_pool_size=0,
                )
                torch.testing.assert_close(actual, torch.stack(expected, dim=1))

    def test_first_proposal_is_uncorrected(self):
        torch.manual_seed(9)
        hidden = torch.randn(2, 2, 8)
        weight = torch.randn(32, 8)
        actual = domino_greedy_rollout(
            draft_hidden=hidden,
            verified_ids=torch.tensor([2, 7]),
            target_embedding=None,
            lm_head_weight=weight,
            prefix_gru=None,
            embed_proj=None,
            vocab_size=32,
            shift_label=True,
        )
        expected = (hidden[:, 0] @ weight.T).argmax(-1, keepdim=True)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
