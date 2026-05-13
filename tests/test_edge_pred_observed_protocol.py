import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dhgbench"))

from lib_dataset.edge_loaders import (  # noqa: E402
    build_observed_support_data,
    generate_observed_split_hyperedges,
    load_val,
)


def make_hyperedge_index(hyperedges):
    rows = []
    cols = []
    for edge_id, nodes in enumerate(hyperedges):
        for node in nodes:
            rows.append(node)
            cols.append(edge_id)
    return torch.tensor([rows, cols], dtype=torch.long)


class ObservedEdgePredictionProtocolTest(unittest.TestCase):
    def test_observed_protocol_separates_support_targets_and_negatives(self):
        hyperedges = [
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 6],
            [0, 6, 7],
            [1, 3, 5],
            [2, 5, 7],
            [0, 3, 6],
            [1, 4, 7],
            [0, 2, 5],
            [3, 6, 7],
            [1, 5, 6],
            [2, 4, 7],
        ]
        data = SimpleNamespace(hyperedge_index=make_hyperedge_index(hyperedges))
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                train_prop=0.6,
                valid_prop=0.2,
                edge_save_dir=f"{tmpdir}/",
                edge_split_mode="trand",
                edge_pred_protocol="observed",
                dname="toy",
                device="cpu",
            )
            generate_observed_split_hyperedges(data, args, seed=0)
            split_path = Path(tmpdir) / "trand_observed" / "toy" / "split_0.pt"
            data_dict = torch.load(split_path, weights_only=False)

        support = {frozenset(edge) for edge in data_dict["support_pos"]}
        train_pos = {frozenset(edge) for edge in data_dict["train_only_pos"]}
        valid_pos = {frozenset(edge) for edge in data_dict["ground_valid"] + data_dict["valid_only_pos"]}
        test_pos = {frozenset(edge) for edge in data_dict["test_pos"]}
        all_positive = {frozenset(edge) for edge in hyperedges}

        self.assertEqual(data_dict["edge_pred_protocol"], "observed")
        self.assertTrue(support)
        self.assertTrue(train_pos)
        self.assertTrue(valid_pos)
        self.assertTrue(test_pos)
        self.assertTrue(support.isdisjoint(valid_pos))
        self.assertTrue(support.isdisjoint(test_pos))

        val_loader = load_val(data_dict, bs=128, device="cpu", label="pos")
        self.assertEqual({frozenset(edge) for edge in val_loader.hyperedges}, valid_pos)

        for key in [
            "train_mns",
            "train_sns",
            "train_cns",
            "valid_mns",
            "valid_sns",
            "valid_cns",
            "test_mns",
            "test_sns",
            "test_cns",
        ]:
            self.assertTrue({frozenset(edge) for edge in data_dict[key]}.isdisjoint(all_positive), key)

    def test_support_data_uses_only_observed_hyperedges(self):
        data = SimpleNamespace(
            hyperedge_index=make_hyperedge_index([[0, 1, 2], [2, 3, 4], [4, 5, 6]]),
            data=SimpleNamespace(),
        )
        data_dict = {"support_pos": [[0, 1, 2], [2, 3, 4]]}
        args = SimpleNamespace(device="cpu")

        support_data = build_observed_support_data(data, data_dict, args)

        self.assertEqual(support_data.num_hyperedges, 2)
        self.assertEqual(set(support_data.hyperedge_index[1].tolist()), {0, 1})
        self.assertEqual(support_data.hyperedge_index.shape[1], 6)


if __name__ == "__main__":
    unittest.main()
