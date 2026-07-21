import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dhgbench"))

from lib_dataset.edge_loaders import (  # noqa: E402
    HEBatchGenerator,
    OBSERVED_EDGE_SPLIT_SCHEMA,
    build_observed_train_data,
    deduplicate_hyperedges,
    generate_observed_ind_split_hyperedges,
    generate_observed_split_hyperedges,
    hyperedges_from_data,
    load_train,
    load_val,
    observed_edge_split_dir,
)
from lib_dataset.preprocessing import data_processing  # noqa: E402
from lib_utils.metrics import evaluate_edge_loader  # noqa: E402
from parameter_parser import set_task_args  # noqa: E402


def make_hyperedge_index(hyperedges):
    rows = []
    cols = []
    for edge_id, nodes in enumerate(hyperedges):
        for node in nodes:
            rows.append(node)
            cols.append(edge_id)
    return torch.tensor([rows, cols], dtype=torch.long)


def make_raw_preprocessing_data():
    node_to_edge = torch.tensor(
        [[0, 1, 1, 2, 3], [4, 4, 5, 5, 5]],
        dtype=torch.long,
    )
    return SimpleNamespace(
        x=torch.ones(4, 2),
        y=torch.zeros(4, dtype=torch.long),
        edge_index=torch.cat([node_to_edge, node_to_edge.flip(0)], dim=1),
        n_x=torch.tensor([4]),
        num_hyperedges=torch.tensor([2]),
    )


class ObservedEdgePredictionProtocolTest(unittest.TestCase):
    def test_deduplicate_hyperedges_uses_node_set_identity(self):
        hyperedges = [
            [0, 1, 2],
            [2, 1, 0],
            [3, 4, 5],
            [3, 4, 5],
            [5, 6, 7],
        ]

        unique = deduplicate_hyperedges(hyperedges)

        self.assertEqual(unique, [frozenset([0, 1, 2]), frozenset([3, 4, 5]), frozenset([5, 6, 7])])

    def test_observed_protocol_uses_train_validation_test_partitions(self):
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
            [2, 1, 0],
            [7, 4, 2],
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
            split_path = (
                Path(tmpdir)
                / "trand_observed_v4"
                / "train_0.6_valid_0.2"
                / "toy"
                / "split_0.pt"
            )
            data_dict = torch.load(split_path, weights_only=False)

            args.edge_save_dir = f"{tmpdir}/repeat/"
            generate_observed_split_hyperedges(data, args, seed=0)
            repeat_path = (
                Path(tmpdir)
                / "repeat"
                / "trand_observed_v4"
                / "train_0.6_valid_0.2"
                / "toy"
                / "split_0.pt"
            )
            repeat_data_dict = torch.load(repeat_path, weights_only=False)
            self.assertEqual(data_dict, repeat_data_dict)

        train_pos = {frozenset(edge) for edge in data_dict["train_pos"]}
        valid_pos = {frozenset(edge) for edge in data_dict["valid_pos"]}
        test_pos = {frozenset(edge) for edge in data_dict["test_pos"]}
        all_positive = {frozenset(edge) for edge in hyperedges}

        self.assertEqual(data_dict["edge_pred_protocol"], "observed")
        self.assertEqual(data_dict["edge_split_schema"], OBSERVED_EDGE_SPLIT_SCHEMA)
        self.assertEqual(data_dict["train_prop"], 0.6)
        self.assertEqual(data_dict["valid_prop"], 0.2)
        self.assertTrue(train_pos)
        self.assertTrue(valid_pos)
        self.assertTrue(test_pos)
        self.assertTrue(train_pos.isdisjoint(valid_pos))
        self.assertTrue(train_pos.isdisjoint(test_pos))
        self.assertTrue(valid_pos.isdisjoint(test_pos))
        self.assertEqual(train_pos | valid_pos | test_pos, all_positive)
        self.assertEqual(len(train_pos), int(0.6 * len(all_positive)))
        self.assertLessEqual(abs(len(valid_pos) - len(test_pos)), 1)
        self.assertEqual(
            {node for edge in train_pos for node in edge},
            {node for edge in all_positive for node in edge},
        )
        train_loader = load_train(data_dict, bs=128, device="cpu", label="pos")
        val_loader = load_val(data_dict, bs=128, device="cpu", label="pos")
        self.assertEqual({frozenset(edge) for edge in train_loader.hyperedges}, train_pos)
        self.assertEqual({frozenset(edge) for edge in val_loader.hyperedges}, valid_pos)

        for split_name, positives in [
            ("train", train_pos),
            ("valid", valid_pos),
            ("test", test_pos),
        ]:
            for negative_kind in ["mns", "sns", "cns"]:
                key = f"{split_name}_{negative_kind}"
                negatives = {frozenset(edge) for edge in data_dict[key]}
                self.assertEqual(len(data_dict[key]), len(positives), key)
                self.assertTrue(negatives.isdisjoint(all_positive), key)

    def test_observed_split_expands_training_to_cover_every_node(self):
        hyperedges = [list(range(3 * idx, 3 * idx + 3)) for idx in range(7)]
        hyperedges += [[0, 3, 6], [9, 12, 15], [2, 5, 8]]
        data = SimpleNamespace(hyperedge_index=make_hyperedge_index(hyperedges))

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                train_prop=0.6,
                valid_prop=0.2,
                edge_save_dir=tmpdir,
                edge_split_mode="trand",
                edge_pred_protocol="observed",
                dname="toy",
                device="cpu",
            )
            with patch(
                "lib_dataset.edge_loaders.neg_generator_excluding",
                return_value=([], [], []),
            ):
                generate_observed_split_hyperedges(data, args, seed=0)
            split_path = Path(observed_edge_split_dir(args)) / "split_0.pt"
            data_dict = torch.load(split_path, weights_only=False)

        self.assertEqual(len(data_dict["train_pos"]), 7)
        self.assertEqual(len(data_dict["valid_pos"]), 1)
        self.assertEqual(len(data_dict["test_pos"]), 2)
        train_nodes = {node for edge in data_dict["train_pos"] for node in edge}
        all_nodes = {node for edge in hyperedges for node in edge}
        self.assertEqual(train_nodes, all_nodes)

    def test_data_processing_preserves_hyperedges_before_model_self_loops(self):
        data = data_processing(
            SimpleNamespace(
                method="HyperND",
                add_self_loop=True,
                exclude_self=False,
                normtype="all_one",
                device="cpu",
                task_type="edge_pred",
                edge_pred_protocol="observed",
            ),
            SimpleNamespace(data=make_raw_preprocessing_data(), sens=None),
        )

        canonical_hyperedges = set(
            hyperedges_from_data(data, data.canonical_hyperedge_index)
        )
        model_hyperedges = set(hyperedges_from_data(data))

        self.assertEqual(
            canonical_hyperedges,
            {frozenset([0, 1]), frozenset([1, 2, 3])},
        )
        self.assertEqual(
            model_hyperedges,
            canonical_hyperedges | {frozenset([node]) for node in range(4)},
        )
        self.assertEqual(data.canonical_hyperedge_index.device.type, "cpu")

    def test_unrelated_tasks_do_not_preserve_canonical_hyperedges(self):
        configurations = [
            ("node_cls", "legacy"),
            ("edge_pred", "legacy"),
        ]
        for task_type, edge_pred_protocol in configurations:
            with self.subTest(
                task_type=task_type,
                edge_pred_protocol=edge_pred_protocol,
            ):
                data = data_processing(
                    SimpleNamespace(
                        method="HGNN",
                        add_self_loop=False,
                        exclude_self=False,
                        normtype="all_one",
                        device="cpu",
                        task_type=task_type,
                        edge_pred_protocol=edge_pred_protocol,
                    ),
                    SimpleNamespace(
                        data=make_raw_preprocessing_data(),
                        sens=None,
                    ),
                )
                self.assertFalse(hasattr(data, "canonical_hyperedge_index"))

    def test_observed_edge_prediction_rejects_all_perturbations(self):
        perturbation_modes = [
            "spar_feat",
            "noise_feat",
            "drop_incidence",
            "add_incidence",
            "spar_label",
            "flip_label",
        ]
        for perturbation_mode in perturbation_modes:
            for is_perturbed in (True, "True"):
                for is_poison in (True, False):
                    with self.subTest(
                        perturbation_mode=perturbation_mode,
                        is_perturbed=is_perturbed,
                        is_poison=is_poison,
                    ):
                        args = SimpleNamespace(
                            task_type="edge_pred",
                            edge_pred_protocol="observed",
                            is_perturbed=is_perturbed,
                            pert_mode=perturbation_mode,
                            is_poison=is_poison,
                        )
                        with self.assertRaisesRegex(
                            ValueError,
                            "Perturbations are not supported",
                        ):
                            set_task_args(args)

        args = set_task_args(
            SimpleNamespace(
                task_type="edge_pred",
                edge_pred_protocol="observed",
                is_perturbed="False",
                dname="cora",
                method="HGNN",
                use_bench_prop=True,
                device="cpu",
            )
        )
        self.assertFalse(args.is_perturbed)

    def test_model_self_loops_are_message_passing_only(self):
        canonical_hyperedges = [
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
        data = SimpleNamespace(
            canonical_hyperedge_index=make_hyperedge_index(canonical_hyperedges),
            hyperedge_index=make_hyperedge_index(
                canonical_hyperedges + [[node] for node in range(10)]
            ),
            num_nodes=10,
            data=SimpleNamespace(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                train_prop=0.6,
                valid_prop=0.2,
                edge_save_dir=tmpdir,
                edge_split_mode="trand",
                edge_pred_protocol="observed",
                dname="toy",
                device="cpu",
                add_self_loop=True,
            )
            with patch(
                "lib_dataset.edge_loaders.neg_generator_excluding",
                return_value=([], [], []),
            ):
                generate_observed_split_hyperedges(data, args, seed=0)
            data_dict = torch.load(
                Path(observed_edge_split_dir(args)) / "split_0.pt",
                weights_only=False,
            )

        split_positives = {
            frozenset(edge)
            for key in ("train_pos", "valid_pos", "test_pos")
            for edge in data_dict[key]
        }
        canonical_positive_set = {
            frozenset(edge) for edge in canonical_hyperedges
        }
        self.assertEqual(split_positives, canonical_positive_set)
        self.assertTrue(all(len(edge) > 1 for edge in split_positives))

        train_data = build_observed_train_data(data, data_dict, args)
        train_positive_set = {
            frozenset(edge) for edge in data_dict["train_pos"]
        }
        self.assertEqual(
            set(hyperedges_from_data(train_data)),
            train_positive_set | {frozenset([node]) for node in range(10)},
        )
        train_loader = load_train(
            data_dict,
            bs=128,
            device="cpu",
            label="pos",
        )
        self.assertEqual(
            {frozenset(edge) for edge in train_loader.hyperedges},
            train_positive_set,
        )

    def test_train_data_uses_all_and_only_training_hyperedges(self):
        data = SimpleNamespace(
            hyperedge_index=make_hyperedge_index([[0, 1, 2], [2, 3, 4], [4, 5, 6]]),
            data=SimpleNamespace(),
        )
        data_dict = {
            "edge_split_schema": OBSERVED_EDGE_SPLIT_SCHEMA,
            "train_prop": 0.6,
            "valid_prop": 0.2,
            "train_pos": [[0, 1, 2], [2, 3, 4]],
        }
        args = SimpleNamespace(device="cpu", train_prop=0.6, valid_prop=0.2)

        train_data = build_observed_train_data(data, data_dict, args)

        self.assertEqual(train_data.num_hyperedges, 2)
        self.assertEqual(set(train_data.hyperedge_index[1].tolist()), {0, 1})
        self.assertEqual(train_data.hyperedge_index.shape[1], 6)
        self.assertEqual(
            set(hyperedges_from_data(train_data)),
            {frozenset([0, 1, 2]), frozenset([2, 3, 4])},
        )

    def test_observed_cache_and_artifact_validate_split_proportions(self):
        args = SimpleNamespace(
            edge_save_dir="/tmp/splits",
            edge_split_mode="trand",
            dname="toy",
            device="cpu",
            train_prop=0.6,
            valid_prop=0.2,
        )
        default_dir = observed_edge_split_dir(args)
        args.train_prop = 0.8
        args.valid_prop = 0.1
        custom_dir = observed_edge_split_dir(args)

        self.assertNotEqual(default_dir, custom_dir)
        self.assertTrue(default_dir.endswith("trand_observed_v4/train_0.6_valid_0.2/toy"))
        self.assertTrue(custom_dir.endswith("trand_observed_v4/train_0.8_valid_0.1/toy"))

        data = SimpleNamespace(
            hyperedge_index=make_hyperedge_index([[0, 1, 2], [2, 3, 4]]),
        )
        data_dict = {
            "edge_split_schema": OBSERVED_EDGE_SPLIT_SCHEMA,
            "train_prop": 0.6,
            "valid_prop": 0.2,
            "train_pos": [[0, 1, 2]],
        }
        with self.assertRaisesRegex(ValueError, "incompatible proportions"):
            build_observed_train_data(data, data_dict, args)

    def test_observed_split_rejects_invalid_or_empty_proportions(self):
        args = SimpleNamespace(
            edge_save_dir="/tmp/splits",
            edge_split_mode="trand",
            dname="toy",
            train_prop=0.6,
            valid_prop=0.2,
        )
        invalid_proportions = [
            (0.0, 0.2),
            (-0.1, 0.2),
            (0.6, 0.0),
            (0.6, -0.1),
            (0.8, 0.2),
            (0.9, 0.2),
            (0.7, 0.1),
            (float("inf"), 0.2),
            (0.6, float("nan")),
        ]
        for train_prop, valid_prop in invalid_proportions:
            with self.subTest(train_prop=train_prop, valid_prop=valid_prop):
                args.train_prop = train_prop
                args.valid_prop = valid_prop
                with self.assertRaisesRegex(ValueError, "proportions must be finite"):
                    observed_edge_split_dir(args)

        args.train_prop = 0.6
        args.valid_prop = 0.2
        data = SimpleNamespace(
            hyperedge_index=make_hyperedge_index(
                [[0, 1, 2], [2, 3, 4], [4, 5, 6], [0, 6, 7]]
            )
        )
        with self.assertRaisesRegex(ValueError, "non-empty train"):
            generate_observed_split_hyperedges(data, args, seed=0)

    def test_old_observed_split_schema_is_rejected(self):
        data = SimpleNamespace(
            hyperedge_index=make_hyperedge_index([[0, 1, 2], [2, 3, 4]]),
        )
        old_data_dict = {
            "edge_split_schema": "train_valid_test_cover_v2",
            "train_prop": 0.6,
            "valid_prop": 0.2,
            "support_pos": [[0, 1, 2]],
            "train_only_pos": [[2, 3, 4]],
        }

        with self.assertRaisesRegex(ValueError, "incompatible schema"):
            build_observed_train_data(
                data,
                old_data_dict,
                SimpleNamespace(device="cpu"),
            )

    def test_observed_ind_split_is_disabled(self):
        data = SimpleNamespace(
            hyperedge_index=make_hyperedge_index([[0, 1, 2], [2, 3, 4], [4, 5, 6]])
        )
        args = SimpleNamespace(
            train_prop=0.6,
            valid_prop=0.2,
            edge_save_dir="/tmp/",
            edge_split_mode="ind",
            edge_pred_protocol="observed",
            dname="toy",
            device="cpu",
        )

        with self.assertRaisesRegex(ValueError, "observed edge prediction protocol"):
            generate_observed_ind_split_hyperedges(data, args, seed=0)

    def test_singleton_edge_eval_returns_lists(self):
        class SingletonEvalModel:
            def eval(self):
                pass

            def encoding(self, data):
                return torch.zeros(2, 4), None

            def aggregate(self, nfeat, hedges, mode="Eval"):
                return [torch.tensor(0.0) for _ in hedges]

        loader = HEBatchGenerator([[0, 1]], [1], batch_size=128, device="cpu", test_generator=True)

        preds, labels = evaluate_edge_loader(SingletonEvalModel(), object(), loader)

        self.assertEqual(preds, [0.5])
        self.assertEqual(labels, [1.0])


if __name__ == "__main__":
    unittest.main()
