import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dhgbench"))

from lib_utils.aggregator import MaxAggregator, MaxminAggregator  # noqa: E402
from lib_dataset.edge_loaders import HEBatchGenerator  # noqa: E402
from lib_utils import train_agent  # noqa: E402
from lib_utils.train_agent import Trainer  # noqa: E402


class FixedClassifier(torch.nn.Module):
    def forward(self, embedding):
        return torch.tensor([[-2.0], [0.0], [2.0]], dtype=embedding.dtype, device=embedding.device)


class TrainableEdgeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.zero_()

    def encoding(self, data):
        return torch.zeros(3, 4), None

    def aggregate(self, nfeat, hedges, mode="Train"):
        return self.weight.expand(len(hedges))


class EdgePredictionLogitContractTest(unittest.TestCase):
    def test_max_aggregators_return_raw_logits(self):
        for aggregator_cls in [MaxAggregator, MaxminAggregator]:
            with self.subTest(aggregator=aggregator_cls.__name__):
                aggregator = aggregator_cls.__new__(aggregator_cls)
                torch.nn.Module.__init__(aggregator)
                aggregator.classifier = FixedClassifier()

                logits = aggregator.classify(torch.zeros(3, 4))

                expected = torch.tensor([[-2.0], [0.0], [2.0]])
                self.assertTrue(torch.equal(logits.cpu(), expected))
                self.assertLess(float(logits.min()), 0.0)
                self.assertGreater(float(logits.max()), 1.0)

    def test_edge_training_uses_logits_for_bce_loss(self):
        source = (ROOT / "dhgbench" / "lib_utils" / "train_agent.py").read_text()

        self.assertIn("F.binary_cross_entropy_with_logits(pos_preds, pos_labels)", source)
        self.assertIn("F.binary_cross_entropy_with_logits(neg_preds, neg_labels)", source)
        self.assertNotIn("torch.sigmoid(pos_preds)", source)
        self.assertNotIn("torch.sigmoid(neg_preds)", source)

    def test_edge_early_stop_evaluates_final_epoch_before_return(self):
        def edge_metrics(prefix):
            if prefix == "train":
                return {"roc_train": 0.5, "ap_train": 0.5}
            metrics = {}
            for name in ["sns", "mns", "cns", "mixed", "average"]:
                metrics[f"roc_{name}"] = 0.5
                metrics[f"ap_{name}"] = 0.5
            return metrics

        args = SimpleNamespace(
            device="cpu",
            lr=0.01,
            wd=0.0,
            epochs=1,
            display_step=20,
            early_stop=True,
        )
        batch_loaders = {
            "train_pos": HEBatchGenerator([[0, 1]], [1], batch_size=1, device="cpu"),
            "train_neg": HEBatchGenerator([[1, 2]], [0], batch_size=1, device="cpu"),
        }
        original_evaluate_edge = train_agent.evaluate_edge
        train_agent.evaluate_edge = lambda model, data, loaders: (
            edge_metrics("train"),
            edge_metrics("val"),
            edge_metrics("test"),
        )
        try:
            model = Trainer(args).semi_edge_pred_training(
                TrainableEdgeModel(), object(), batch_loaders, args
            )
        finally:
            train_agent.evaluate_edge = original_evaluate_edge

        self.assertIsInstance(model, TrainableEdgeModel)


if __name__ == "__main__":
    unittest.main()
