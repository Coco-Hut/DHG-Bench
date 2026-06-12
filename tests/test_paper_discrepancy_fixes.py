from pathlib import Path
from types import SimpleNamespace
import os
import sys

import torch
from torch_geometric.data import Data


REPO_ROOT = Path(__file__).resolve().parents[1]
DHGBENCH_ROOT = REPO_ROOT / "dhgbench"
sys.path.insert(0, str(DHGBENCH_ROOT))


def _configured_args(method, task_type="node_cls", dname="cora"):
    from parameter_parser import method_config, parameter_parser

    old_argv = sys.argv[:]
    old_cwd = os.getcwd()
    try:
        os.chdir(DHGBENCH_ROOT)
        sys.argv = [
            "test",
            "--method",
            method,
            "--task_type",
            task_type,
            "--dname",
            dname,
            "--is_default",
            "True",
            "--device",
            "cpu",
        ]
        return method_config(parameter_parser())
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def test_hcha_paper_uses_attention_branch_and_runs_forward():
    from lib_models.HNN.hgnn import HCHA

    args = _configured_args("HCHA_Paper")
    args.All_num_layers = 2
    args.MLP_hidden = 16
    args.dropout = 0.0

    model = HCHA(num_features=4, num_targets=3, args=args)
    model.eval()

    assert all(conv.use_attention for conv in model.convs)
    assert model.convs[0].heads == 8
    assert model.convs[0].attention_mode == "edge"
    assert model.convs[-1].heads == 1

    data = Data(
        x=torch.arange(16, dtype=torch.float32).view(4, 4) / 10.0,
        hyperedge_index=torch.tensor(
            [
                [0, 1, 2, 1, 2, 3, 0, 3],
                [0, 0, 1, 1, 2, 2, 3, 4],
            ],
            dtype=torch.long,
        ),
    )
    out, edge_embed = model(data)

    assert out.shape == (4, 3)
    assert edge_embed.shape == (5, 3)
    assert torch.isfinite(out).all()
    assert torch.isfinite(edge_embed).all()


def test_hypergt_paper_preprocessing_uses_mean_hyperedge_features():
    from lib_models.HNN.preprocessing import hypergt_preprocessing

    args = _configured_args("HyperGT_Paper")
    assert args.hefeat == "mean"
    assert args.pe == "HEPEHtEPE"
    assert args.use_edge_loss is True

    original_x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 8.0],
        ]
    )
    data = Data(
        x=original_x.clone(),
        hyperedge_index=torch.tensor(
            [
                [0, 1, 1, 2],
                [5, 5, 6, 6],
            ],
            dtype=torch.long,
        ),
        num_nodes=3,
    )

    out = hypergt_preprocessing(data, args, add_reverse=False, add_token_loops=True)

    expected_hyperedge_x = torch.tensor(
        [
            [2.0, 3.0],
            [4.0, 6.0],
        ]
    )
    assert out.num_tokens == 5
    assert out.H.shape == torch.Size([3, 2])
    assert torch.allclose(out.x[:3], original_x)
    assert torch.allclose(out.x[3:], expected_hyperedge_x)

    loops = out.adjs[0][:, -out.num_tokens :]
    assert torch.equal(loops[0], torch.arange(out.num_tokens))
    assert torch.equal(loops[1], torch.arange(out.num_tokens))


def test_dphgnn_paper_alias_selects_paper_dff():
    from lib_models.HNN.dphgnn import DFF, DPHGNN, PaperDFF

    args = _configured_args("DPHGNN_Paper")
    assert args.DPHGNN_dff_mode == "paper"
    assert args.mediator is True

    args.expan_dim = 4
    args.taa_spatial_dim = 4
    args.taa_spectral_dim = 4
    args.spectral_embed_dim = 4
    args.fc_dim = 4
    args.dff_MLP_hidden = 4
    args.dff_num_layers = 1
    args.cg_num_layer = args.hg_num_layer = args.sg_num_layer = 1
    args.cg_MLP_hidden = args.hg_MLP_hidden = args.sg_MLP_hidden = 4
    args.cg_dropout = args.hg_dropout = args.sg_dropout = 0.0
    args.num_heads = 1
    args.chunk_size = -1
    args.device = "cpu"

    paper_model = DPHGNN(num_features=3, num_targets=2, args=args)
    assert isinstance(paper_model.dff_layer, PaperDFF)

    args.DPHGNN_dff_mode = "attention"
    plain_model = DPHGNN(num_features=3, num_targets=2, args=args)
    assert isinstance(plain_model.dff_layer, DFF)


def test_paper_dphgnn_conv_matches_additive_dff_equation():
    from lib_models.HNN.dphgnn import PaperDPHGNNConv

    args = SimpleNamespace(expan_dim=3)
    conv = PaperDPHGNNConv(in_channels=2, out_channels=2, args=args, drop_rate=0.0)
    conv.eval()

    with torch.no_grad():
        conv.skip_proj.weight.zero_()
        conv.skip_proj.bias.zero_()
        conv.msg_proj.weight.copy_(torch.eye(2))
        conv.msg_proj.bias.zero_()
        conv.star_proj.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            )
        )
        conv.star_proj.bias.zero_()

    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 1.0],
        ]
    )
    s_features = torch.tensor(
        [
            [2.0, 4.0, 99.0],
            [6.0, 8.0, 99.0],
        ]
    )
    hyperedge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [0, 0, 1, 1],
        ],
        dtype=torch.long,
    )

    out = conv(x, hyperedge_index, s_features)

    degree_v = torch.tensor([1.0, 2.0, 1.0])
    degree_e = torch.tensor([2.0, 2.0])
    dv_inv_sqrt = degree_v.pow(-0.5)
    de_inv = degree_e.pow(-1.0)
    v, e = hyperedge_index

    node_to_edge = torch.zeros(2, 2)
    node_to_edge.index_add_(0, e, x[v] * dv_inv_sqrt[v].unsqueeze(-1))
    star_term = s_features[:, :2] * de_inv.unsqueeze(-1)
    fused_edge = node_to_edge + star_term

    expected = torch.zeros(3, 2)
    expected.index_add_(0, v, fused_edge[e] * de_inv[e].unsqueeze(-1))

    assert torch.allclose(out, expected, atol=1e-6)
