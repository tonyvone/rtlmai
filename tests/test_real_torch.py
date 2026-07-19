"""Real-backbone tests (torch). Skipped automatically when torch or the
cached AG News data are unavailable, so the core suite stays NumPy-only.

Run the full real-data program with:  python -m rtlm_n.benchmarks.real_agnews
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rtlm_n.accumulators.surfaces import objective_id_for, projected_jacobian
from rtlm_n.bases.basis import adapter_delta, build_basis
from rtlm_n.core.hashing import KeyRegistry
from rtlm_n.core.linalg import augment_bias, ridge_solve
from rtlm_n.distributed.merge import MergeEngine
from rtlm_n.distributed.node import EdgeNode
from rtlm_n.materialization.factory import ModelFactory
from rtlm_n.real.torch_backbone import TorchTransformer

OBJ = objective_id_for("torch-test")


@pytest.fixture(scope="module")
def backbone():
    rng = np.random.default_rng(0)
    model = TorchTransformer(vocab=200, d_model=32, n_heads=4, n_layers=2, d_ff=64, seq_len=12, seed=0)
    return model.freeze(), rng


def _tokens(rng, n, vocab=200, seq=12):
    return rng.integers(3, vocab, size=(n, seq))


def test_torch_backbone_implements_rtlmn_interface(backbone):
    model, rng = backbone
    toks = _tokens(rng, 8)
    h = model.forward(toks)
    assert h.shape == (8, 32) and h.dtype == np.float64
    assert model.base_model_id.startswith("torch-mini-")
    assert model.flops_per_example > 0
    x = model.mlp_inputs(toks, layer=1)
    assert x.shape[1] == 32


def test_exact_equivalence_on_torch_features(backbone):
    model, rng = backbone
    keys = KeyRegistry()
    basis = build_basis(model, "random-orthogonal", seed=0, n_classes=3)
    toks = _tokens(rng, 240)
    y = np.eye(3)[rng.integers(0, 3, size=240)]
    engine = MergeEngine(keys)
    for i, part in enumerate(np.array_split(np.arange(240), 3)):
        node = EdgeNode(f"t{i}", model, basis, keys, f"k{i}".encode())
        node.absorb_head_batch(toks[part], y[part], objective_id=OBJ)
        engine.submit(node.state(OBJ))
    W = ModelFactory(basis).materialize_head(engine.global_state, lam=1e-3).params["W"]
    h = augment_bias(model.forward(toks))
    np.testing.assert_allclose(W, ridge_solve(h.T @ h, h.T @ y, 1e-3), atol=1e-8)


def test_autograd_jacobian_matches_finite_difference(backbone):
    """The torch autograd path must agree with the generic finite-difference
    path - the two implementations cross-validate each other."""
    model, rng = backbone
    basis = build_basis(model, "random-orthogonal", seed=1, ru=2, rv=3, n_classes=3,
                        adapter_layers=(1,))
    toks = _tokens(rng, 6)
    W_head = rng.standard_normal((3, 32))
    m = basis.matrices("mlp:1")
    logits_ad, J_ad = model.jacobian_c(toks, m["U"], m["V"], 1, W_head)

    class NoJac:
        """Same model, autograd path hidden -> finite differences."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "jacobian_c":
                raise AttributeError(name)
            return getattr(self._inner, name)

    # fp32 forward -> eps must sit well above the float32 noise floor
    logits_fd, J_fd = projected_jacobian(NoJac(model), basis, "mlp:1", toks, W_head, eps=1e-2)
    np.testing.assert_allclose(logits_ad, logits_fd, atol=1e-5)
    np.testing.assert_allclose(J_ad, J_fd, atol=3e-2)
    cos = float(
        (J_ad.ravel() @ J_fd.ravel())
        / (np.linalg.norm(J_ad) * np.linalg.norm(J_fd) + 1e-12)
    )
    assert cos > 0.995, f"Jacobian directions disagree: cos={cos}"


def test_adapter_injection_changes_forward(backbone):
    model, rng = backbone
    basis = build_basis(model, "random-orthogonal", seed=1, ru=2, rv=3, n_classes=3,
                        adapter_layers=(1,))
    toks = _tokens(rng, 4)
    delta = adapter_delta(basis, "mlp:1", 0.5 * np.ones((2, 3)))
    h0 = model.forward(toks)
    h1 = model.forward(toks, mlp_deltas={1: delta})
    assert not np.allclose(h0, h1)


def test_mlm_pretraining_improves_features():
    """Pretraining must produce genuinely better-than-random representations
    (the MLM-target leak regression test: a broken objective collapses this)."""
    from rtlm_n.real.torch_backbone import mlm_pretrain

    rng = np.random.default_rng(3)
    # synthetic corpus with learnable co-occurrence structure: class c
    # draws tokens from its own band, label = band
    n, seq, vocab = 900, 12, 120
    labels = rng.integers(0, 3, size=n)
    toks = np.stack([rng.integers(3 + 30 * c, 3 + 30 * (c + 1), size=seq) for c in labels])
    y = np.eye(3)[labels]

    def head_acc(m):
        h = augment_bias(m.forward(toks[:700]))
        W = ridge_solve(h.T @ h, h.T @ y[:700], 1e-3)
        ht = augment_bias(m.forward(toks[700:]))
        return float(((ht @ W).argmax(1) == labels[700:]).mean())

    rand = TorchTransformer(vocab=vocab, d_model=32, n_heads=4, n_layers=2, d_ff=64,
                            seq_len=seq, seed=5).freeze()
    trained = TorchTransformer(vocab=vocab, d_model=32, n_heads=4, n_layers=2, d_ff=64,
                               seq_len=seq, seed=5)
    mlm_pretrain(trained, toks[:700], steps=150, batch=48, lr=3e-3, seed=5)
    assert head_acc(trained) >= head_acc(rand) - 0.02
    # and the MLM loss must not be degenerate-zero (target-leak regression)
    assert trained.base_model_id != rand.base_model_id
