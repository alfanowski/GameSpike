"""`tt_bond_decay`: the geometric bond profile (ablation A8, docs/EXPERIMENT_LOG.md 14.7).

Four things are defended here, in descending order of how expensive they would be
to get wrong:

  1. BIT-IDENTITY AT THE DEFAULT. `tt_bond_decay=1.0` -- the default -- must
     reproduce the pre-change i.i.d. Gaussian construction EXACTLY, and passing it
     explicitly must not perturb the RNG stream. This repository's published results
     depend on existing constructions being reproducible bit-for-bit, and A8's whole
     sweep is confounded if the lambda=1.0 arm is not literally the old reservoir.
  2. THE INDEXING CONVENTION. A TT-matrix core is (r_left, m, n, r_right). The
     profile must touch axes 0 and 3 and NOTHING else. Applying it to a physical
     mode instead would still produce a plausible-looking entropy curve, which is
     precisely why it is asserted here against the axis structure directly rather
     than inferred from the entropy moving.
  3. THE ENTROPY ESTIMATOR ITSELF, validated against two cases whose answer is known
     in closed form (a product state must give S-bar = 0; a maximally entangled bond
     of dimension R must give S-bar = 1) plus one designed spectrum checked against
     the analytic Shannon entropy. A5/A6 and A8 all rest on this one method, so it
     is pinned rather than trusted.
  4. THAT THE KNOB ACTUALLY DOES THE JOB, at the production geometry: S-bar is
     monotone in lambda and reaches the band [0.1, 0.5] that neither
     `spectral_radius` (A5: provably zero effect) nor `tt_rank` (A6: 0.96221-0.99596)
     could reach.
"""
import math

import pytest
import torch

from models.spiking_reservoir import SpikingReservoir

# The production geometry (mirrors training.train.build_model's reservoir arm and
# tests/test_embedding_centering.py's RES_KWARGS), so the numbers pinned below are
# the numbers the experiment actually runs under.
PROD = dict(reservoir_size=8192, input_dim=32, use_tensor_train=True,
            tt_rank=8, tt_n_cores=4)
SMALL = dict(reservoir_size=512, input_dim=8, use_tensor_train=True,
             tt_rank=8, tt_n_cores=3)


def _cores(res):
    return [getattr(res, f"tt_core_{k}") for k in range(res.tt_n_cores)]


# --------------------------------------------------------------------------- #
# 1. lambda = 1.0 is the historical construction, bit for bit.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kwargs", [PROD, SMALL])
def test_default_is_bit_identical_to_the_construction_without_the_knob(kwargs):
    """The parameter must be invisible at its default. Compared as full state dicts
    (W_in included), because the profile is applied after W_in is drawn and a knob
    that perturbed the generator would show up there first."""
    implicit = SpikingReservoir(seed=0, **kwargs)
    explicit = SpikingReservoir(seed=0, tt_bond_decay=1.0, **kwargs)
    a, b = implicit.state_dict(), explicit.state_dict()
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), (
            f"tt_bond_decay=1.0 changed {key!r}; the default path is not bit-identical, "
            "which would make every existing TT result irreproducible"
        )
    assert implicit.tt_bond_decay == 1.0


@pytest.mark.parametrize("lam", [0.9, 0.7, 0.5, 0.1])
def test_the_profile_never_touches_the_rng_stream(lam):
    """A lambda sweep is a controlled comparison only if the six reservoirs differ by
    the deterministic profile and by NOTHING else. Asserted as: same W_in bit for
    bit, and each core exactly equal to the lambda=1.0 core times lambda^(a_l+a_r).
    `torch.equal`, not `allclose`: the profile is applied by multiplication after the
    draw, so the relation is exact in floating point, not approximate."""
    base = SpikingReservoir(seed=3, **SMALL)
    prof = SpikingReservoir(seed=3, tt_bond_decay=lam, **SMALL)
    assert torch.equal(base.W_in, prof.W_in), "the profile perturbed the W_in draw"
    assert base.tt_modes == prof.tt_modes and base.tt_ranks == prof.tt_ranks
    assert base.tt_core_std == prof.tt_core_std, "the profile changed the derived std"
    for k, (c0, cl) in enumerate(zip(_cores(base), _cores(prof))):
        r_left, m, n, r_right = c0.shape
        left = lam ** torch.arange(r_left, dtype=c0.dtype).view(-1, 1, 1, 1)
        right = lam ** torch.arange(r_right, dtype=c0.dtype).view(1, 1, 1, -1)
        assert torch.equal(cl, c0 * left * right), f"core {k} is not the profiled core"


# --------------------------------------------------------------------------- #
# 2. The indexing convention: bond axes are 0 and 3, and only those.
# --------------------------------------------------------------------------- #

def test_the_profile_is_applied_to_the_bond_axes_and_not_the_physical_modes():
    """Read the ratio core_lambda / core_1 directly and check it is constant over the
    physical axes (1, 2) and geometric over the bond axes (0, 3). This is the one
    assertion that would fail if the profile were put on a mode index -- an error
    that changes W_res's mode geometry rather than its bond correlations while still
    moving the entropy, i.e. exactly the wrong-but-plausible outcome."""
    lam = 0.6
    base = SpikingReservoir(seed=1, **SMALL)
    prof = SpikingReservoir(seed=1, tt_bond_decay=lam, **SMALL)
    for k, (c0, cl) in enumerate(zip(_cores(base), _cores(prof))):
        r_left, m, n, r_right = c0.shape
        ratio = (cl.double() / c0.double())
        # constant across both physical modes, for every fixed (a_left, a_right).
        # RELATIVE spread: the cores are float32, so the ratio of two stored values
        # carries ~1e-7 of rounding that has nothing to do with the profile.
        spread = ((ratio.amax(dim=(1, 2)) - ratio.amin(dim=(1, 2)))
                  / ratio.amax(dim=(1, 2))).abs().max().item()
        assert spread < 1e-6, f"core {k}: the profile varies over a PHYSICAL mode"
        # and exactly lambda^(a_left + a_right) across the bond axes
        for a in range(r_left):
            for b in range(r_right):
                assert ratio[a, 0, 0, b].item() == pytest.approx(lam ** (a + b), rel=1e-6)


def test_the_boundary_bonds_are_a_no_op_by_construction():
    """r_0 = r_d = 1, so the outer bonds carry only index 0 and lambda^0 = 1. The
    first core's left slice and the last core's right slice must therefore be
    untouched even at an aggressive lambda -- stated as a test so it stays a
    structural fact rather than a lucky coincidence of the shapes."""
    base = _cores(SpikingReservoir(seed=2, **SMALL))
    prof = _cores(SpikingReservoir(seed=2, tt_bond_decay=0.2, **SMALL))
    assert base[0].shape[0] == 1 and base[-1].shape[-1] == 1
    # the whole (unprofiled) right-bond-0 slab of the first core is unchanged
    assert torch.equal(prof[0][0, :, :, 0], base[0][0, :, :, 0])
    assert torch.equal(prof[-1][0, :, :, 0], base[-1][0, :, :, 0])


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, 2.0])
def test_out_of_range_decay_is_rejected(bad):
    with pytest.raises(ValueError, match="tt_bond_decay"):
        SpikingReservoir(seed=0, tt_bond_decay=bad, **SMALL)


def test_a_profiled_reservoir_still_steps():
    """The construction has to remain a usable reservoir, not just a nice spectrum."""
    res = SpikingReservoir(seed=0, tt_bond_decay=0.5, **SMALL)
    B, N = 4, SMALL["reservoir_size"]
    mem, spk = torch.zeros(B, N), torch.zeros(B, N)
    imem = torch.zeros(B, N)
    torch.manual_seed(0)
    x = torch.randn(B, SMALL["input_dim"])
    for _ in range(5):
        spk, mem, imem = res.step(x, mem, spk, imem)
    assert spk.shape == (B, N) and torch.isfinite(mem).all()
    # and the TT matvec still equals the materialised operator it stands for
    v = torch.randn(2, N)
    assert torch.allclose(res._tt_matvec(v), v @ res._tt_to_dense().T, atol=1e-5)


# --------------------------------------------------------------------------- #
# 3. The entropy estimator, against cases whose answer is known in closed form.
# --------------------------------------------------------------------------- #

def _reservoir_with_hand_built_cores(core0, core1):
    """A 2-core TT (16 = 4x4, bond 4) whose cores are replaced by designed ones.
    `entanglement_entropy` reads the buffers through `_tt_cores`, so overwriting them
    is enough -- and it keeps the validation on the SHIPPED method rather than on a
    re-implementation of it."""
    res = SpikingReservoir(reservoir_size=16, input_dim=4, use_tensor_train=True,
                           tt_rank=4, tt_n_cores=2, seed=0)
    assert [tuple(c.shape) for c in _cores(res)] == [(1, 4, 4, 4), (4, 4, 4, 1)]
    res.tt_core_0 = core0
    res.tt_core_1 = core1
    return res


def _left_orthonormal_first_core(seed=0):
    """(1,4,4,4) whose (m*n, r) flattening has orthonormal columns. Then the method's
    left-QR returns R = diag(+-1), so the centre core's singular values are exactly
    the ones the second core was built to carry -- which is what makes the two
    reference cases below EXACT rather than approximate."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(16, 4, generator=g), mode="reduced")
    return Q.reshape(1, 4, 4, 4).contiguous()


def _second_core_with_singular_values(sv, seed=1):
    """(4,4,4,1) whose (r, m*n) flattening has orthogonal rows with norms `sv`."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(16, 4, generator=g), mode="reduced")
    M = Q.transpose(0, 1) * torch.as_tensor(sv, dtype=Q.dtype).view(-1, 1)
    return M.reshape(4, 4, 4, 1).contiguous()


def test_entropy_of_a_product_state_is_zero():
    """Rank 1 across the cut = no entanglement. S-bar must be exactly 0."""
    core1 = _second_core_with_singular_values([1.0, 0.0, 0.0, 0.0])
    res = _reservoir_with_hand_built_cores(_left_orthonormal_first_core(), core1)
    assert res.entanglement_entropy() == 0.0


def test_entropy_of_a_maximally_entangled_bond_is_one():
    """A flat Schmidt spectrum over a bond of dimension R has S = log R, and the
    method normalises by log R, so S-bar must be 1."""
    core1 = _second_core_with_singular_values([0.5, 0.5, 0.5, 0.5])
    res = _reservoir_with_hand_built_cores(_left_orthonormal_first_core(), core1)
    assert res.entanglement_entropy() == pytest.approx(1.0, abs=1e-6)


def test_entropy_of_a_designed_spectrum_matches_the_analytic_value():
    """Neither reference case above exercises the -sum p log p sum on a non-degenerate
    spectrum, so here is one that does, checked against the closed form."""
    sv = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64)
    p = sv ** 2 / (sv ** 2).sum()
    expected = float(-(p * p.log()).sum() / math.log(4))
    res = _reservoir_with_hand_built_cores(
        _left_orthonormal_first_core(), _second_core_with_singular_values(sv.tolist()))
    assert res.entanglement_entropy() == pytest.approx(expected, abs=1e-6)
    assert 0.0 < expected < 1.0        # the case is genuinely intermediate


def test_entropy_is_invariant_to_a_global_rescaling_of_the_cores():
    """A5's proof, kept as a live test: S-bar is a function of the NORMALISED Schmidt
    spectrum, so rescaling every core cancels exactly. This is why `spectral_radius`
    and `tt_core_std` provably cannot move it -- and why `tt_bond_decay` had to break
    the i.i.d. assumption ALONG THE BOND INDEX instead of rescaling it."""
    res = SpikingReservoir(seed=0, **SMALL)
    before = res.entanglement_entropy()
    for k in range(res.tt_n_cores):
        setattr(res, f"tt_core_{k}", getattr(res, f"tt_core_{k}") * 1000.0)
    assert res.entanglement_entropy() == pytest.approx(before, abs=1e-9)


# --------------------------------------------------------------------------- #
# 4. The knob does the job, at the geometry the experiment actually runs.
# --------------------------------------------------------------------------- #

def test_sbar_is_monotone_in_lambda_and_reaches_the_productive_band():
    """A8/H8a's claim, at the production geometry, seed 0. The numbers are from the
    A8 sweep; they are pinned loosely (band membership and ordering, not digits) so
    this stays a test of the CONSTRUCTION rather than of one BLAS version."""
    sbar = {lam: SpikingReservoir(seed=0, tt_bond_decay=lam, **PROD).entanglement_entropy()
            for lam in (1.0, 0.9, 0.7, 0.5, 0.1)}
    values = [sbar[lam] for lam in (1.0, 0.9, 0.7, 0.5, 0.1)]
    assert all(a > b for a, b in zip(values, values[1:])), (
        f"S-bar is not monotone decreasing in lambda: {sbar}")
    # lambda=1.0 reproduces the i.i.d. Gaussian result A5/A6 measured (0.9918).
    assert sbar[1.0] > 0.98, f"lambda=1.0 no longer sits at the near-maximal S-bar: {sbar}"
    assert 0.1 <= sbar[0.7] <= 0.5, f"lambda=0.7 left the productive band: {sbar}"
    assert sbar[0.1] < 0.01, f"lambda=0.1 should be all but a product state: {sbar}"


def test_the_schmidt_spectrum_actually_decays_rather_than_the_entropy_moving_for_some_other_reason():
    """S-bar could in principle fall because the estimator's absolute `sv > 1e-12`
    cutoff started discarding singular values, not because the spectrum decayed.
    Check the cause directly: at lambda=1.0 the middle-bond spectrum is near-flat
    (A6's finding), at lambda=0.7 it is strongly decaying, and every one of the 8
    Schmidt values is still well clear of the cutoff in both cases."""
    def spectrum(lam):
        res = SpikingReservoir(seed=0, tt_bond_decay=lam, **PROD)
        cores = [c.reshape(c.shape[0], -1, c.shape[3]).clone().double() for c in _cores(res)]
        bond = res.tt_n_cores // 2
        for k in range(bond):
            rp, phys, rk = cores[k].shape
            Q, R = torch.linalg.qr(cores[k].reshape(rp * phys, rk), mode="reduced")
            cores[k] = Q.reshape(rp, phys, Q.shape[1])
            cores[k + 1] = torch.einsum("ij,jpk->ipk", R, cores[k + 1])
        for k in range(res.tt_n_cores - 1, bond, -1):
            rp, phys, rk = cores[k].shape
            Qt, Rt = torch.linalg.qr(cores[k].reshape(rp, phys * rk).transpose(0, 1),
                                     mode="reduced")
            L = Rt.transpose(0, 1)
            cores[k] = Qt.transpose(0, 1).reshape(L.shape[1], phys, rk)
            cores[k - 1] = torch.einsum("ipj,jk->ipk", cores[k - 1], L)
        c = cores[bond]
        sv = torch.linalg.svdvals(c.reshape(c.shape[0], -1))
        return sv, sv ** 2 / (sv ** 2).sum(), c.shape[0]

    p_by_lam = {}
    for lam in (1.0, 0.7):
        sv, p, rp = spectrum(lam)
        p_by_lam[lam] = p
        # this canonicalisation is the method's own, so it must reproduce it exactly
        s = float(-(p * p.log()).sum() / math.log(rp))
        assert s == pytest.approx(
            SpikingReservoir(seed=0, tt_bond_decay=lam, **PROD).entanglement_entropy(),
            abs=1e-9)
        assert sv.min() > 1e-9, (
            f"lambda={lam}: a Schmidt value is near the estimator's 1e-12 cutoff, so "
            f"the entropy could be an artefact of truncation: {sv.tolist()}")
    flat, decayed = p_by_lam[1.0], p_by_lam[0.7]
    assert flat.max() / flat.min() < 3.0, "lambda=1.0 should be a near-flat spectrum"
    assert decayed.max() / decayed.min() > 1e3, "lambda=0.7 should be strongly decaying"
