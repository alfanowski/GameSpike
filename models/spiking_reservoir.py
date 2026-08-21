import math
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

NEURON_MODELS = ("lif", "rf")


class ResonateFireCell(nn.Module):
    """Discrete-time resonate-and-fire neuron, exactly as pre-registered in
    docs/EXPERIMENT_LOG.md §23.2. One complex pole per unit, carried in REAL
    2-vector form (u, v) so no complex autograd is involved:

        u_next = beta * (cos(w_i)*u - sin(w_i)*v) + I
        v_next = beta * (sin(w_i)*u + cos(w_i)*v)

    `u` is the membrane -- it receives the input current and it is the only
    component the threshold and the reset touch. `v` is its quadrature companion:
    no input, no threshold, no reset. |lambda| = beta is IDENTICAL to the LIF
    path's beta, so the envelope decay -- and hence the memory horizon -- is
    unchanged by construction; only the rotation is added (§23.2).

    LIF IS THE omega == 0 POINT OF THIS FAMILY, BIT FOR BIT, AND THAT IS A TEST
    (§23.5 G0e-ii), NOT A REMARK. At w == 0: cos = 1.0 and sin = 0.0 exactly in
    IEEE-754, `1.0*u` is exact, `0.0*v` is +0.0 for finite v, `u - 0.0` is exact,
    so the update collapses to `beta*u + I` -- the literal `snn.Leaky` arithmetic.
    That only holds if the expression is evaluated in snntorch's order, which is
    why the code below is written the way it is rather than the algebraically
    equivalent `beta*cos*u - beta*sin*v + I`.

    RESET SEMANTICS ARE SNNTORCH'S, NOT THE PSEUDOCODE'S. §23.2 fixes
    `reset_delay=True`, and snntorch's `Leaky.forward` implements that as: the
    reset signal is derived from the PREVIOUS step's membrane, BEFORE this step's
    threshold comparison, and subtracted inside the state update --

        reset   = Theta(u_prev - theta).detach()
        u_next  = beta*rot(u_prev, v_prev) + I - reset*theta
        spk     = Theta(u_next - theta)                    # returned un-reset

    -- so the membrane handed back to the caller has NOT yet had this step's
    spike subtracted; that happens one step later. §23.2's pseudocode describes
    the same mechanism with the subtraction written at the end of the step, which
    is `reset_delay=False`. snntorch's arithmetic is the specification here
    because G0e-ii demands bit-exactness against `snn.Leaky`, and the two
    orderings are NOT bit-equal (they differ by one step of delay on every reset).

    beta and theta are the LIF path's own constants, taken from the same source
    (`SpikingReservoir.__init__`'s `beta` argument and `snn.Leaky`'s threshold),
    and the surrogate gradient is snntorch's default -- `surrogate.atan()`, alpha
    2.0, which is what `snn.Leaky()` installs when constructed with no
    `spike_grad=` argument (snntorch/_neurons/neurons.py: `self.spike_grad =
    atan()`). Imported rather than reimplemented so it cannot silently drift.

    Holds NO state: (u, v) are threaded explicitly by the caller, exactly as
    `mem`/`spk` already are. That is deliberate -- snntorch's `Leaky` keeps a
    transient `mem` buffer that the frozen-reservoir tripwire has to special-case
    (models/policy_value_reservoir.py's TRANSIENT_RESERVOIR_BUFFERS); this cell
    adds no such exception, every buffer it owns is a frozen constant.
    """

    def __init__(self, omega, beta=0.9, threshold=1.0, spike_grad=None):
        super().__init__()
        if not isinstance(omega, torch.Tensor):
            omega = torch.as_tensor(omega)
        omega = omega.detach().clone().to(torch.get_default_dtype())
        # Frequencies and their cosines/sines are frozen constants: registered as
        # buffers so they move with .to()/.cuda() and are covered, unexceptioned,
        # by PolicyValueReservoir.assert_reservoir_frozen().
        self.register_buffer("omega", omega)
        self.register_buffer("cos_omega", torch.cos(omega))
        self.register_buffer("sin_omega", torch.sin(omega))
        # Same construction snntorch uses for its own beta/threshold buffers
        # (torch.as_tensor on a Python float -> a float32 scalar tensor), so the
        # stored constants are bit-identical to snn.Leaky's.
        self.register_buffer("beta", torch.as_tensor(beta)
                             if not isinstance(beta, torch.Tensor)
                             else beta.detach().clone())
        self.register_buffer("threshold", torch.as_tensor(threshold)
                             if not isinstance(threshold, torch.Tensor)
                             else threshold.detach().clone())
        self.spike_grad = surrogate.atan() if spike_grad is None else spike_grad

    def forward(self, input_, u, v):
        """One timestep. input_/u/v: (B, N). Returns (spk, u_next, v_next).

        Every line below mirrors a line of `snn.Leaky.forward` +
        `Leaky._base_sub`; the evaluation order is load-bearing (see class doc)."""
        beta = self.beta.clamp(0, 1)
        # snntorch: `self.reset = self.mem_reset(self.mem)` -- previous membrane,
        # detached, before the state update.
        reset = self.spike_grad(u - self.threshold).clone().detach()
        rot_u = self.cos_omega * u - self.sin_omega * v
        rot_v = self.sin_omega * u + self.cos_omega * v
        # snntorch: `(beta*mem + input_) - reset*threshold`, in that association.
        u_next = beta * rot_u + input_ - reset * self.threshold
        v_next = beta * rot_v
        # snntorch: `spk = self.fire(self.mem)` on the ALREADY-updated membrane.
        spk = self.spike_grad(u_next - self.threshold)
        return spk, u_next, v_next


def _balanced_factorization(n, n_cores):
    """Factor `n` into up to `n_cores` integer factors (>1) whose product is `n`,
    as balanced as possible. Used to pick the TT-matrix mode dimensions m_k=n_k so
    that prod(m_k) == reservoir_size. Falls back to fewer cores if `n` has fewer
    prime factors than requested (e.g. a prime `n` yields a single factor -> no
    valid TT decomposition, caught by the caller)."""
    primes = []
    m = n
    d = 2
    while d * d <= m:
        while m % d == 0:
            primes.append(d)
            m //= d
        d += 1
    if m > 1:
        primes.append(m)
    if len(primes) < n_cores:
        return primes if primes else [n]
    # distribute prime factors (largest first) into the currently-smallest bucket,
    # greedily balancing the per-core product.
    primes.sort(reverse=True)
    factors = [1] * n_cores
    for p in primes:
        i = min(range(n_cores), key=lambda k: factors[k])
        factors[i] *= p
    factors.sort()
    return factors


class SpikingReservoir(nn.Module):
    def __init__(self, vocab_size=256, reservoir_size=2048, spectral_radius=1.0, beta=0.9, seed=0,
                 use_tensor_train=False, tt_rank=8, tt_n_cores=4, tt_core_std=None,
                 tt_bond_decay=1.0,
                 input_dim=None, soft_spike=False, soft_spike_sigma_frac=0.1,
                 neuron_model="lif", rf_period_min=2.0, rf_period_max=32.0):
        """Frozen spiking reservoir (LIF neurons, random recurrent wiring).

        `input_dim` controls the width of the frozen input matrix W_in (reservoir_size
        x input_dim), i.e. the dimensionality of the feature vector fed to the reservoir
        at each timestep. It defaults to `vocab_size`, which keeps the original behavior
        EXACTLY: the caller feeds a (B, T, vocab_size) one-hot byte sequence (CMA-ES
        path). Set input_dim to a small embedding width to feed a trainable
        nn.Embedding(vocab_size, input_dim)'s output instead of raw one-hot bytes
        (backprop path, §13): W_in then maps embed_dim -> reservoir_size, and the input
        to forward()/step() is a dense (B[,T], input_dim) feature vector rather than a
        one-hot. Nothing else about the reservoir changes; W_in stays frozen either way.

        Two construction modes for the recurrent weight matrix W_res (reservoir_size
        x reservoir_size):
          * use_tensor_train=False (DEFAULT, unchanged): a dense random matrix,
            rescaled to `spectral_radius` via the largest eigenvalue magnitude.
          * use_tensor_train=True: W_res is built NATIVELY as a tensor-train / matrix
            product operator (MPO) -- a chain of small random cores whose contraction
            reconstructs the effective NxN matrix. The dense NxN matrix is NEVER
            materialized (neither at construction nor in the forward/step hot path);
            the recurrent matrix-vector product contracts the cores directly against
            the (dense, batched) spike vector. This keeps stored parameters tiny as
            the reservoir scales up (reservoir_size^2 -> a few thousand core entries).

        W_in (reservoir_size x vocab_size) stays dense in both modes: it grows only
        linearly with reservoir_size, so it is not the term TT construction targets.

        TT knobs (only used when use_tensor_train=True):
          tt_rank    -- internal bond dimension (same for every internal bond). Per
                        Sato et al. 2025, bond dimension drives an order-to-chaos
                        transition analogous to the dense spectral radius.
          tt_n_cores -- target number of cores; reservoir_size is factored into that
                        many balanced mode dimensions (m_k = n_k).
          tt_core_std-- std of the i.i.d. Gaussian core entries. If None, it is
                        derived so the induced per-entry variance of the effective
                        W_res matches a dense reservoir at `spectral_radius`
                        (v = spectral_radius^2 / N); see _build_tt_cores. This is the
                        second, finer criticality knob (empirically tunable if the
                        reservoir fires silent or saturates).
          tt_bond_decay-- geometric bond profile ("lambda"), in (0, 1]. DEFAULT 1.0 =
                        NO-OP = the historical i.i.d. Gaussian construction, bit for
                        bit. Below 1.0 it breaks the i.i.d. assumption ALONG THE BOND
                        INDEX, which is the only thing that can move the normalised
                        entanglement entropy; see _build_tt_cores for the full
                        argument and for why `spectral_radius` and `tt_core_std`
                        provably cannot.

        NEURON MODEL (docs/EXPERIMENT_LOG.md §23, ablation A3):
          neuron_model  -- "lif" (DEFAULT, unchanged) or "rf". "rf" swaps the LIF
                        cell for `ResonateFireCell`: each unit keeps the same
                        |lambda| = beta envelope but acquires a frozen rotation
                        w_i, so its DC gain |1/(1 - beta*exp(i*w))| collapses while
                        its AC accumulation gain 1/sqrt(1-beta^2) is untouched.
                        The state gains a second component (the quadrature
                        companion); it adds ZERO trainable parameters, and the
                        readout still sees an N-wide binary spike vector.
          rf_period_min/rf_period_max -- support of the resonant-period draw, in env
                        steps. §23.2 fixes [2, 32]: 2 is the Nyquist period of the
                        discrete-time system, 32 is a quarter of the 128-step
                        truncated-BPTT window, so every unit completes >= 4 cycles
                        inside the horizon the readout ever sees a gradient over.
                        T_i is drawn i.i.d. LOG-uniform (equal density per octave
                        across the five octaves spanned) and w_i = 2*pi/T_i.

        THE `omega` DRAW IS THE LAST THING THAT TOUCHES THE GENERATOR, AND ONLY IN
        THE "rf" BRANCH. §23.5's G0e-i requires the `neuron_model="lif"` path to be
        bit-identical to the code that produced the published v2 runs -- those runs
        are the pilot's experimental control, so a shifted RNG stream would not
        merely change a number, it would silently substitute a different control.
        Placing the draw after W_in and W_res/the TT cores, inside the branch,
        guarantees every pre-existing draw consumes exactly the generator values it
        consumed before this parameter existed. Pinned in
        tests/test_resonate_and_fire.py against `git show 708b32d:`.
        """
        super().__init__()
        if neuron_model not in NEURON_MODELS:
            raise ValueError(
                f"unknown neuron_model: {neuron_model!r}; expected one of {NEURON_MODELS}")
        self.neuron_model = neuron_model
        self.rf_period_min = float(rf_period_min)
        self.rf_period_max = float(rf_period_max)
        self.reservoir_size = reservoir_size
        self.use_tensor_train = use_tensor_train
        if input_dim is None:
            input_dim = vocab_size
        self.input_dim = input_dim

        # --- soft-spike (stochastic-resonance) OUTPUT feature -- OPT-IN --------- #
        # When soft_spike=True the reservoir exposes the graded Gaussian-CDF
        # "soft spike" Phi((mem - theta)/sigma) to the downstream readout INSTEAD
        # of the binary spike, with sigma = soft_spike_sigma_frac * theta (theta =
        # the frozen LIF threshold). This changes ONLY the per-timestep feature
        # handed to the readout: the recurrent dynamics still use the HARD spike
        # (step() is untouched), so the reservoir trajectory, spike rate and every
        # frozen buffer are bit-identical to the hard-spike reservoir. Default
        # False reproduces the original behavior EXACTLY (the readout feature IS
        # the literal hard spike, via the same code path as before). See
        # PAPER.md Section 6.2: sigma=0.1*theta
        # is the empirically-best value (a robust inverted-U across sigma). NOTE:
        # set BEFORE the generator `g` is created so this never perturbs the frozen
        # weights' RNG stream -- same seed => byte-identical W_in/W_res either way.
        assert soft_spike_sigma_frac > 0, "soft_spike_sigma_frac must be > 0"
        self.soft_spike = soft_spike
        self.soft_spike_sigma_frac = soft_spike_sigma_frac

        g = torch.Generator().manual_seed(seed)

        # Input scale 0.3 (was 0.1): at 0.1, steady-state input-driven membrane std
        # is ~0.23 vs threshold=1.0, so the reservoir almost never fires (~1e-5 spike
        # rate, measured empirically) and carries no information. 0.3 is the smallest
        # scale (of 0.1/0.3/0.5/0.8/1.0 tried) that reliably lands in a healthy ~2%
        # mean spike rate across seeds without saturating. (Tuned for one-hot input;
        # the backprop path's trainable embedding is init-scaled to keep the induced
        # input current in this same band -- see models/spiking_backprop_lm.py.)
        W_in = torch.randn(reservoir_size, input_dim, generator=g) * 0.3
        self.register_buffer("W_in", W_in)

        if not use_tensor_train:
            # --- DENSE path: byte-for-byte identical to the original construction. ---
            W_res = torch.randn(reservoir_size, reservoir_size, generator=g)
            eigvals = torch.linalg.eigvals(W_res).abs()
            W_res = W_res * (spectral_radius / eigvals.max())
            self.register_buffer("W_res", W_res)
        else:
            # --- TT-native path: build W_res as random TT cores (no dense NxN). ---
            self._build_tt_cores(reservoir_size, spectral_radius, tt_rank, tt_n_cores,
                                 tt_core_std, g, tt_bond_decay)

        # beta/threshold not learnable -> snntorch registers them as buffers, not
        # nn.Parameter, so this reservoir has no trainable parameters at all.
        # Constructed in BOTH neuron models: it is the single source of the frozen
        # beta/threshold constants (`_soft_spike` and the resonate-and-fire cell
        # both read them off it), and it consumes no generator values, so its
        # presence cannot perturb anything.
        self.lif = snn.Leaky(beta=beta, learn_beta=False, learn_threshold=False)

        # --- resonate-and-fire frequencies: THE LAST DRAW, AND ONLY HERE -------- #
        if neuron_model == "rf":
            if not (0.0 < self.rf_period_min <= self.rf_period_max):
                raise ValueError(
                    f"need 0 < rf_period_min <= rf_period_max; got "
                    f"{rf_period_min!r}, {rf_period_max!r}")
            if self.rf_period_min < 2.0:
                raise ValueError(
                    f"rf_period_min={rf_period_min!r} is below the Nyquist period of "
                    "2 steps; nothing faster is representable in discrete time (§23.2)")
            # Log-uniform on [T_min, T_max]: uniform in log T, i.e. equal density per
            # octave. Drawn from the reservoir's own seeded generator so the
            # frequencies are reproducible from `seed` alone, like every other frozen
            # weight here.
            log_lo = math.log(self.rf_period_min)
            log_hi = math.log(self.rf_period_max)
            u = torch.rand(reservoir_size, generator=g)
            periods = torch.exp(log_lo + u * (log_hi - log_lo))
            omega = (2.0 * math.pi) / periods
            self.rf = ResonateFireCell(omega, beta=beta,
                                       threshold=self.lif.threshold)

        for p in self.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------ #
    # Tensor-train (MPO) construction and native matrix-vector product.
    # ------------------------------------------------------------------ #
    def _build_tt_cores(self, N, spectral_radius, tt_rank, tt_n_cores, tt_core_std, g,
                        tt_bond_decay=1.0):
        """Build W_res natively as a square TT-matrix (MPO): a chain of `d` cores
        C_k of shape (r_{k-1}, m_k, n_k, r_k) with r_0=r_d=1, m_k=n_k, prod(m_k)=N.
        The effective (never materialized) matrix element is the ordered product of
        the per-index core slices contracted over the bond dimensions.

        THE GEOMETRIC BOND PROFILE (`tt_bond_decay`, "lambda"), AND WHY IT IS THE ONLY
        KNOB HERE THAT CAN MOVE THE ENTANGLEMENT ENTROPY. Normalised entanglement
        entropy S-bar is computed (see `entanglement_entropy`) from the NORMALISED
        Schmidt spectrum p = sigma^2 / sum(sigma^2) of the mixed-canonical centre
        core. That normalisation makes S-bar exactly invariant to any GLOBAL
        rescaling of the cores, because every singular value picks up the same
        factor and it cancels. Both pre-existing knobs are exactly such a global
        rescaling: with tt_core_std=None, `spectral_radius` enters only through the
        derived scalar `s` below, and `tt_core_std` IS that scalar. Measured
        consequence (docs/EXPERIMENT_LOG.md Section 12, A5): multiplying every core by
        1000 moves S-bar by 2.8e-11 -- the dependence is not weak, it is zero. And
        i.i.d. Gaussian cores generically produce a near-FLAT Schmidt spectrum
        (measured 0.16881 ... 0.09403 against 0.125 for a perfectly uniform
        8-dimensional bond), which pins S-bar near its maximum of 1 regardless of
        `tt_rank` (A6: 0.96221-0.99596 over tt_rank in {4,8,16,32}).

        Lowering S-bar therefore requires a DECAYING Schmidt spectrum, which requires
        breaking the i.i.d. assumption ALONG THE BOND INDEX rather than rescaling it.
        `tt_bond_decay=lambda` does exactly that and nothing else: bond index r is
        multiplied by lambda^r.

        WHICH AXES ARE THE BOND AXES, stated explicitly because getting it wrong
        yields a wrong-but-plausible number. A core is (r_{k-1}, m_k, n_k, r_k):
        axis 0 is the LEFT bond, axes 1/2 are the physical row/column modes, axis 3
        is the RIGHT bond. This is the convention `_tt_matvec` contracts under
        ('bpans,amnc->bpmcs': a = left bond, c = right bond) and the one
        `entanglement_entropy` flattens under (rp, m*n, rk). The profile touches
        ONLY axes 0 and 3; the physical modes are untouched, so this changes the
        correlation structure of W_res, not its mode geometry.

        HOW A SHARED BOND IS HANDLED. Internal bond k is shared: it is core k-1's
        RIGHT axis and core k's LEFT axis. The profile is applied to EVERY bond axis
        of EVERY core, so both cores adjacent to a shared bond see the IDENTICAL
        weight vector lambda^r on it -- the construction is symmetric and no core
        "owns" a bond. The price of that symmetry, stated rather than buried: each
        internal bond index r is then suppressed TWICE, so the effective suppression
        of bond index r in the contracted matrix element is lambda^(2r), and the
        induced Schmidt values decay like lambda^(2r) (hence p like lambda^(4r)). A
        one-sided convention would be this same family reparametrised by
        lambda -> sqrt(lambda); it is not a different construction. The boundary
        bonds have r_0 = r_d = 1, so their only index is 0 and lambda^0 = 1: the
        profile is a no-op there by construction, never a hidden edge case.

        lambda = 1.0 IS A NO-OP, BIT FOR BIT, AND IS THE DEFAULT. lambda^r == 1.0
        exactly for every r, and IEEE-754 multiplication by 1.0 is exact, so the
        cores are bit-identical to the pre-change construction. The profile is
        applied AFTER the `torch.randn` draw and never consumes the generator, so
        the RNG stream -- draw order, shapes, and every downstream draw -- is
        untouched at ANY lambda: two reservoirs differing only in `tt_bond_decay`
        differ only by the deterministic per-entry factor. That is what makes a
        lambda sweep a controlled comparison rather than six unrelated reservoirs,
        and it is asserted in tests/test_structured_tt_cores.py. Default 1.0 for the
        same reason `--grad-clip-mode` defaults to `global` and `--embed-init-mode`
        to `legacy`: published results depend on the existing construction being
        reproducible bit-for-bit.

        NOT NORMALISED, DELIBERATELY. The profile shrinks the cores (every factor is
        <= 1), so it also shrinks the effective operator: this knob changes the
        spectrum's SHAPE and its SCALE at the same time. Restoring the scale is the
        pre-existing `tt_core_std` knob's job -- the two compose, and since S-bar is
        provably invariant to `tt_core_std` (above), composing them separates a
        genuine structural change from mere shrinkage.
        """
        lam = float(tt_bond_decay)
        if not (0.0 < lam <= 1.0):
            raise ValueError(
                f"tt_bond_decay must be in (0, 1]; got {tt_bond_decay!r}. 1.0 is the "
                "no-op (i.i.d. Gaussian cores); values > 1 would GROW the high bond "
                "indices, and <= 0 would zero or sign-flip them.")
        self.tt_bond_decay = lam
        modes = _balanced_factorization(N, tt_n_cores)
        d = len(modes)
        if d < 2:
            raise ValueError(
                f"reservoir_size={N} cannot be factored into >=2 TT modes (got "
                f"{modes}); TT construction needs a composite reservoir_size.")
        self.tt_modes = modes
        self.tt_n_cores = d
        ranks = [1] + [tt_rank] * (d - 1) + [1]
        self.tt_ranks = ranks

        # Per-entry variance of the effective W_res. For i.i.d. zero-mean cores with
        # std s, cross bond-paths cancel in E[W_ij^2] and only the R_int coinciding
        # paths survive, giving Var(W_ij) = R_int * s^(2d), where R_int is the product
        # of the internal bond dimensions. To roughly match a dense reservoir at the
        # requested spectral radius (dense circular law: radius ~ sigma*sqrt(N), i.e.
        # entry variance spectral_radius^2 / N), solve for s. This is a PROXY: a
        # TT-structured matrix has no cheap classical spectral radius, and its entries
        # are correlated, so the induced dynamical regime is verified empirically
        # (healthy spike rate) and via the TT-native entanglement-entropy diagnostic.
        R_int = 1
        for r in ranks[1:-1]:
            R_int *= r
        if tt_core_std is None:
            v = (spectral_radius ** 2) / N
            s = (v / R_int) ** (1.0 / (2 * d))
        else:
            s = float(tt_core_std)
        self.tt_core_std = s

        for k in range(d):
            m = modes[k]  # square matrix: m_k == n_k
            core = torch.randn(ranks[k], m, m, ranks[k + 1], generator=g) * s
            # Geometric bond profile: axis 0 (left bond) and axis 3 (right bond) only.
            # At lam == 1.0 both factors are exactly 1.0 and this is a bit-exact no-op.
            left = lam ** torch.arange(ranks[k], dtype=core.dtype).view(-1, 1, 1, 1)
            right = lam ** torch.arange(ranks[k + 1], dtype=core.dtype).view(1, 1, 1, -1)
            core = core * left * right
            self.register_buffer(f"tt_core_{k}", core)

    def _tt_cores(self):
        """Fetch the TT cores as a list (re-read from buffers each call so device
        moves via .to()/.cuda() are respected)."""
        return [getattr(self, f"tt_core_{k}") for k in range(self.tt_n_cores)]

    def _tt_matvec(self, x):
        """Batched y = W_res @ x with W_res in TT form, WITHOUT ever forming the
        dense NxN matrix. x: (B, N) -> returns (B, N).

        Left-to-right sweep. Working tensor layout (B, P, r_prev, R_rest):
          P      = product of output-mode sizes already produced (m_0..m_{k-1}),
          r_prev = current left bond dimension r_{k-1},
          R_rest = product of input-mode sizes not yet contracted (n_k..n_{d-1}).
        Each step splits the leading input mode n_k out of R_rest and contracts it
        together with r_prev against core C_k, emitting output mode m_k and bond r_k.
        This reproduces (W_res @ x) exactly (cross-checked against a materialized
        dense reconstruction in the tests). Index convention: the first mode consumed
        is the OUTERMOST (slowest-varying) index of x, matching _tt_to_dense."""
        B, Nx = x.shape
        t = x.reshape(B, 1, 1, Nx)  # P=1, r_prev=1, R_rest=N
        for C in self._tt_cores():
            _, m_k, n_k, r_k = C.shape
            Bc, P, r_prev, R = t.shape
            t = t.reshape(Bc, P, r_prev, n_k, R // n_k)      # split leading input mode
            # contract left bond (a=r_prev) and input mode (n) with the core:
            #   out[b,P,m,r_k,R'] = sum_{a,n} t[b,P,a,n,R'] * C[a,m,n,r_k]
            t = torch.einsum('bpans,amnc->bpmcs', t, C)      # (B, P, m_k, r_k, R')
            t = t.reshape(Bc, P * m_k, r_k, R // n_k)        # merge P,m_k; new bond r_k
        return t.reshape(B, -1)                              # (B, N); trailing bond=1

    def _tt_to_dense(self):
        """Materialize the full effective W_res by contracting the TT cores. FOR
        VERIFICATION / DIAGNOSTICS ONLY -- deliberately never used in step/forward,
        since avoiding this NxN materialization is the entire point of TT storage."""
        cores = self._tt_cores()
        _, m0, n0, r1 = cores[0].shape
        W = cores[0].reshape(m0, n0, r1)   # (rows=m0, cols=n0, bond=r1)
        Mrows, Ncols = m0, n0
        for C in cores[1:]:
            _, mk, nk, rk = C.shape
            # (Mrows, Ncols, r) x (r, mk, nk, rk) -> (Mrows, mk, Ncols, nk, rk),
            # ordering row-modes before col-modes so the final reshape is a proper
            # (row multi-index, col multi-index) matrix.
            W = torch.einsum('ijp,pklc->ikjlc', W, C)
            Mrows *= mk
            Ncols *= nk
            W = W.reshape(Mrows, Ncols, rk)
        return W.reshape(Mrows, Ncols)

    def tt_param_count(self):
        """Total number of stored TT-core scalars for W_res (the compressed
        recurrent matrix). Compare against reservoir_size**2 for the dense form."""
        return sum(c.numel() for c in self._tt_cores())

    def entanglement_entropy(self, bond=None):
        """Normalized entanglement entropy across one TT bond (a cheap, TT-native
        indicator of the reservoir's dynamical regime; Sato et al. 2025 report a
        productive band S_bar in [0.1, 0.5] for randomized TT reservoirs -- though
        that was validated on rate-based regression, NOT a spiking reservoir, so this
        is a diagnostic to log, not a hard gate).

        Method: bring the TT to mixed-canonical form with its orthogonality center on
        core `bond` (left-orthonormalize cores 0..bond-1 via QR, right-orthonormalize
        cores bond+1..d-1 via LQ), then the singular values of the center core across
        the left bond are the Schmidt coefficients of that cut. Returns
        S / log(bond_dim) in [0, 1]. Works directly on the small cores -- the dense
        matrix is never formed."""
        assert self.use_tensor_train, "entanglement_entropy requires a TT reservoir"
        d = self.tt_n_cores
        if bond is None:
            bond = d // 2
        assert 1 <= bond <= d - 1, f"bond must be in [1, {d - 1}]"
        # flatten each matrix core (r_{k-1}, m, n, r_k) to a 3D TT-vector-style core
        # (r_{k-1}, m*n, r_k); double precision for a stable SVD.
        cores = []
        for C in self._tt_cores():
            rp, m, n, rk = C.shape
            cores.append(C.reshape(rp, m * n, rk).clone().double())
        # left-orthonormalize cores 0..bond-1, absorbing R into the next core
        for k in range(bond):
            rp, phys, rk = cores[k].shape
            Q, R = torch.linalg.qr(cores[k].reshape(rp * phys, rk), mode='reduced')
            cores[k] = Q.reshape(rp, phys, Q.shape[1])
            cores[k + 1] = torch.einsum('ij,jpk->ipk', R, cores[k + 1])
        # right-orthonormalize cores d-1..bond+1, absorbing L into the previous core
        for k in range(d - 1, bond, -1):
            rp, phys, rk = cores[k].shape
            Qt, Rt = torch.linalg.qr(cores[k].reshape(rp, phys * rk).transpose(0, 1),
                                     mode='reduced')
            L = Rt.transpose(0, 1)
            cores[k] = Qt.transpose(0, 1).reshape(L.shape[1], phys, rk)
            cores[k - 1] = torch.einsum('ipj,jk->ipk', cores[k - 1], L)
        # center core holds the entanglement; SVD across its left bond
        center = cores[bond]
        rp, phys, rk = center.shape
        sv = torch.linalg.svdvals(center.reshape(rp, phys * rk))
        sv = sv[sv > 1e-12]
        if rp <= 1 or sv.numel() <= 1:
            return 0.0
        p = (sv ** 2) / (sv ** 2).sum()
        S = -(p * torch.log(p)).sum().item()
        return S / math.log(rp)  # normalize by the bond dimension -> [0, 1]

    def _soft_spike(self, mem):
        """Graded Gaussian-CDF "soft spike" Phi((mem - theta)/sigma) of the LIF
        membrane potential -- the closed form the frozen stochastic-resonance probe
        found beats the hard binary spike at sigma=0.1*theta. Uses the erf identity
        Phi(z) = 0.5*(1 + erf(z/sqrt(2))): numerically identical to the probe's
        torch.distributions.Normal(0,1).cdf(...) (which is implemented via the same
        erf formula) but without allocating a distribution object in the
        per-timestep hot loop. Differentiable in `mem` (in fact smoother than the
        surrogate-gradient hard spike), so a readout can be trained on it directly."""
        theta = self.lif.threshold  # frozen LIF threshold buffer; moves with .to()
        sigma = self.soft_spike_sigma_frac * theta
        return 0.5 * (1.0 + torch.erf((mem - theta) / (sigma * math.sqrt(2.0))))

    def readout_feature(self, spk, mem):
        """The per-timestep feature this reservoir exposes to a downstream readout:
        the hard binary spike by default, or the graded soft spike when soft_spike
        is on. Centralized so forward() AND the incremental generation loop feed the
        readout the SAME feature -- a soft-trained model must be READ with the soft
        feature at generation time too."""
        return self._soft_spike(mem) if self.soft_spike else spk

    @property
    def omega(self):
        """The frozen per-unit resonant frequencies (rf mode only).

        Read through to `self.rf`'s buffer rather than kept as a second copy on
        this module: two `register_buffer` entries holding the same tensor stop
        being the same tensor the first time `.to()`/`.cuda()` is called (nn.Module
        re-materializes buffers per module), which would leave a stale `omega`
        that the frozen-weight tripwire still happily checks. One buffer, one
        owner -- `named_buffers()` reports it as `rf.omega`, and the tripwire
        covers it there.

        A LIF reservoir raises rather than returning a tensor of zeros: zeros are a
        valid omega (they are exactly the LIF point of the family), so returning
        them would let rf-specific analysis run silently against the control arm and
        report plausible numbers. NOTE that nn.Module's own `__getattr__` catches
        the AttributeError raised below and re-raises its generic "object has no
        attribute 'omega'", so the message here is not what a caller sees -- the
        exception TYPE is the contract.
        """
        if self.neuron_model != "rf":
            raise AttributeError(
                f"omega exists only for neuron_model='rf' (this reservoir is "
                f"{self.neuron_model!r})")
        return self.rf.omega

    def ac_gain(self):
        """Accumulation gain for a zero-mean fluctuating input, 1/sqrt(1-beta^2).

        Identical in both neuron models BY CONSTRUCTION: it depends only on the pole
        MAGNITUDE |lambda| = beta, which §23.2 holds fixed at the LIF path's own
        value, and not on the rotation. Exposed as a method so §23.5's G0a ratio is
        computed from the reservoir under test in both arms rather than from a
        constant typed into the analysis twice."""
        beta = float(self.lif.beta.clamp(0, 1))
        return 1.0 / math.sqrt(1.0 - beta * beta)

    def dc_gain(self):
        """Per-unit steady-state gain to a CONSTANT input: |1/(1 - beta*e^{i*w})|.

        This is the quantity §23.3's H10 turns on. A LIF unit (w == 0) integrates a
        constant input to 1/(1-beta) = 10.0 at beta = 0.9 while accumulating a
        zero-mean one to only 1/sqrt(1-beta^2) = 2.2942 -- a 4.36x amplification of
        the component that carries no information, which is the measured defect
        §23.1 tabulates (induced membrane offset ending at 4.2x the threshold).
        Rotating the pole collapses the first and leaves the second exactly alone,
        and because the pole is FROZEN that attenuation cannot decay as the
        embedding trains, which a bias initialisation demonstrably did.

        Returns a (reservoir_size,) tensor in BOTH modes -- a full tensor of
        1/(1-beta) for LIF -- so the G0a gate and the analysis are one code path,
        not two. Written from (1 - beta*cos w)^2 + (beta*sin w)^2 rather than via
        complex tensors: identical value, no complex dtype leaking into a
        diagnostic that callers will want to reduce and log."""
        beta = float(self.lif.beta.clamp(0, 1))
        if self.neuron_model != "rf":
            return torch.full((self.reservoir_size,), 1.0 / (1.0 - beta),
                              device=self.W_in.device, dtype=self.W_in.dtype)
        w = self.rf.omega
        re = 1.0 - beta * torch.cos(w)
        im = beta * torch.sin(w)
        return 1.0 / torch.sqrt(re * re + im * im)

    def step(self, byte_onehot_step, mem, spk, imem=None):
        """Advance the reservoir exactly ONE timestep.

        byte_onehot_step: (B, vocab_size); mem, spk, imem: (B, reservoir_size).
        Returns (spk_next, mem_next, imem_next). This is the single per-timestep
        body shared with `forward`, exposed so callers can cache state and feed one
        new byte at a time -- turning the O(n^2) "re-run the whole growing sequence"
        generation loop into an O(n) incremental loop with bit-identical output.

        `imem` is the resonate-and-fire quadrature companion (`v` in §23.2; `mem`
        is `u`). In LIF mode it is inert PASS-THROUGH -- returned exactly as handed
        in, or zeros if omitted -- and the LIF arithmetic below never reads it, so
        the LIF path is bit-identical to the v2 code with or without it. It is
        threaded unconditionally, rather than only in rf mode, so callers never
        branch on the neuron model to know the shape of the state they are
        carrying: a state tuple whose ARITY depends on a flag is the kind of thing
        that runs fine and silently drops a component.
        """
        input_current = byte_onehot_step @ self.W_in.T
        if self.use_tensor_train:
            # TT-native recurrent drive: contract cores against spk directly.
            # Equivalent to (spk @ W_res.T) but W_res is never materialized.
            recurrent_current = self._tt_matvec(spk)
        else:
            recurrent_current = spk @ self.W_res.T
        current = input_current + recurrent_current
        if self.neuron_model == "rf":
            if imem is None:
                imem = torch.zeros_like(mem)
            return self.rf(current, mem, imem)               # (spk, u, v)
        spk_next, mem_next = self.lif(current, mem)
        if imem is None:
            imem = torch.zeros_like(mem_next)
        return spk_next, mem_next, imem

    def forward(self, byte_onehot_seq):
        # byte_onehot_seq: (B, T, vocab_size)
        B, T, _ = byte_onehot_seq.shape
        device = byte_onehot_seq.device
        mem = torch.zeros(B, self.reservoir_size, device=device)
        imem = torch.zeros(B, self.reservoir_size, device=device)
        spk = torch.zeros(B, self.reservoir_size, device=device)
        feats_out = []
        for t in range(T):
            spk, mem, imem = self.step(byte_onehot_seq[:, t, :], mem, spk, imem)
            # readout_feature is `spk` verbatim in the default (hard-spike) mode --
            # byte-for-byte the original loop -- or the graded soft spike when on.
            feats_out.append(self.readout_feature(spk, mem))
        return torch.stack(feats_out, dim=1)  # (B, T, reservoir_size)

    # ------------------------------------------------------------------ #
    # Parallel / associative-scan reformulation (research artifact).
    #
    # NEGATIVE RESULT -- read before using. The exact LIF (subtract reset,
    # reset_delay=True) recurrence for a SINGLE neuron is affine BETWEEN its own
    # resets:  mem[t] = beta*mem[t-1] + input[t] , so an associative prefix scan
    # can in principle replace the T sequential steps by O(log T) parallel rounds
    # (arXiv:2306.12666, arXiv:2603.13283). The implementation below does exactly
    # that, correctly and bit-exactly, via speculative "earliest-crossing"
    # segmentation.
    #
    # BUT it does NOT speed up THIS reservoir, and cannot, for two compounding
    # reasons (both measured, see PAPER.md Section 5.2.2):
    #   1. All-to-all recurrent coupling (W_res, spectral radius 1.0): a spike in
    #      ANY neuron injects current into ALL neurons next step, so a segment is
    #      "linear" only while the WHOLE reservoir is silent.
    #   2. Union firing rate: with N neurons each firing ~2.2%, the probability
    #      that *some* neuron crosses on a given timestep is ~1 for N in the
    #      hundreds/thousands. Measured: at N=2048 every one of 200 timesteps has a
    #      crossing, so each "segment" is length 1 and the scan degenerates to the
    #      sequential loop plus overhead (~6x slower).
    # The irreducible cost is the T sequential recurrent matmuls spk @ W_res.T
    # (2048x2048, ~64% of runtime), which cannot be parallelized across time
    # because each depends on the previous step's full nonlinear coupled state.
    # For the actual user-facing latency (generation) the correct fix is the
    # O(n) incremental `step` loop, not this scan. Kept for reference/verification.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _affine_scan(A, B):
        """Inclusive Hillis-Steele parallel prefix scan of affine maps x -> a*x + b
        along dim=1 (time). A, B: (batch, L, N). Composition of two maps applied
        earlier-then-later is (a2*a1, a2*b1 + b2). Returns composed (A_pref, B_pref)
        so that state[j] = A_pref[j] * state_in + B_pref[j]."""
        A = A.clone(); B = B.clone()
        L = A.shape[1]
        d = 1
        while d < L:
            # element j (>= d) composes with element j-d: (a_j*a_{j-d}, a_j*b_{j-d}+b_j).
            # Read both new values from the OLD A, B, then commit -- so the b-update
            # sees the pre-update a-multiplier, matching sequential composition order.
            newA = A.clone(); newB = B.clone()
            newA[:, d:, :] = A[:, d:, :] * A[:, :-d, :]
            newB[:, d:, :] = A[:, d:, :] * B[:, :-d, :] + B[:, d:, :]
            A, B = newA, newB
            d *= 2
        return A, B

    def forward_parallel(self, byte_onehot_seq):
        """Associative-scan reformulation of `forward`, bit-exact but NOT faster on
        the coupled reservoir (see class-level note). Uses speculative earliest-
        crossing segmentation: within each segment assume no neuron spikes, compute
        the linear membrane trajectory with a parallel affine prefix scan, find the
        earliest timestep any neuron crosses threshold (that prefix is then exact by
        the exact reset semantics -- reset_delay=True means a spike affects mem only
        one step later), resolve that single spike, and continue from there."""
        if self.neuron_model != "lif":
            # The affine-scan segmentation is LIF-SPECIFIC: it composes the scalar
            # map mem -> beta*mem + input, which is exactly what a resonate-and-fire
            # unit is NOT -- its between-reset map is a 2x2 rotation-scaling on
            # (u, v), so a scan over scalar (a, b) pairs would silently drop the
            # quadrature component and return plausible, WRONG numbers. Refused
            # rather than approximated: this is a negative-result research artifact
            # (see the block comment above) and not worth generalising.
            raise NotImplementedError(
                f"forward_parallel is a LIF-only research artifact; this reservoir is "
                f"neuron_model={self.neuron_model!r}, whose between-reset dynamics are "
                "a 2x2 rotation on (u, v) rather than a scalar affine map. Use "
                "forward() or step().")
        if self.use_tensor_train:
            # Dense-only reference path: it indexes self.W_res directly, which a TT
            # reservoir does not materialize. Use forward()/step() for TT reservoirs.
            raise NotImplementedError(
                "forward_parallel is a dense-only research artifact; a TT reservoir "
                "has no materialized W_res. Use forward() or step() instead.")
        Bb, T, _ = byte_onehot_seq.shape
        device = byte_onehot_seq.device
        N = self.reservoir_size
        beta = float(self.lif.beta.clamp(0, 1))
        thr = float(self.lif.threshold)
        input_ext = byte_onehot_seq @ self.W_in.T          # (B, T, N) external drive
        W_res_T = self.W_res.T
        out = torch.zeros(Bb, T, N, device=device)
        mem_prev = torch.zeros(Bb, N, device=device)
        spk_prev = torch.zeros(Bb, N, device=device)
        t0 = 0
        while t0 < T:
            # Additive per-step term under the "no new spikes in this segment"
            # speculation. Only the first step of the segment carries the coupling
            # + reset from the spike that ended the previous segment.
            c = input_ext[:, t0:, :].clone()
            c[:, 0, :] = c[:, 0, :] + spk_prev @ W_res_T - spk_prev * thr
            A = torch.full_like(c, beta)
            A_pref, B_pref = self._affine_scan(A, c)
            mem_seg = A_pref * mem_prev.unsqueeze(1) + B_pref   # (B, L, N)
            crossed = (mem_seg > thr).any(dim=2).any(dim=0)     # (L,) any batch/neuron
            idx = torch.nonzero(crossed, as_tuple=False)
            if idx.numel() == 0:
                break                                           # no spikes to the end
            rel = int(idx[0])
            tstar = t0 + rel
            spk_star = (mem_seg[:, rel, :] > thr).float()
            out[:, tstar, :] = spk_star
            mem_prev = mem_seg[:, rel, :]
            spk_prev = spk_star
            t0 = tstar + 1
        return out
