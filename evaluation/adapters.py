"""
Per-model adapters: give every model ONE interface

    adapter.predict(x1bit_np)  ->  (khat [N] int,  doa_pred list[N] of angle arrays)

so the harness treats Trans / TCN / DA (direct DoA regression) and LCPSN
(spectrum peak-picking) uniformly.  DoA is always decoded with K_hat (the
agreed "K-hat decoding" metric).

Pass random_init=True only for plumbing tests. Paper evaluation requires every
selected checkpoint; run_eval.py reports an error if any are missing.
"""
import os
import sys
from pathlib import Path
import math
import numpy as np
import torch
from decoder import separated_topk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_TCN = PROJECT_ROOT / "baselines" / "TCN_MUSIC"
REPO_LCPSN = PROJECT_ROOT / "lcpsn"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"

# STEERING grid -- matches lcpsn/train.py:precompute_steering_vectors, i.e. the
# grid the Bartlett prior was computed on during training. Endpoint-exclusive,
# spacing 180/G = 0.50000 deg. Also used by the classical MUSIC baselines, which
# have no trained bin->angle mapping and so are indifferent to the choice.
ANGLES = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)

# DECODING grid for LCPSN -- matches lcpsn/criterion.py:build_angle_grid_rad,
# i.e. the grid the TARGET spectrum was built on during training.
# Endpoint-inclusive, spacing 180/(G-1) = 0.50139 deg.
#
# These two grids are NOT interchangeable. LCPSN's output bin g was trained to
# fire when a source sits at ANGLES_DECODE[g]; decoding that bin as ANGLES[g]
# mislabels it by (g/(G-1) - g/G) * 180 deg, which grows monotonically with the
# angle and shows up as a systematic bias of about -0.21 deg (mean over
# |theta| < 80 deg, up to -0.5 deg at endfire). Measured on
# checkpoints/lcpsn/best_lcpsn_sep8_uncovON_seed0.pt: mean signed error
# -0.206 deg with ANGLES vs +0.028 deg with ANGLES_DECODE.
#
# Set LCPSN_DECODE_GRID=exclusive to reproduce the pre-fix behaviour.
ANGLES_DECODE = np.linspace(-np.pi / 2, np.pi / 2, 360)
if os.environ.get("LCPSN_DECODE_GRID", "inclusive").lower() == "exclusive":
    ANGLES_DECODE = ANGLES.copy()

K_MIN, K_MAX = 2, 5


# ----------------------------------------------------------------------------- 3-model family
class ThreeModelAdapter:
    """Trans / TCN / DA — regressor returns (doa[b,6], feat); classifier -> K."""

    def __init__(self, name, reg_ctor, clf_ctor, reg_ckpt, clf_ckpt,
                 device="cuda", random_init=False):
        self.name = name
        self.device = device
        if str(REPO_TCN) not in sys.path:
            sys.path.insert(0, str(REPO_TCN))
        self.reg = reg_ctor(device=torch.device(device)).to(device)
        self.clf = clf_ctor(self.reg, device).to(device)
        self.available = True
        rp = CHECKPOINT_ROOT / "baselines" / "sep8" / reg_ckpt
        cp = CHECKPOINT_ROOT / "baselines" / "sep8" / clf_ckpt
        if random_init:
            pass                                    # keep random weights (smoke test)
        elif rp.exists() and cp.exists():
            self.reg.load_state_dict(torch.load(rp, weights_only=True, map_location=device))
            self.clf.load_state_dict(torch.load(cp, weights_only=True, map_location=device))
        else:
            self.available = False
        self.reg.eval()
        self.clf.eval()

    @torch.no_grad()
    def predict(self, x1bit):
        x = torch.from_numpy(x1bit).float().to(self.device)   # already +-1/sqrt2
        doa, feat = self.reg(x)                                 # (b,6),(b,128)
        logit = self.clf.classifier(feat)                       # (b,4)
        khat = (torch.argmax(logit, 1) + K_MIN).cpu().numpy().astype(np.int64)
        doa = doa.cpu().numpy()
        doa_pred = [np.sort(doa[i, :khat[i]]) for i in range(len(khat))]
        return khat, doa_pred


# ----------------------------------------------------------------------------- LCPSN
class LCPSNAdapter:
    """LCPSN — spectrum mu[b,360] + K_logits[b,4]; DoA = top-Khat separated peaks."""

    def __init__(self, name, ckpt, device="cuda", random_init=False,
                 min_sep_deg=2.0, model_kwargs=None):
        # decode floor = spectral resolution (~training sigma), NOT the test min-sep:
        # condition-agnostic and fair (3-model regressors impose no separation floor).
        self.name = name
        self.device = device
        self.min_sep_deg = min_sep_deg
        if str(REPO_LCPSN) not in sys.path:
            sys.path.insert(0, str(REPO_LCPSN))
        from model import LCPSN
        from physics import ULA_action_vector

        kw = dict(m=8, snapshots=200, angles_count=360, d_model=96, nhead=8,
                  num_layers=4, ff_dim=256, dropout=0.1, spectrum_hidden=512,
                  k_min=K_MIN, k_max=K_MAX, variance_floor=5e-2)
        if model_kwargs:
            kw.update(model_kwargs)
        self.model = LCPSN(**kw).to(device)
        self.available = True
        if random_init:
            pass
        elif Path(ckpt).exists():
            state = torch.load(ckpt, map_location=device)
            self.model.load_state_dict(state, strict=False)
        else:
            self.available = False
        self.model.eval()

        # steering vectors on the training steering grid; peaks decoded on the
        # training TARGET grid -- see the ANGLES / ANGLES_DECODE note at the top.
        a = np.stack([ULA_action_vector(t, 8) for t in ANGLES], axis=1)   # [M,360]
        self.a_steering = torch.from_numpy(a).to(torch.complex64).to(device)
        self.grid = ANGLES_DECODE.copy()

    def _peaks(self, score, khat):
        """Common top-khat separated decoder used by LCPSN and MUSIC."""
        return separated_topk(score, khat, self.grid, min_sep_deg=self.min_sep_deg)

    @torch.no_grad()
    def predict(self, x1bit):
        x = torch.from_numpy(np.sign(x1bit)).float().to(self.device)   # LCPSN wants +-1
        out = self.model(x, self.a_steering)
        mu = out["mu"].cpu().numpy()
        khat = (torch.argmax(out["K_logits"], 1) + K_MIN).cpu().numpy().astype(np.int64)
        khat = np.clip(khat, K_MIN, K_MAX)
        doa_pred = [self._peaks(mu[i], int(khat[i])) for i in range(len(khat))]
        return khat, doa_pred


# ----------------------------------------------------------------------------- classical MUSIC
class MUSICAdapter:
    """Classical MUSIC — (covariance front-end) x (source-enumeration rule).

    No checkpoint, so `.available` is always True. Two extras are requested from
    the harness through `.wants`:
      * "ktrue" — oracle-K rows only (the true count, not an estimate).
      * "analog" — full-precision rows only (the unquantized snapshots).
      * "ctx"   — per-batch scratch dict. Nine MUSIC rows share three covariance
                  front-ends, so the covariance and its eigen-decomposition are
                  computed once per batch and reused by every rule built on it.
    Decoding (grid, peak rule, 2 deg floor) is identical to LCPSNAdapter.
    """

    def __init__(self, name, cov_kind, enum_rule, min_sep_deg=2.0, m=8):
        import music_baselines as mb
        self.mb = mb
        self.name = name
        self.cov_kind = cov_kind
        self.enum_rule = enum_rule
        self.min_sep_deg = min_sep_deg
        self.available = True
        # MUSIC has no trained bin->angle mapping, so its grid choice is free;
        # we sample it on ANGLES_DECODE so that decoding is literally identical
        # to LCPSNAdapter and a row differs from LCPSN only in the estimator.
        self.a_grid = mb.steering_grid(m, ANGLES_DECODE)
        wants = ["ctx"]
        if enum_rule == "oracle":
            wants.append("ktrue")
        if cov_kind == "analog":
            wants.append("analog")
        self.wants = tuple(wants)

    def _eig(self, ctx, x1bit, analog):
        """Covariance + eigen-decomposition, memoized per batch in `ctx`."""
        key = f"eig_{self.cov_kind}"
        if ctx is None:
            R = self.mb.covariance(self.cov_kind, x1bit=x1bit, analog=analog)
            return self.mb.eigh_desc(R)
        if key not in ctx:
            R = self.mb.covariance(self.cov_kind, x1bit=x1bit, analog=analog)
            ctx[key] = self.mb.eigh_desc(R)
        return ctx[key]

    def predict(self, x1bit, analog=None, ktrue=None, ctx=None):
        vals, vecs = self._eig(ctx, x1bit, analog)
        snapshots = x1bit.shape[-1]
        if self.enum_rule == "oracle":
            khat = np.clip(np.asarray(ktrue, dtype=np.int64), K_MIN, K_MAX)
        else:
            khat = self.mb.info_criterion_k(vals, snapshots, self.enum_rule)
        spec = self.mb.music_spectrum(vecs, khat, self.a_grid)
        doa = [self.mb.top_k_peaks(spec[i], int(khat[i]), grid=ANGLES_DECODE,
                                   min_sep_deg=self.min_sep_deg)
               for i in range(len(khat))]
        return khat, doa


MUSIC_SPECS = [
    # (family, covariance front-end, enumeration rule)
    ("1bMUSIC-naive",   "naive",   "mdl"),
    ("1bMUSIC-naive",   "naive",   "aic"),
    ("1bMUSIC-naive",   "naive",   "oracle"),
    ("1bMUSIC-arcsine", "arcsine", "mdl"),
    ("1bMUSIC-arcsine", "arcsine", "aic"),
    ("1bMUSIC-arcsine", "arcsine", "oracle"),
    ("FP-MUSIC",        "analog",  "mdl"),
    ("FP-MUSIC",        "analog",  "aic"),
    ("FP-MUSIC",        "analog",  "oracle"),
]


def build_music(device=None, min_sep_deg=2.0, **_unused):
    """The classical baseline block: 3 covariance front-ends x 3 enumeration rules.

    `device` is accepted and ignored — these run on the CPU in numpy.
    """
    adapters = []
    for family, cov_kind, rule in MUSIC_SPECS:
        a = MUSICAdapter(f"{family}/{rule}", cov_kind, rule, min_sep_deg=min_sep_deg)
        a.family, a.seed = f"{family}/{rule}", None
        adapters.append(a)
    return adapters


# ----------------------------------------------------------------------------- registries
THREE_SPECS = None          # filled in by _three_specs() to keep imports lazy


def _three_specs():
    """(display name, regressor class, classifier class, checkpoint basename)."""
    if str(REPO_TCN) not in sys.path:
            sys.path.insert(0, str(REPO_TCN))
    from Estimators.Trans_MUSIC.transformer_music_model import TransMUSICModel
    from Estimators.Trans_MUSIC.transformer_music_classifier_model import TransMUSICClassifierModel
    from Estimators.DA_MUSIC.da_music_model import DeepAugmentedMUSICModel
    from Estimators.DA_MUSIC.da_music_classifier_model import DAMUSICClassifierModel
    from Estimators.TCN_MUSIC.tcn_music_model import TemporalConvolutionalNetworkMUSICModel
    from Estimators.TCN_MUSIC.tcn_music_classifier_model import TemporalConvolutionalNetworkMUSICClassifierModel
    return [
        ("Trans-MUSIC", TransMUSICModel, TransMUSICClassifierModel, "one_bit_trans_music"),
        ("TCN-MUSIC", TemporalConvolutionalNetworkMUSICModel,
         TemporalConvolutionalNetworkMUSICClassifierModel, "one_bit_tcn_music"),
        ("DA-MUSIC", DeepAugmentedMUSICModel, DAMUSICClassifierModel, "one_bit_da_music"),
    ]


def build_sep8(device="cuda", seeds=(0, 1, 2), random_init=False,
               lcpsn_variants=("uncovON", "uncovOFF"), lcpsn_dir=None,
               include_three=True):
    """Adapters for the training-matched 8-deg study.

    Trans / TCN / DA come from TCN_MUSIC/saved_models/sep8 (one checkpoint per
    training seed); LCPSN comes from lcpsn/sep8 (one per variant x seed), where
    uncovON/uncovOFF is the --lambda-uncov 0.001 / 0 ablation.

    Every adapter carries `.family` (the curve it belongs to) and `.seed`, so the
    caller can average over seeds; `.name` stays unique for logging.
    """
    lcpsn_dir = Path(lcpsn_dir) if lcpsn_dir else CHECKPOINT_ROOT / "lcpsn"
    adapters = []

    if include_three:
        for family, reg_ctor, clf_ctor, base in _three_specs():
            for s in seeds:
                a = ThreeModelAdapter(
                    f"{family}/seed{s}", reg_ctor, clf_ctor,
                    f"{base}_sep8_seed{s}.pt",
                    f"{base}_classifier_sep8_seed{s}.pt",
                    device, random_init)
                a.family, a.seed = family, s
                adapters.append(a)

    for variant in lcpsn_variants:
        for s in seeds:
            family = f"LCPSN-{variant}"
            ckpt = lcpsn_dir / f"best_lcpsn_sep8_{variant}_seed{s}.pt"
            a = LCPSNAdapter(f"{family}/seed{s}", ckpt, device, random_init)
            a.family, a.seed = family, s
            adapters.append(a)

    return adapters


