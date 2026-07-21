"""DSpark-OPD loss configuration (Step-1 decoupling).

Parses the flat `dspark_loss_*` override_config keys into a typed, validated config that
`compose_dspark_loss` (losses.py) consumes. No verl deps beyond torch-free stdlib, so it
can be built + validated in isolation.

NOTE: loss knobs live as FLAT scalar override_config keys (dspark_loss_reverse_kl_weight,
dspark_loss_confidence_weight, ...) rather than a nested dict — verl's update_model_config
recurses into any dict-valued override_config key as an HF sub-config, so a nested block
crashes with AttributeError on the model config.

Design: docs/opd/loss-refactor-design.md §7. Decoupling replaces the reward→advantage→PPO
path with self-contained differentiable loss operators; this config drives which operators
run (`terms`) and how their per-token losses are aggregated. Legacy flat keys
(confidence_head_alpha / loss_decay_gamma) are read as fallbacks so old yamls keep working.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The only reverse-KL implementation shipped in Step-1 (design decision D1). The k3 /
# reinforce variants are intentionally NOT registered (D2); the registry interface is left
# open so adding them later is just "register one more function".
_SUPPORTED_REVERSE_KL_MODES = ("topk_pathwise",)
# Operator names that `terms` may reference (must match losses.py DSPARK_LOSS_REGISTRY keys).
# reject_kl  = Draft-OPD dual-stream reject-position term (T5): reverse-KL math on the reject slots.
# forward_kl = accept-stream forward KL (T5, §4b): mass-covering, teacher top-K support/weight.
# reverse_kl covers the accept stream when forward_kl is off (default single-stream / cache path).
_KNOWN_TERMS = ("reverse_kl", "forward_kl", "reject_kl", "confidence")


@dataclass
class DSparkOPDLossConfig:
    """Typed OPD loss config.

    terms: {operator_name: weight}. An operator runs iff its weight != 0. Order-independent
        (compose sums weighted per-term losses). Default reproduces the pre-decoupling loss:
        reverse_kl weight 1.0 + confidence weight = confidence_head_alpha.
    reverse_kl_mode: reverse-KL operator variant. Only "topk_pathwise" is implemented (D1/D2).
    loss_decay_gamma: block-internal exp(-pos/gamma) decay (SFT-consistent, applied ONCE in
        compose via decay_mask). None/<=0 disables decay.
    loss_agg_mode: aggregation passed to verl agg_loss ("token-mean" default).
    loss_max_clamp: optional symmetric per-token loss clamp (None = no clamp).
    """

    terms: dict[str, float] = field(default_factory=lambda: {"reverse_kl": 1.0, "confidence": 1.0})
    reverse_kl_mode: str = "topk_pathwise"
    loss_decay_gamma: float | None = 4.0
    loss_agg_mode: str = "token-mean"
    loss_max_clamp: float | None = None

    def __post_init__(self):
        if self.reverse_kl_mode not in _SUPPORTED_REVERSE_KL_MODES:
            raise NotImplementedError(
                f"reverse_kl_mode={self.reverse_kl_mode!r} is not implemented. "
                f"Step-1 supports only {_SUPPORTED_REVERSE_KL_MODES} (design D2). "
                "k3 / reinforce are deferred."
            )
        for name, weight in self.terms.items():
            if name not in _KNOWN_TERMS:
                raise ValueError(
                    f"Unknown loss term {name!r} in terms; known terms: {_KNOWN_TERMS}."
                )
            if float(weight) < 0:
                raise ValueError(f"loss term weight for {name!r} must be non-negative, got {weight}.")
        if not any(float(w) != 0 for w in self.terms.values()):
            raise ValueError("At least one loss term must have a non-zero weight.")

    def enabled_terms(self) -> list[tuple[str, float]]:
        """(name, weight) pairs with non-zero weight, in a stable order (reverse_kl before confidence)."""
        order = {name: i for i, name in enumerate(_KNOWN_TERMS)}
        items = [(n, float(w)) for n, w in self.terms.items() if float(w) != 0]
        return sorted(items, key=lambda kv: order.get(kv[0], len(order)))

    @classmethod
    def from_override_config(cls, override_config: dict | None) -> "DSparkOPDLossConfig":
        """Build from an actor override_config dict.

        Reads FLAT `dspark_loss_*` scalar keys (NOT a nested dict — verl's update_model_config
        recurses into dict-valued override_config keys as HF sub-configs and would crash; see the
        yaml comment). Falls back to legacy flat keys (confidence_head_alpha, loss_decay_gamma) so
        pre-decoupling yamls train identically. `reward_weight_mode` is intentionally ignored under
        topk_pathwise (the weight is softmax_K(S_grad), part of the KL — see design §4).
        """
        oc = dict(override_config or {})

        # term weights: new dspark_loss_*_weight keys, else legacy (reverse_kl=1 + confidence_head_alpha).
        rk_w = oc.get("dspark_loss_reverse_kl_weight")
        conf_w = oc.get("dspark_loss_confidence_weight")
        if rk_w is None and conf_w is None:
            rk_w = 1.0
            conf_w = float(oc.get("confidence_head_alpha", 1.0))
        else:
            rk_w = 1.0 if rk_w is None else float(rk_w)
            conf_w = 0.0 if conf_w is None else float(conf_w)
        # reject_kl (dual-stream reject term): default 0 = off (single-stream, pre-change behavior).
        # Set dspark_loss_reject_kl_weight>0 to train on the reject/boundary stream (T5 path only).
        rej_w = float(oc.get("dspark_loss_reject_kl_weight", 0.0) or 0.0)
        # forward_kl (accept-stream forward KL, §4b): default 0 = off (accept stream uses reverse_kl).
        # Set dspark_loss_forward_kl_weight>0 (and typically reverse_kl_weight=0) to switch the accept
        # stream to mass-covering forward KL (teacher top-K). T5 path only.
        fwd_w = float(oc.get("dspark_loss_forward_kl_weight", 0.0) or 0.0)
        terms = {}
        if rk_w != 0:
            terms["reverse_kl"] = rk_w
        if fwd_w != 0:
            terms["forward_kl"] = fwd_w
        if rej_w != 0:
            terms["reject_kl"] = rej_w
        if conf_w != 0:
            terms["confidence"] = conf_w

        return cls(
            terms=terms,
            reverse_kl_mode=str(oc.get("dspark_loss_reverse_kl_mode", "topk_pathwise")),
            loss_decay_gamma=_opt_float(
                oc.get("dspark_loss_decay_gamma", oc.get("loss_decay_gamma", 4.0))),
            loss_agg_mode=str(oc.get("dspark_loss_agg_mode", "token-mean")),
            loss_max_clamp=_opt_float(oc.get("dspark_loss_max_clamp", None)),
        )


def _opt_float(v) -> float | None:
    """None passes through (disabled); everything else -> float."""
    if v is None:
        return None
    return float(v)


__all__ = ["DSparkOPDLossConfig"]
