"""Estimating lift from a holdout group, the way a real store has to.

A real store never knows which shoppers "would have bought anyway". What it can
do is randomly keep a share of abandoned checkouts message-free (the holdout)
and compare revenue per shopper between the groups. Because assignment is
random, the two groups are alike in everything except the messages, so the
difference in average revenue is an unbiased estimate of what messaging adds.

    lift per shopper     = mean revenue (messaged) - mean revenue (holdout)
    estimated total lift = lift per shopper x number of messaged shoppers

The 95% interval uses the normal approximation to the difference of two means
(Welch's standard error). With a few hundred shoppers per group it is wide;
that is the honest answer, and why small stores can't measure small effects.
"""
import math
from dataclasses import dataclass

Z_95 = 1.96


@dataclass(frozen=True)
class LiftEstimate:
    n_treatment: int
    n_holdout: int
    mean_treatment: float        # revenue per messaged shopper, cents
    mean_holdout: float          # revenue per holdout shopper, cents
    lift_per_shopper: float      # cents
    total_lift: float            # cents, scaled to the messaged group
    ci_low: float                # cents, 95% interval for total_lift
    ci_high: float
    conversion_treatment: float  # share who bought
    conversion_holdout: float

    @property
    def significant(self) -> bool:
        """True when the whole interval is on one side of zero."""
        return self.ci_low > 0 or self.ci_high < 0


def _mean_var(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return mean, var


def estimate_lift(
    treatment_revenue: list[int],
    holdout_revenue: list[int],
    treatment_bought: list[bool],
    holdout_bought: list[bool],
) -> LiftEstimate | None:
    """Needs at least two shoppers in each group; otherwise there is nothing to compare."""
    n_t, n_h = len(treatment_revenue), len(holdout_revenue)
    if n_t < 2 or n_h < 2:
        return None
    mean_t, var_t = _mean_var([float(x) for x in treatment_revenue])
    mean_h, var_h = _mean_var([float(x) for x in holdout_revenue])
    diff = mean_t - mean_h
    se = math.sqrt(var_t / n_t + var_h / n_h)
    return LiftEstimate(
        n_treatment=n_t,
        n_holdout=n_h,
        mean_treatment=mean_t,
        mean_holdout=mean_h,
        lift_per_shopper=diff,
        total_lift=diff * n_t,
        ci_low=(diff - Z_95 * se) * n_t,
        ci_high=(diff + Z_95 * se) * n_t,
        conversion_treatment=sum(treatment_bought) / n_t,
        conversion_holdout=sum(holdout_bought) / n_h,
    )
