"""
Model *training* for the learning-based prediction modules of
"A Recommendation System for Participatory Budgeting"
(Leibiker & Talmon, 2023), Section 2.1.

This module is intentionally separate from the module that *uses* the models
(``pb_recommendation``): a ``train_*`` function fits an estimator and returns an
opaque model object, while the matching ``predict_by_*`` function in
``pb_recommendation`` consumes it to complete a single Target Voter's partial
ballot.

What each model trains on follows the paper (Section 3.1: predictions use the
preferences of LV *and* the preferences in E_TV):

* :py:func:`train_classification` is supervised, so it trains on the **Learning
  Voters' full ballots** (features = votes on the exposed projects, one binary
  classifier per project). The Target Voter's exposed votes enter later, as
  features, at prediction time.
* :py:func:`train_matrix_factorization` and
  :py:func:`train_factorization_machines` are collaborative filtering, so the
  Target Voter must be **inside** the fitted user-item matrix: they train on the
  Learning Voters' full ballots **and** that Target Voter's exposed set E_v
  (passed in as the ``approved`` / ``disapproved`` project sets).

.. note::
    The paper backs these with ``xgboost`` (classification), ``scikit-surprise``
    (matrix factorization) and ``lightfm`` (factorization machines). Following
    the maintainer's advice, the ML libraries are **not a hard requirement**:
    ``xgboost`` is imported lazily inside :py:func:`train_classification`. The
    ``scikit-surprise`` / ``lightfm`` wheels could not be built in this
    environment (Windows + NumPy 2), so the matrix-factorization and
    factorization-machines models are computed with a small, self-contained
    NumPy factorization instead - ``numpy`` is already a pabutools dependency.

This module is deliberately *ballot-agnostic*: it takes plain project sets, not
the partial CardinalBallot. The caller (``pb_recommendation``) decodes the ballot
into approved / disapproved / exposed sets and passes those in.

Programmer: Roei Yanku
Date: 2026-06-20.
"""

from __future__ import annotations

import logging

import numpy as np

from pabutools.election import (
    Instance,
    Project,
    ApprovalProfile,
    ApprovalBallot,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section 2.1.1 - Binary classification (one classifier per project).
# ---------------------------------------------------------------------------
def train_classification(
    instance: Instance,
    lv_profile: ApprovalProfile,
    exposed: set[Project],
) -> dict:
    """
    Train the per-project binary classifiers of Section 2.1.1 on the LV ballots.
    For every project the voter might be asked to predict, a classifier is fitted
    whose features are the votes on the ``exposed`` projects and whose label is
    the vote on the target project. Backed by the external ``xgboost`` library
    (:py:class:`xgboost.XGBClassifier`, class-weighted for the imbalanced data),
    imported lazily. When there are no exposed features, or all LV voters agree on
    the target (a single class), no classifier can be trained, so we fall back to
    the constant majority label.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters, used as the training data.
        exposed : set[:py:class:`~pabutools.election.instance.Project`]
            The projects used as features (those exposed to the TV voters).

    Returns
    -------
        dict
            ``{"features": [...], "per_project": {project: ("const", 0/1) or
            ("model", classifier)}}``, consumed by
            :py:func:`pb_recommendation.predict_by_classification`.

    Examples
    --------
    With nothing exposed and unanimous LV approvals, every project falls back to
    the constant majority label (1 for the approved projects, 0 otherwise).

    >>> p1, p2 = Project("p1", 1), Project("p2", 1)
    >>> inst = Instance([p1, p2], budget_limit=2)
    >>> lv = ApprovalProfile([ApprovalBallot([p1])] * 3)
    >>> model = train_classification(inst, lv, exposed=set())
    >>> model["per_project"][p1], model["per_project"][p2]
    (('const', 1), ('const', 0))
    """
    features = sorted(exposed, key=str)
    lv = list(lv_profile)
    per_project: dict[Project, tuple] = {}
    for target in instance:
        labels = [1 if target in ballot else 0 for ballot in lv]
        if not features or not lv or len(set(labels)) < 2:
            # No features or a single class -> the majority label is the best we
            # can do; a classifier cannot be trained.
            approve = 1 if (labels and 2 * sum(labels) >= len(labels)) else 0
            per_project[target] = ("const", approve)
            continue
        try:
            import xgboost
        except ImportError:
            raise ImportError(
                "You need to install xgboost to train the classification "
                "predictor (pip install pabutools[recommendation])."
            )
        X = np.array([[1 if f in ballot else 0 for f in features] for ballot in lv])
        y = np.array(labels)
        positives = int(y.sum())
        negatives = len(y) - positives
        scale_pos_weight = (negatives / positives) if positives else 1.0
        classifier = xgboost.XGBClassifier(
            n_estimators=50, max_depth=3, verbosity=0,
            scale_pos_weight=scale_pos_weight,
        )
        classifier.fit(X, y)
        per_project[target] = ("model", classifier)
    logger.info(
        "train_classification: %d features, %d per-project classifiers",
        len(features), len(per_project),
    )
    return {"features": features, "per_project": per_project}


# ---------------------------------------------------------------------------
# Section 2.1.2 - Collaborative filtering via Matrix Factorization.
# ---------------------------------------------------------------------------
def _factorized_tv_scores(
    instance: Instance,
    lv_profile: ApprovalProfile,
    approved: set[Project],
    disapproved: set[Project],
    rank: int = 20,
) -> dict[Project, float]:
    """
    Shared collaborative-filtering core (Section 2.1.2). Builds the user-item
    rating matrix - one row per LV voter (approve=1, else 0) plus one row for the
    Target Voter (``approved`` -> 1, ``disapproved`` -> 0, unknown -> the item's
    LV mean) - centres it by the item means, and reconstructs it from a truncated
    SVD of rank ``rank``. Returns the reconstructed score of every project in the
    Target Voter's row, i.e. the predicted approval level in [0, 1] (roughly).

    Examples
    --------
    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2])] * 3)
    >>> scores = _factorized_tv_scores(inst, lv, set(), set())
    >>> scores[p1] >= 0.5 and scores[p2] >= 0.5 and scores[p3] < 0.5
    True
    """
    items = sorted(instance, key=str)
    lv = list(lv_profile)
    if lv:
        lv_matrix = np.array(
            [[1.0 if item in ballot else 0.0 for item in items] for ballot in lv]
        )
        item_means = lv_matrix.mean(axis=0)
    else:
        lv_matrix = np.zeros((0, len(items)))
        item_means = np.zeros(len(items))
    # The Target Voter's row: known exposed votes, unknown cells seeded with the
    # item mean so the matrix is complete before factorising.
    tv_row = np.array([
        1.0 if item in approved else (0.0 if item in disapproved else item_means[j])
        for j, item in enumerate(items)
    ])
    matrix = np.vstack([lv_matrix, tv_row])
    centered = matrix - item_means
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    kept = min(rank, len(singular))
    reconstruction = (u[:, :kept] * singular[:kept]) @ vt[:kept] + item_means
    tv_scores = reconstruction[-1]
    return {item: float(tv_scores[j]) for j, item in enumerate(items)}


def train_matrix_factorization(
    instance: Instance,
    lv_profile: ApprovalProfile,
    approved: set[Project],
    disapproved: set[Project],
) -> dict[Project, float]:
    """
    Train the Matrix Factorization model of Section 2.1.2. Collaborative filtering
    needs the Target Voter inside the matrix, so the model is fitted on **both**
    the Learning Voters' full ballots **and** the Target Voter's exposed set E_v
    (``approved`` -> 1, ``disapproved`` -> 0). The user-item matrix is factorised
    (truncated SVD) and the Target Voter's row is reconstructed; the score of each
    hidden project is what gets thresholded at 0.5 at prediction time. Without her
    exposed votes the model could only reproduce the LV average.

    .. note::
        The paper uses ``scikit-surprise`` (:py:class:`surprise.SVD`). That wheel
        could not be built here (Windows + NumPy 2), so the factorization is done
        with NumPy via :py:func:`_factorized_tv_scores`.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the Learning Voters (LV).
        approved : set[:py:class:`~pabutools.election.instance.Project`]
            The Target Voter's exposed approvals A_v (known cells set to 1).
        disapproved : set[:py:class:`~pabutools.election.instance.Project`]
            The Target Voter's exposed disapprovals D_v (known cells set to 0).

    Returns
    -------
        dict[:py:class:`~pabutools.election.instance.Project`, float]
            The reconstructed approval score of every project for this Target
            Voter, consumed by
            :py:func:`pb_recommendation.predict_by_matrix_factorization`.

    Examples
    --------
    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2])] * 3)
    >>> scores = train_matrix_factorization(inst, lv, set(), set())
    >>> scores[p1] >= 0.5 and scores[p3] < 0.5
    True
    """
    logger.info("train_matrix_factorization: factorising the user-item matrix")
    return _factorized_tv_scores(instance, lv_profile, approved, disapproved)


# ---------------------------------------------------------------------------
# Section 2.1.2 - Hybrid Factorization Machines.
# ---------------------------------------------------------------------------
def train_factorization_machines(
    instance: Instance,
    lv_profile: ApprovalProfile,
    approved: set[Project],
    disapproved: set[Project],
) -> dict[Project, float]:
    """
    Train the Factorization Machines model of Section 2.1.2: like Matrix
    Factorization but with a linear (bias) term plus pairwise latent interactions.
    It is fitted on **both** the Learning Voters' full ballots **and** the Target
    Voter's exposed set E_v (``approved`` -> 1, ``disapproved`` -> 0). Centring the
    matrix by the item means captures the linear per-item bias, and the truncated
    SVD captures the pairwise latent interactions, so the same
    :py:func:`_factorized_tv_scores` core is reused; the hidden projects are what
    the fitted model predicts.

    .. note::
        The paper uses an external FM library (``lightfm`` / ``fastFM``). Those
        wheels could not be built here (Windows + NumPy 2), so the model is
        computed with the same NumPy factorization core.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the Learning Voters (LV).
        approved : set[:py:class:`~pabutools.election.instance.Project`]
            The Target Voter's exposed approvals A_v (known cells set to 1).
        disapproved : set[:py:class:`~pabutools.election.instance.Project`]
            The Target Voter's exposed disapprovals D_v (known cells set to 0).

    Returns
    -------
        dict[:py:class:`~pabutools.election.instance.Project`, float]
            The predicted approval score of every project for this Target Voter,
            consumed by
            :py:func:`pb_recommendation.predict_by_factorization_machines`.

    Examples
    --------
    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2])] * 3)
    >>> scores = train_factorization_machines(inst, lv, set(), set())
    >>> scores[p1] >= 0.5 and scores[p3] < 0.5
    True
    """
    logger.info("train_factorization_machines: fitting the FM model")
    return _factorized_tv_scores(instance, lv_profile, approved, disapproved)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
