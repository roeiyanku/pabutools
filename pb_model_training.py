"""
Model *training* for the learning-based prediction modules of
"A Recommendation System for Participatory Budgeting"
(Leibiker & Talmon, 2023), Section 2.1.

This module is intentionally separate from the module that *uses* the models
(``pb_recommendation``): a ``train_*`` function fits an estimator and returns an
opaque model object, while the matching ``predict_by_*`` function in
``pb_recommendation`` consumes it to complete a single Target Voter's partial
ballot. Keeping them apart also lets the (heavy) ML dependencies be imported only
on the training side.

What each model trains on follows the paper (Section 3.1: predictions use the
preferences of LV *and* the preferences in E_TV):

* :py:func:`train_classification` is supervised, so it trains on the **Learning
  Voters' full ballots only** (features = votes on the exposed projects). The
  Target Voter's exposed votes enter later, as features, at prediction time - so
  one trained classifier can be reused across many Target Voters.
* :py:func:`train_matrix_factorization` and
  :py:func:`train_factorization_machines` are collaborative filtering, so the
  Target Voter must be **inside** the fitted user-item matrix: they train on the
  Learning Voters' full ballots **and** that Target Voter's exposed set E_v
  (passed in as the ``approved`` / ``disapproved`` project sets), and are
  therefore (re)fitted per Target Voter.

.. note::
    Per the pabutools maintainer (Simon Rey, issue thread): completing a partial
    vote should live in a **separate module** and the ML libraries must **not be
    a hard requirement** - exactly how pabutools treats Jinja for rule
    explanations. So every ML dependency here is imported lazily inside the
    ``train_*`` function that needs it, raising a friendly ``ImportError`` if it
    is missing, and is declared only as an optional extra in ``pyproject.toml``.

Programmer: Roei Yanku
Date: 2026-06-20.
"""

from __future__ import annotations

from pabutools.election import (
    Instance,
    Project,
    ApprovalProfile,
)

# This module is deliberately *ballot-agnostic*: it takes plain project sets, not
# the partial CardinalBallot. The caller (``pb_recommendation``) decodes the
# ballot into approved / disapproved / exposed sets and passes those in. That
# keeps the +1/-1/0 convention in one place (``pb_recommendation``) and the
# import one-directional (``pb_recommendation`` -> here), with no cycle.


# ---------------------------------------------------------------------------
# Section 2.1.1 - Binary classification (one classifier per project).
# ---------------------------------------------------------------------------
def train_classification(
    instance: Instance,
    lv_profile: ApprovalProfile,
    exposed: set[Project],
) -> object:
    """
    Train the per-project binary classifiers of Section 2.1.1 on the LV ballots.
    For every project the voter might be asked to predict, a classifier is fitted
    whose features are the votes on the ``exposed`` projects and whose label is
    the vote on the target project. Backed by the external ``xgboost`` library
    (:py:class:`xgboost.XGBClassifier`, class-weighted loss for the imbalanced
    data), imported inside the implementation.

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
        object
            A fitted model (e.g. a mapping from each hidden project to its
            trained classifier) to be passed to
            :py:func:`pb_recommendation.predict_by_classification`.
    """
    return None  # Empty implementation


# ---------------------------------------------------------------------------
# Section 2.1.2 - Collaborative filtering via Matrix Factorization.
# ---------------------------------------------------------------------------
def train_matrix_factorization(
    instance: Instance,
    lv_profile: ApprovalProfile,
    approved: set[Project],
    disapproved: set[Project],
) -> object:
    """
    Train the Matrix Factorization model of Section 2.1.2. Collaborative
    filtering needs the Target Voter inside the matrix, so the model is fitted on
    **both** the Learning Voters' full ballots **and** the Target Voter's exposed
    set E_v (paper Section 3.1: predictions use the preferences of LV *and* the
    preferences in E_TV): build the sparse user-item rating matrix with one row
    per LV voter (full) plus one row for this Target Voter holding only her
    exposed votes (``approved`` -> 1, ``disapproved`` -> 0; the hidden projects
    are left empty), then factorise it. The reconstructed cells of the Target
    Voter's row over the hidden projects are what gets read off at prediction
    time. Without her exposed votes the model could only reproduce the LV
    average. Backed by the external ``scikit-surprise`` library
    (:py:class:`surprise.SVD`), imported inside the implementation.

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
        object
            A fitted factorisation model to be passed to
            :py:func:`pb_recommendation.predict_by_matrix_factorization`.
    """
    return None  # Empty implementation


# ---------------------------------------------------------------------------
# Section 2.1.2 - Hybrid Factorization Machines.
# ---------------------------------------------------------------------------
def train_factorization_machines(
    instance: Instance,
    lv_profile: ApprovalProfile,
    approved: set[Project],
    disapproved: set[Project],
) -> object:
    """
    Train the Factorization Machines model of Section 2.1.2: like Matrix
    Factorization but with a linear term plus pairwise latent interactions and
    optional side features. Being collaborative filtering, it is fitted on
    **both** the Learning Voters' full ballots **and** the Target Voter's exposed
    set E_v (``approved`` -> 1, ``disapproved`` -> 0), exactly as in
    :py:func:`train_matrix_factorization`; the hidden projects are what the fitted
    model predicts. Backed by an external FM library (e.g. ``lightfm`` /
    ``fastFM``), imported inside the implementation.

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
        object
            A fitted FM model to be passed to
            :py:func:`pb_recommendation.predict_by_factorization_machines`.
    """
    return None  # Empty implementation


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
