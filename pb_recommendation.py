"""
An implementation of the algorithms in:
"A Recommendation System for Participatory Budgeting",
by Gil Leibiker and Nimrod Talmon (2023), https://optlearnmas23.github.io/files/p17.pdf

Programmer: Roei Yanku
Date: 2026-06-20.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable

import numpy as np

from pabutools.election import (
    Instance,
    Project,
    ApprovalProfile,
    ApprovalBallot,
    CardinalBallot,
    total_cost,
)
from pabutools.election.satisfaction import Cost_Sat
from pabutools.rules import BudgetAllocation, greedy_utilitarian_welfare

# Model training lives in a separate module, ``pb_model_training``. The
# learning-based predictors below delegate the fitting to its ``train_*``
# functions; ``pb_model_training`` is ballot-agnostic (it takes plain project
# sets), so the dependency is one-directional and there is no import cycle.
from pb_model_training import (
    train_classification,
    train_matrix_factorization,
    train_factorization_machines,
)

# Log the steps of every algorithm at INFO/DEBUG level.
#  Configure a handler in your own script to see them, e.g.
# ``logging.basicConfig(level=logging.INFO)``.
logger = logging.getLogger(__name__)


# ===========================================================================
# Section 2.3 - Partial (three-state) ballots, encoded as a CardinalBallot.
# ===========================================================================
# A Target Voter answers only k projects, so "unknown" must be distinct from
# "disapproved" - which a plain approval set (just the approved projects) cannot
# express. On the pabutools maintainer's advice (Simon Rey) we do NOT add a new
# ballot type; instead we reuse the existing
# :py:class:`~pabutools.election.ballot.cardinalballot.CardinalBallot` (a dict of
# project -> score) under this sign convention:
#
#     score > 0   ->  APPROVED     (A_v)   we store +1  (APPROVAL)
#     score < 0   ->  DISAPPROVED  (D_v)   we store -1  (DISAPPROVAL)
#     score == 0  ->  HIDDEN       (H_v)   we store  0  (HIDDEN)
#
# A project simply absent from the ballot is also HIDDEN. This convention is not
# self-evident, so never hand-write the scores: build ballots with
# ``partial_ballot`` / ``reveal_ballot`` and read them back with the
# ``*_projects`` accessors / ``as_approval_ballot``.

#: Score stored for an approved project (A_v). Any strictly positive score works.
APPROVAL = 1
#: Score stored for a disapproved project (D_v). Any strictly negative score works.
DISAPPROVAL = -1
#: Score stored for a hidden project (H_v); absent projects mean the same.
HIDDEN = 0


def partial_ballot(
    approved: Iterable[Project] = (),
    disapproved: Iterable[Project] = (),
    hidden: Iterable[Project] = (),
) -> CardinalBallot:
    """
    Build a partial ballot from the three explicit sets, hiding the convention:
    approved projects get +1, disapproved get -1, and hidden get 0. This is the
    intended way to create a partial ballot - callers name the three sets instead
    of writing raw scores.

    Examples
    --------
    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> b = partial_ballot(approved={p1}, disapproved={p2}, hidden={p3})
    >>> b[p1], b[p2], b[p3]
    (1, -1, 0)
    >>> exposed_projects(b) == {p1, p2}
    True
    """
    return CardinalBallot(
        {p: APPROVAL for p in approved}
        | {p: DISAPPROVAL for p in disapproved}
        | {p: HIDDEN for p in hidden}
    )


def reveal_ballot(
    instance: Instance,
    full_ballot: set[Project],
    exposed: set[Project],
) -> CardinalBallot:
    """
    Build the partial ballot exposing a set of projects of a voter whose full
    ballot in the ideal instance is known (Section 2.3): A_v = E_v ∩ full_ballot,
    D_v = E_v \\ full_ballot, H_v = P \\ E_v.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance (used as the universe of projects P).
        full_ballot : set[:py:class:`~pabutools.election.instance.Project`]
            The voter's full ballot - her approval set A_v in the ideal instance.
        exposed : set[:py:class:`~pabutools.election.instance.Project`]
            The projects revealed for this voter (the exposed set E_v).

    Returns
    -------
        :py:class:`~pabutools.election.ballot.cardinalballot.CardinalBallot`
            The corresponding three-state ballot under the sign convention.

    Examples
    --------
    Example 2.4 from the paper: P = {p1, p2, p3, p4}, the voter's full ballot is
    {p1, p2}, and {p1, p3} is exposed. Then p1 is approved, p3 disapproved, and
    p2, p4 remain hidden.

    >>> p1, p2, p3, p4 = (Project("p1", 1), Project("p2", 1),
    ...                   Project("p3", 1), Project("p4", 1))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=4)
    >>> b = reveal_ballot(inst, {p1, p2}, {p1, p3})
    >>> approved_projects(b) == {p1}, disapproved_projects(b) == {p3}
    (True, True)
    >>> hidden_projects(b, inst) == {p2, p4}
    True
    """
    return partial_ballot(
        approved=exposed & full_ballot,       # A_v = E_v ∩ full ballot
        disapproved=exposed - full_ballot,    # D_v = E_v \ full ballot
        hidden=set(instance) - exposed,       # H_v = P \ E_v
    )


def approved_projects(ballot: CardinalBallot) -> set[Project]:
    """The approval set A_v: the projects with a strictly positive score."""
    return {project for project, score in ballot.items() if score > 0}


def disapproved_projects(ballot: CardinalBallot) -> set[Project]:
    """The disapproval set D_v: the projects with a strictly negative score."""
    return {project for project, score in ballot.items() if score < 0}


def exposed_projects(ballot: CardinalBallot) -> set[Project]:
    """The exposed set E_v = A_v ∪ D_v: the projects with a non-zero score."""
    return {project for project, score in ballot.items() if score != 0}


def hidden_projects(ballot: CardinalBallot, instance: Instance) -> set[Project]:
    """
    The hidden set H_v = P \\ E_v: every project of the instance that the voter
    was not asked about (score 0 or absent from the ballot).
    """
    return set(instance) - exposed_projects(ballot)


def as_approval_ballot(ballot: CardinalBallot) -> ApprovalBallot:
    """
    The pabutools approval ballot made of the approved projects A_v, so that a
    (completed) partial ballot can be fed to approval-based voting rules.
    """
    return ApprovalBallot(approved_projects(ballot))


# ---------------------------------------------------------------------------
# Section 2.2.3 - Popularity and consensus (primitives used by every module).
# ---------------------------------------------------------------------------
# Definition 2.1 (Approval scores) needs no function of our own: it is exactly
# pabutools' ``profile.approval_scores()``. Callers below use the library method
# directly (``.get(p, 0)`` supplies the 0 for projects that nobody approved).
def consensus_levels(
    instance: Instance, profile: ApprovalProfile
) -> dict[Project, int]:
    """
    Definition 2.2 (Consensus levels): the absolute difference between the number
    of voters approving a project and the number disapproving it (so a unanimous
    approval or a unanimous rejection both reach the maximum n).

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The approval profile.

    Returns
    -------
        dict[:py:class:`~pabutools.election.instance.Project`, int]
            A mapping from each project to its consensus level.

    Examples
    --------
    Example 7 from the paper (4 LV voters): p1 approved 4/0, p2 split 2/2,
    p3 rejected 0/4. Both p1 and p3 are in full consensus, p2 in none.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> prof = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1]),
    ...                         ApprovalBallot([p1, p2]), ApprovalBallot([p1])])
    >>> c = consensus_levels(inst, prof)
    >>> [c[p] for p in (p1, p2, p3)]
    [4, 0, 4]
    """
    # consensus(p) = |approvers(p) - disapprovers(p)|. With n voters and
    # score(p) approvers, disapprovers = n - score(p), so the difference is
    # |score - (n - score)| = |2*score - n|. Scores come from the library
    # (Definition 2.1 = pabutools' approval_scores()).
    n = profile.num_ballots()
    scores = profile.approval_scores()
    consensus = {
        project: abs(2 * scores.get(project, 0) - n) for project in instance
    }
    logger.debug("consensus_levels (n=%d): %s", n, consensus)
    return consensus


def most_consensual_projects(
    instance: Instance, profile: ApprovalProfile
) -> list[Project]:
    """
    The paper's ordering gamma (Section 2.2.3): the projects sorted by decreasing
    consensus level, ties broken lexicographically by project name. gamma_1 is the
    project most in consensus, gamma_m the "most controversial". Helper used by
    offline revealing-by-consensus and by-controversiality.

    Examples
    --------
    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> prof = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1]),
    ...                         ApprovalBallot([p1, p2]), ApprovalBallot([p1])])
    >>> most_consensual_projects(inst, prof)  # consensus p1=4, p3=4, p2=0
    [p1, p3, p2]
    """
    consensus = consensus_levels(instance, profile)
    return sorted(instance, key=lambda project: (-consensus[project], str(project)))


# ---------------------------------------------------------------------------
# Section 2.2.5 - The voting rule (greedy approval).
# ---------------------------------------------------------------------------
def greedy_approval(
    instance: Instance, profile: ApprovalProfile
) -> BudgetAllocation:
    """
    Greedy approval voting rule (Section 2.2.5). Projects are considered in
    decreasing order of their approval score and funded while the budget
    allows; a project that would exceed the remaining budget is skipped. Ties
    are broken lexicographically by project name.

    .. note::
        Delegates to pabutools'
        :py:func:`~pabutools.rules.greedy_utilitarian_welfare` with
        :py:class:`~pabutools.election.satisfaction.additivesatisfaction.Cost_Sat`:
        the rule ranks by marginal satisfaction divided by cost, and under
        Cost_Sat a project's marginal satisfaction is score(p)*cost(p), so the
        ranking is by the **raw** approval score - exactly the paper's greedy
        approval. (With the default Cardinality_Sat the ranking would be
        score/cost, which is a different rule.)

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The (completed) approval profile.

    Returns
    -------
        :py:class:`~pabutools.rules.budgetallocation.BudgetAllocation`
            The winning bundle.

    Examples
    --------
    Example 1 from the paper: scores p1 = p2 = 2, p3 = 1, budget 3. p1 and p2
    are funded (cost 1 each); p3 (cost 2) no longer fits.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 2)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> prof = ApprovalProfile([ApprovalBallot([p1, p2]),
    ...                         ApprovalBallot([p1, p3]),
    ...                         ApprovalBallot([p2])])
    >>> sorted(greedy_approval(inst, prof), key=str)
    [p1, p2]
    """
    chosen = greedy_utilitarian_welfare(
        instance, profile, sat_class=Cost_Sat, resoluteness=True
    )
    assert isinstance(chosen, BudgetAllocation)  # resolute mode: one allocation
    logger.info(
        "greedy_approval: funded %d/%d projects, cost %s of %s",
        len(chosen), len(instance), total_cost(chosen), instance.budget_limit,
    )
    return chosen


# ---------------------------------------------------------------------------
# Section 3.1.1 - Random setup.
# ---------------------------------------------------------------------------
def random_setup(
    instance: Instance,
    k: int,
    seed: int | None = None,
) -> set[Project]:
    """
    Algorithm 1 - Random setup (Section 3.1.1): the exposed set E_v of a Target
    Voter, made of k projects chosen uniformly at random from P (the rest is
    predicted later). Called once per TV voter, so a different draw can be drawn
    for each; it needs no ballot, since the choice is random.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance (the universe of projects P).
        k : int
            The number of projects to expose.
        seed : int, optional
            Seed for the random generator, for reproducibility.

    Returns
    -------
        set[:py:class:`~pabutools.election.instance.Project`]
            The exposed set E_v: k projects drawn uniformly at random from P.

    Examples
    --------
    Example 5 from the paper: k = 2. Whatever the draw, exactly two projects are
    exposed and they are real projects of the instance.

    >>> p1, p2, p3, p4 = (Project("p1", 2), Project("p2", 2),
    ...                   Project("p3", 3), Project("p4", 3))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=6)
    >>> exposed = random_setup(inst, k=2, seed=0)
    >>> len(exposed) == 2 and exposed <= {p1, p2, p3, p4}
    True
    """
    rng = random.Random(seed)
    # Sort first so the sample is reproducible from the seed (a set has no
    # deterministic iteration order).
    population = sorted(instance, key=str)
    exposed = set(rng.sample(population, k))
    logger.info("random_setup: exposed %d of %d projects at random", k, len(population))
    return exposed


# ---------------------------------------------------------------------------
# Section 3.1.2 - Offline setup (popularity / consensus / controversiality).
# ---------------------------------------------------------------------------
def offline_popularity(
    instance: Instance, lv_profile: ApprovalProfile, k: int
) -> set[Project]:
    """
    Algorithm 2 - Offline revealing by popularity (Section 3.1.2). The exposed
    set E = {sigma_1, ..., sigma_k}: the k most approved projects among the LV
    voters. The same set is exposed to every Target Voter, so it depends only on
    the LV profile (the score ordering sigma is used only to pick the top k; the
    exposed set itself is unordered).

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the Learning Voters (LV).
        k : int
            The number of projects to expose.

    Returns
    -------
        set[:py:class:`~pabutools.election.instance.Project`]
            The exposed set E of the k most popular projects (ties broken by name
            when picking the top k).

    Examples
    --------
    Example 6 from the paper: LV scores p1 = 3, p2 = 1, p3 = 1, so the single
    most popular project is p1.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p3]),
    ...                       ApprovalBallot([p1, p2]),
    ...                       ApprovalBallot([p1])])
    >>> offline_popularity(inst, lv, k=1) == {p1}
    True
    """
    # The paper's sigma ordering: decreasing library approval score, ties by name.
    scores = lv_profile.approval_scores()
    sigma = sorted(instance, key=lambda p: (-scores.get(p, 0), str(p)))
    exposed = set(sigma[:k])
    logger.info("offline_popularity: exposing top-%d popular projects", k)
    return exposed


def offline_consensus(
    instance: Instance, lv_profile: ApprovalProfile, k: int
) -> set[Project]:
    """
    Algorithm 3 - Offline revealing by consensus (Section 3.1.2). The exposed set
    E = {gamma_1, ..., gamma_k}: the k projects with the highest consensus level
    among the LV voters. The same set is exposed to every Target Voter.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the Learning Voters (LV).
        k : int
            The number of projects to expose.

    Returns
    -------
        set[:py:class:`~pabutools.election.instance.Project`]
            The exposed set E of the k projects most in consensus (ties broken by
            name when picking the top k).

    Examples
    --------
    Example 7 from the paper: p1 and p3 both have consensus 4 (p1 unanimously
    approved, p3 unanimously rejected); the tie is broken by name towards p1.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1]),
    ...                       ApprovalBallot([p1, p2]), ApprovalBallot([p1])])
    >>> offline_consensus(inst, lv, k=1) == {p1}
    True
    """
    exposed = set(most_consensual_projects(instance, lv_profile)[:k])
    logger.info("offline_consensus: exposing top-%d consensual projects", k)
    return exposed


def offline_controversiality(
    instance: Instance, lv_profile: ApprovalProfile, k: int
) -> set[Project]:
    """
    Algorithm 4 - Offline revealing by controversiality (Section 3.1.2). The
    exposed set E = {gamma_{m-k+1}, ..., gamma_m}: the k projects *least* in
    consensus among the LV voters (the hardest to predict, so asked directly).
    The same set is exposed to every Target Voter.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the Learning Voters (LV).
        k : int
            The number of projects to expose.

    Returns
    -------
        set[:py:class:`~pabutools.election.instance.Project`]
            The exposed set E of the k most controversial projects (ties broken
            by name when picking the bottom k).

    Examples
    --------
    Example 8 from the paper (same data as Example 7): p2 is split exactly in
    half (consensus 0) and is therefore the single most controversial project.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1]),
    ...                       ApprovalBallot([p1, p2]), ApprovalBallot([p1])])
    >>> offline_controversiality(inst, lv, k=1) == {p2}
    True
    """
    # The k *least* consensual projects are the last k of gamma,
    # {gamma_{m-k+1}, ..., gamma_m}. Slicing from m-k keeps k=0 correct.
    gamma = most_consensual_projects(instance, lv_profile)
    exposed = set(gamma[len(gamma) - k:])
    logger.info("offline_controversiality: exposing bottom-%d consensual projects", k)
    return exposed


# ---------------------------------------------------------------------------
# Section 3.1.3 - Online setup (adaptive controversial).
# ---------------------------------------------------------------------------
def online_adaptive_controversial(
    instance: Instance,
    lv_profile: ApprovalProfile,
    full_ballot: set[Project],
    k: int,
) -> set[Project]:
    """
    Algorithm 5 - Online adaptive-controversial setup (Section 3.1.3): the
    exposed set E_v of one Target Voter, built in k iterations. In each iteration
    the most controversial project is recomputed given the LV ballots and the
    answers already revealed, then asked about (each answer affects the next
    question, hence "adaptive"). The voter's full ballot is needed to simulate
    those answers.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the Learning Voters (LV).
        full_ballot : set[:py:class:`~pabutools.election.instance.Project`]
            The full ballot of the Target Voter (TV) being queried - her approval
            set A_v in the ideal instance - used to answer each adaptive question.
        k : int
            The number of iterations / projects to expose.

    Returns
    -------
        set[:py:class:`~pabutools.election.instance.Project`]
            The exposed set E_v of the k projects that ended up being asked.

    Examples
    --------
    Example 9 from the paper: 4 LV voters, one TV voter v5, k = 2. p1, p2, p3
    are all tied as most controversial; the first question (tie broken by name)
    is p1, and after the voter's answer the next is p2, so E_v = {p1, p2}.

    >>> p1, p2, p3, p4 = (Project("p1", 1), Project("p2", 1),
    ...                   Project("p3", 1), Project("p4", 1))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=4)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1, p3]),
    ...                       ApprovalBallot([p2, p3]), ApprovalBallot([])])
    >>> online_adaptive_controversial(inst, lv, {p1, p2}, k=2) == {p1, p2}
    True
    """
    # The consensus of a project is a property of the electorate that has voted
    # on it. A Target Voter has only answered the projects already asked, so her
    # revealed answers never affect the consensus of the *remaining* projects -
    # which is why, for a single voter, recomputing each round is equivalent to
    # ranking once by the LV consensus. We still run the k rounds explicitly and
    # simulate each answer, matching Algorithm 5's structure.
    consensus = consensus_levels(instance, lv_profile)
    exposed: set[Project] = set()
    for round_index in range(1, k + 1):
        # Most controversial not-yet-asked project: lowest consensus, ties by name.
        project = min(set(instance) - exposed, key=lambda p: (consensus[p], str(p)))
        logger.info(
            "online_adaptive_controversial: round %d asks %s -> voter %s",
            round_index, project,
            "approves" if project in full_ballot else "disapproves",
        )
        exposed.add(project)
    return exposed


# ---------------------------------------------------------------------------
# Section 2.1.1 - Binary classification (predicted with a trained model).
# ---------------------------------------------------------------------------
def predict_by_classification(
    instance: Instance,
    lv_profile: ApprovalProfile,
    ballot: CardinalBallot,
) -> ApprovalBallot:
    """
    Prediction module - binary classification (Section 2.1.1): predict each
    hidden project with a per-project binary classifier trained on the LV ballots
    (features = votes on the exposed projects, label = vote on the target), then
    applied to the TV voter. Exposed approvals A_v are kept and exposed
    disapprovals D_v stay rejected. The classifiers are fitted by
    :py:func:`pb_model_training.train_classification` (backed by ``xgboost``);
    this function only *applies* the trained model.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters, used as the training data.
        ballot : :py:class:`~pabutools.election.ballot.cardinalballot.CardinalBallot`
            The Target Voter's partial ballot to complete (the +1/-1/0 partial
            ballot built by ``partial_ballot`` / ``reveal_ballot``).

    Returns
    -------
        :py:class:`~pabutools.election.ballot.approvalballot.ApprovalBallot`
            The full predicted approval ballot of the TV voter.

    Examples
    --------
    Example 11 from the paper: 2 LV voters both approve {p1, p2}. A TV voter with
    nothing exposed is completed to {p1, p2} - on such an unambiguous pattern any
    correct predictor (XGBoost included) agrees with the LV majority.

    >>> p1, p2, p3 = Project("p1", 4), Project("p2", 4), Project("p3", 6)
    >>> inst = Instance([p1, p2, p3], budget_limit=6)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1, p2])])
    >>> partial = partial_ballot(hidden={p1, p2, p3})
    >>> predict_by_classification(inst, lv, partial) == {p1, p2}
    True
    """
    model = train_classification(instance, lv_profile, exposed_projects(ballot))
    approved = approved_projects(ballot)
    # The TV voter's feature row x: her vote (1/0) on each exposed feature project.
    x = np.array([[1 if f in approved else 0 for f in model["features"]]])
    hidden = hidden_projects(ballot, instance)
    votes = {
        p: (payload if kind == "const" else int(payload.predict(x)[0]))
        for p, (kind, payload) in model["per_project"].items()
        if p in hidden
    }
    # predicted ballot = A_v  u  {p in H_v : classifier of p predicts approval}
    result = approved | {p for p in hidden if votes[p] == 1}
    logger.info("predict_by_classification: completed ballot to %d approvals", len(result))
    return ApprovalBallot(result)


def _approve_hidden_by_score(
    instance: Instance, ballot: CardinalBallot, scores: dict[Project, float]
) -> set[Project]:
    """
    Complete a partial ballot from per-project scores: keep the exposed approvals
    A_v and add each hidden project whose predicted score is >= 0.5 (the exposed
    disapprovals D_v stay rejected). Shared by the matrix-factorization and
    factorization-machines predictors.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> b = partial_ballot(approved={p1}, hidden={p2, p3})
    >>> _approve_hidden_by_score(inst, b, {p1: 1.0, p2: 0.8, p3: 0.2}) == {p1, p2}
    True
    """
    # predicted ballot = A_v  u  {p in H_v : score(p) >= 1/2}
    return approved_projects(ballot) | {
        p for p in hidden_projects(ballot, instance) if scores[p] >= 0.5
    }


def predict_by_matrix_factorization(
    instance: Instance,
    lv_profile: ApprovalProfile,
    ballot: CardinalBallot,
) -> ApprovalBallot:
    """
    Prediction module - collaborative filtering via Matrix Factorization
    (Section 2.1.2): build the sparse user-item matrix from the LV ballots and
    exposed TV votes (approve=1, disapprove=0), factorise it, and predict a
    hidden project as approved iff its reconstructed score is >= 0.5. Exposed
    approvals A_v are kept and exposed disapprovals D_v stay rejected. The model
    is fitted by :py:func:`pb_model_training.train_matrix_factorization` (backed
    by ``scikit-surprise``); this function only *applies* the trained model.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters, used as the training data.
        ballot : :py:class:`~pabutools.election.ballot.cardinalballot.CardinalBallot`
            The Target Voter's partial ballot to complete (the +1/-1/0 partial
            ballot built by ``partial_ballot`` / ``reveal_ballot``).

    Returns
    -------
        :py:class:`~pabutools.election.ballot.approvalballot.ApprovalBallot`
            The full predicted approval ballot of the TV voter.

    Examples
    --------
    Example 10 from the paper: three LV voters all support the cheap pair
    {p1, p2} and reject {p3, p4}. The reconstructed matrix completes a TV voter
    with nothing exposed to {p1, p2}.

    >>> p1, p2, p3, p4 = (Project("p1", 3), Project("p2", 3),
    ...                   Project("p3", 4), Project("p4", 4))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=6)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2])] * 3)
    >>> partial = partial_ballot(hidden={p1, p2, p3, p4})
    >>> predict_by_matrix_factorization(inst, lv, partial) == {p1, p2}
    True
    """
    scores = train_matrix_factorization(
        instance, lv_profile, approved_projects(ballot), disapproved_projects(ballot)
    )
    result = _approve_hidden_by_score(instance, ballot, scores)
    logger.info("predict_by_matrix_factorization: completed to %d approvals", len(result))
    return ApprovalBallot(result)


def predict_by_factorization_machines(
    instance: Instance,
    lv_profile: ApprovalProfile,
    ballot: CardinalBallot,
) -> ApprovalBallot:
    """
    Prediction module - hybrid Factorization Machines (Section 2.1.2): like MF
    but with a linear term plus pairwise latent interactions and optional side
    features, predicting a hidden project as approved iff the FM score is >= 0.5.
    Exposed approvals A_v are kept and exposed disapprovals D_v stay rejected.
    The model is fitted by
    :py:func:`pb_model_training.train_factorization_machines` (backed by an
    external FM library, e.g. ``lightfm`` / ``fastFM``); this function only
    *applies* the trained model.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters, used as the training data.
        ballot : :py:class:`~pabutools.election.ballot.cardinalballot.CardinalBallot`
            The Target Voter's partial ballot to complete (the +1/-1/0 partial
            ballot built by ``partial_ballot`` / ``reveal_ballot``).

    Returns
    -------
        :py:class:`~pabutools.election.ballot.approvalballot.ApprovalBallot`
            The full predicted approval ballot of the TV voter.

    Examples
    --------
    Example 11 from the paper: 2 LV voters both approve {p1, p2}. A TV voter with
    nothing exposed is completed to {p1, p2}.

    >>> p1, p2, p3 = Project("p1", 4), Project("p2", 4), Project("p3", 6)
    >>> inst = Instance([p1, p2, p3], budget_limit=6)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1, p2])])
    >>> partial = partial_ballot(hidden={p1, p2, p3})
    >>> predict_by_factorization_machines(inst, lv, partial) == {p1, p2}
    True
    """
    scores = train_factorization_machines(
        instance, lv_profile, approved_projects(ballot), disapproved_projects(ballot)
    )
    result = _approve_hidden_by_score(instance, ballot, scores)
    logger.info("predict_by_factorization_machines: completed to %d approvals", len(result))
    return ApprovalBallot(result)


# ---------------------------------------------------------------------------
# Full pipeline (sampling -> prediction -> greedy approval).
# ---------------------------------------------------------------------------
# The paper's design space is a matrix: one setup (Section 3.1) times one
# predictor (Section 2.1). These registries name every choice so a caller can
# pick a cell (``run_pipeline``) or sweep the whole matrix (``run_all_experiments``).
SETUPS = (
    "random",
    "offline_popularity",
    "offline_consensus",
    "offline_controversiality",
    "online_adaptive_controversial",
)

PREDICTORS = {
    "classification": predict_by_classification,
    "matrix_factorization": predict_by_matrix_factorization,
    "factorization_machines": predict_by_factorization_machines,
}


def exposed_sets(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_ballots: dict[str, set[Project]],
    setup: str,
    k: int,
    seed: int | None = None,
) -> dict[str, set[Project]]:
    """
    The exposed set E_v of every Target Voter under one of the five Section 3.1
    setups, keyed by voter id. Helper that hides the setups' differing
    signatures behind one interface: the three offline samplers expose the
    *same* k projects to everyone, while ``random`` and
    ``online_adaptive_controversial`` are computed per voter (the latter needs
    each voter's full ballot to answer its adaptive questions).

    Examples
    --------
    Offline popularity exposes the single most popular LV project (p1) to every
    Target Voter.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1])])
    >>> exposed_sets(inst, lv, {"v3": {p1}}, "offline_popularity", k=1)
    {'v3': {p1}}
    """
    if setup == "random":
        return {vid: random_setup(instance, k, seed) for vid in tv_ballots}
    if setup == "online_adaptive_controversial":
        return {
            vid: online_adaptive_controversial(instance, lv_profile, full_ballot, k)
            for vid, full_ballot in tv_ballots.items()
        }
    offline = {
        "offline_popularity": offline_popularity,
        "offline_consensus": offline_consensus,
        "offline_controversiality": offline_controversiality,
    }[setup]
    shared = offline(instance, lv_profile, k)  # same set for every TV voter
    return {vid: shared for vid in tv_ballots}


def split_lv_tv(
    profile: ApprovalProfile,
    sample_degree: float,
    lv_degree: float,
    seed: int | None = None,
) -> tuple[ApprovalProfile, dict[str, set[Project]]]:
    """
    Step 1 of the pipeline (Section 2.4 / 3.0.1): partition the voters of the
    *ideal* instance into Learning Voters (LV, who keep their full ballots) and
    Target Voters (TV, whose ballots start hidden and are later completed).

    The partition follows the paper's two knobs (Example 3.1), read here at the
    voter level:

    * ``sample_degree`` - the fraction of voters that are *sampled* (participate).
      The rest are dropped from the partial instance I1.
    * ``lv_degree`` - among the sampled voters, the fraction that are LV; the rest
      are TV. ``lv_degree == 1`` is the paper's naive "sampling" baseline (every
      sampled voter gives a full ballot, no prediction needed).

    Parameters
    ----------
        profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The ideal instance's full ballots (all n voters).
        sample_degree : float
            Fraction of voters sampled, in [0, 1].
        lv_degree : float
            Fraction of the sampled voters that are LV, in [0, 1].
        seed : int, optional
            Seed for the random partition, for reproducibility.

    Returns
    -------
        tuple[:py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`, dict[str, set[:py:class:`~pabutools.election.instance.Project`]]]
            The LV profile (full ballots) and the TV voters' full ballots keyed by
            voter id (their known ground truth, used to simulate the k answers).

    Examples
    --------
    Four voters. Sampling everyone with ``lv_degree == 1`` makes everyone an LV
    and leaves no TV; a half sample split evenly gives one LV and one TV.

    >>> p1, p2 = Project("p1", 1), Project("p2", 1)
    >>> prof = ApprovalProfile([ApprovalBallot([p1]), ApprovalBallot([p2]),
    ...                         ApprovalBallot([p1, p2]), ApprovalBallot([])])
    >>> lv, tv = split_lv_tv(prof, sample_degree=1.0, lv_degree=1.0)
    >>> lv.num_ballots(), len(tv)
    (4, 0)
    >>> lv, tv = split_lv_tv(prof, sample_degree=0.5, lv_degree=0.5, seed=0)
    >>> lv.num_ballots(), len(tv)
    (1, 1)
    """
    voters = list(profile)
    order = list(range(len(voters)))
    random.Random(seed).shuffle(order)
    n_sample = round(sample_degree * len(voters))
    sampled = order[:n_sample]
    n_lv = round(lv_degree * n_sample)
    lv_profile = ApprovalProfile([voters[i] for i in sampled[:n_lv]])
    tv_ballots = {f"v{i}": set(voters[i]) for i in sampled[n_lv:]}
    logger.info(
        "split_lv_tv: sample=%.2f lv=%.2f -> %d LV, %d TV (of %d voters)",
        sample_degree, lv_degree, lv_profile.num_ballots(), len(tv_ballots),
        len(voters),
    )
    return lv_profile, tv_ballots


def run_pipeline(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_ballots: dict[str, set[Project]],
    k: int,
    *,
    setup: str,
    predict,
    seed: int | None = None,
) -> BudgetAllocation:
    """
    The complete Section 3 pipeline for one (setup, predictor) choice: expose k
    projects to each TV voter with ``setup``, complete the hidden votes with the
    ``predict`` module, then run greedy approval on the LV plus completed TV
    ballots. ``setup`` and ``predict`` are required - the pipeline runs exactly
    one cell of the design matrix, so the caller must name both explicitly (the
    sweep over every cell is :py:func:`run_all_experiments`).

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters.
        tv_ballots : dict[str, set[:py:class:`~pabutools.election.instance.Project`]]
            The full ballot of every Target Voter (TV) - her approval set A_v in
            the ideal instance - keyed by voter id.
        k : int
            The number of projects to expose per TV voter.
        setup : str
            The name of the Section 3.1 setup to use (one of :py:data:`SETUPS`).
        predict : callable
            The Section 2.1 prediction module (one of :py:data:`PREDICTORS`'
            values).
        seed : int, optional
            Seed for the ``random`` setup, ignored by the others.

    Returns
    -------
        :py:class:`~pabutools.rules.budgetallocation.BudgetAllocation`
            The predicted winning bundle.

    Examples
    --------
    Example 10 from the paper - the "perfect" case. Three LV voters and one TV
    voter all support the two cheap projects {p1, p2}; the predicted bundle
    coincides with the real one.

    >>> p1, p2, p3, p4 = (Project("p1", 3), Project("p2", 3),
    ...                   Project("p3", 4), Project("p4", 4))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=6)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2])] * 3)
    >>> sorted(run_pipeline(inst, lv, {"v4": {p1, p2}}, k=1,
    ...                     setup="offline_popularity",
    ...                     predict=predict_by_matrix_factorization), key=str)
    [p1, p2]
    """
    # Step 1 (the LV/TV split) is done by the caller / :py:func:`split_lv_tv`.
    # Step 2 - sampling: pick the exposed set of every TV voter under the setup.
    exposed = exposed_sets(instance, lv_profile, tv_ballots, setup, k, seed)
    # Steps 3-4 - prediction + combine: complete every TV ballot from its k
    # exposed answers and merge with the known LV ballots.
    combined = ApprovalProfile(
        list(lv_profile)
        + [
            predict(
                instance,
                lv_profile,
                reveal_ballot(instance, tv_ballots[vid], exposed[vid]),
            )
            for vid in tv_ballots
        ]
    )
    logger.info(
        "recommend: setup=%s predict=%s, %d LV + %d TV ballots",
        setup, predict.__name__, lv_profile.num_ballots(), len(tv_ballots),
    )
    # Steps 5-6 - voting rule: greedy approval on the LV + completed TV ballots.
    return greedy_approval(instance, combined)


#: The partiality grid swept in the paper's experiments (Section 6).
SAMPLE_DEGREES = (0.1, 0.15, 0.3, 0.5, 0.7, 0.9)
LV_DEGREES = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)


def run_all_experiments(
    instance: Instance,
    profile: ApprovalProfile,
    k: int,
    *,
    setups: Iterable[str] = SETUPS,
    predictors: Iterable[str] = tuple(PREDICTORS),
    sample_degrees: Iterable[float] = SAMPLE_DEGREES,
    lv_degrees: Iterable[float] = LV_DEGREES,
    n_repeat: int = 50,
    seed: int | None = None,
) -> dict[tuple[float, float, str, str], dict[str, float]]:
    """
    The paper's full treatment matrix (Section 6, Figure 4). For every cell -
    setup x predictor x sample_degree x lv_degree - the ideal ``profile`` is
    split into LV/TV with :py:func:`split_lv_tv`, the pipeline is run with
    :py:func:`run_pipeline`, and the predicted bundle is compared to the *real*
    bundle (greedy approval on the whole ideal profile). Because the split is
    random, each cell is repeated ``n_repeat`` times and the two bundle metrics -
    Fractional Allocation (:py:func:`fractional_allocation_score`) and the
    Symmetric Distance ``|rb △ pb|`` (Section 5.2.1, computed inline) - are
    averaged.

    .. note::
        The paper repeats the sampling module 20 times and the prediction module
        50 times; ``n_repeat`` collapses both into one knob (default 50). The full
        grid is ``len(setups) * len(predictors) * len(sample_degrees) *
        len(lv_degrees)`` cells, so shrink the iterables or ``n_repeat`` for a
        quick run. ``classification`` needs ``scikit-learn`` installed.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The ideal instance's full ballots (all voters).
        k : int
            The number of projects to expose per TV voter.
        setups : Iterable[str], optional
            The setups to sweep (names from :py:data:`SETUPS`).
        predictors : Iterable[str], optional
            The predictors to sweep (names from :py:data:`PREDICTORS`). Defaults to
            all three paper predictors (classification / MF / FM).
        sample_degrees, lv_degrees : Iterable[float], optional
            The partiality grid (Section 3.0.1). Default to the paper's ranges.
        n_repeat : int, optional
            Random splits averaged per cell (default 50).
        seed : int, optional
            Seed for the whole sweep, for reproducibility.

    Returns
    -------
        dict[tuple[float, float, str, str], dict[str, float]]
            Keyed by ``(sample_degree, lv_degree, setup, predictor)``, each value
            is ``{"FA": mean fractional allocation, "SD": mean symmetric distance}``.

    Examples
    --------
    Example 10 from the paper - the "perfect" case, swept over a tiny grid with
    the two ML-free recommendation predictors. Every cell yields an FA in [0, 1]
    and a non-negative SD.

    >>> p1, p2, p3, p4 = (Project("p1", 3), Project("p2", 3),
    ...                   Project("p3", 4), Project("p4", 4))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=6)
    >>> prof = ApprovalProfile([ApprovalBallot([p1, p2])] * 4)
    >>> results = run_all_experiments(
    ...     inst, prof, k=1,
    ...     predictors=("matrix_factorization", "factorization_machines"),
    ...     sample_degrees=(0.5,), lv_degrees=(0.5,), n_repeat=2, seed=0)
    >>> len(results) == 5 * 2  # 5 setups x 2 predictors x 1 x 1 cells
    True
    >>> all(0.0 <= c["FA"] <= 1.0 and c["SD"] >= 0 for c in results.values())
    True
    """
    # The real bundle: greedy approval on the whole ideal profile (all voters).
    real_bundle = set(greedy_approval(instance, profile))
    rng = random.Random(seed)
    results: dict[tuple[float, float, str, str], dict[str, float]] = {}
    for sample_degree in sample_degrees:
        for lv_degree in lv_degrees:
            for setup in setups:
                for name in predictors:
                    predict = PREDICTORS[name]
                    fa_sum = sd_sum = 0.0
                    for _ in range(n_repeat):
                        lv_profile, tv_ballots = split_lv_tv(
                            profile, sample_degree, lv_degree,
                            seed=rng.randrange(2**32),
                        )
                        predicted = set(
                            run_pipeline(
                                instance, lv_profile, tv_ballots, k,
                                setup=setup, predict=predict,
                                seed=rng.randrange(2**32),
                            )
                        )
                        fa_sum += fractional_allocation_score(
                            real_bundle, predicted, instance.budget_limit
                        )
                        # Symmetric Distance (Section 5.2.1): |rb △ pb|.
                        sd_sum += len(real_bundle ^ predicted)
                    cell = {"FA": fa_sum / n_repeat, "SD": sd_sum / n_repeat}
                    results[(sample_degree, lv_degree, setup, name)] = cell
                    logger.info(
                        "run_all_experiments: sample=%.2f lv=%.2f setup=%s "
                        "predict=%s -> FA=%.3f SD=%.2f",
                        sample_degree, lv_degree, setup, name,
                        cell["FA"], cell["SD"],
                    )
    return results


# ---------------------------------------------------------------------------
# Section 5.1 - Classification accuracy metrics.
# ---------------------------------------------------------------------------
def classification_metrics(
    real_approved: set[Project],
    predicted_approved: set[Project],
    hidden: set[Project],
) -> dict[str, float]:
    """
    Section 5.1 (Classification Accuracy Metrics): precision, recall and F1 of a
    prediction module, measured over one Target Voter's *hidden* projects (the
    test set - the exposed votes are known, not predicted, so they are excluded).
    Approval is the positive class, so from the confusion matrix over the hidden
    projects:

    * TP = hidden projects the voter really approves and we predicted approve,
    * FP = hidden projects we predicted approve but she really rejects,
    * FN = hidden projects she really approves but we predicted reject,

    then ``precision = TP / (TP + FP)``, ``recall = TP / (TP + FN)`` and
    ``F1 = 2 * precision * recall / (precision + recall)``. A denominator of 0
    (no predicted or no real approvals among the hidden projects) yields 0.0 for
    that metric, the usual convention for an undefined score.

    Parameters
    ----------
        real_approved : set[:py:class:`~pabutools.election.instance.Project`]
            The projects the voter really approves (her ideal ballot A_v).
        predicted_approved : set[:py:class:`~pabutools.election.instance.Project`]
            The projects the prediction module marked as approved.
        hidden : set[:py:class:`~pabutools.election.instance.Project`]
            The voter's hidden set H_v, i.e. the projects scored (from
            :py:func:`hidden_projects`).

    Returns
    -------
        dict[str, float]
            ``{"precision": ..., "recall": ..., "f1": ...}``, each in [0, 1].

    Examples
    --------
    Four hidden projects, the voter really approves {p1, p2}, the model predicts
    {p1, p3}: one hit (p1), one false alarm (p3), one miss (p2), so precision =
    recall = F1 = 0.5. A perfect prediction scores 1.0 across the board.

    >>> p1, p2, p3, p4 = (Project("p1", 1), Project("p2", 1),
    ...                   Project("p3", 1), Project("p4", 1))
    >>> hidden = {p1, p2, p3, p4}
    >>> classification_metrics({p1, p2}, {p1, p3}, hidden)
    {'precision': 0.5, 'recall': 0.5, 'f1': 0.5}
    >>> classification_metrics({p1, p2}, {p1, p2}, hidden)
    {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    """
    # Restrict everything to the hidden projects: the exposed votes are known.
    real = real_approved & hidden
    predicted = predicted_approved & hidden
    tp = len(real & predicted)
    fp = len(predicted - real)
    fn = len(real - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    logger.info(
        "classification_metrics: TP=%d FP=%d FN=%d -> P=%.3f R=%.3f F1=%.3f",
        tp, fp, fn, precision, recall, f1,
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Section 5.2 - Bundle evaluation metrics.
# ---------------------------------------------------------------------------
# The Symmetric Distance (Section 5.2.1), |rb △ pb|, is a one-liner
# ``len(real_bundle ^ predicted_bundle)`` computed inline where needed (see
# ``run_all_experiments``), so it gets no function of its own.
def fractional_allocation_score(
    real_bundle: Iterable[Project],
    predicted_bundle: Iterable[Project],
    budget_limit: int,
) -> float:
    """
    Definition 5.1 (Fractional Allocation score): the total cost of the
    projects predicted correctly (those in both bundles) divided by the budget
    limit, FA = lambda / B with lambda = sum of cost(p) over p in pb ∩ rb.

    Parameters
    ----------
        real_bundle : Iterable[:py:class:`~pabutools.election.instance.Project`]
            The bundle obtained from the real (full) ballots - any iterable of
            projects, e.g. the :py:class:`~pabutools.rules.budgetallocation.BudgetAllocation`
            returned by :py:func:`greedy_approval`, as is.
        predicted_bundle : Iterable[:py:class:`~pabutools.election.instance.Project`]
            The bundle obtained from the predicted ballots (same, any iterable).
        budget_limit : int
            The budget limit B of the instance.

    Returns
    -------
        float
            The fractional allocation score, in [0, 1].

    Examples
    --------
    Example 10: pb = rb = {p1, p2}, costs 3 and 3, budget 6, so FA = 6/6 = 1.0.
    Example 11: disjoint bundles, so FA = 0/6 = 0.0.

    >>> p1, p2, p3 = Project("p1", 3), Project("p2", 3), Project("p3", 6)
    >>> fractional_allocation_score({p1, p2}, {p1, p2}, budget_limit=6)
    1.0
    >>> fractional_allocation_score({p3}, {p1}, budget_limit=6)
    0.0
    """
    real_bundle, predicted_bundle = set(real_bundle), set(predicted_bundle)
    # lambda = total cost of the correctly-predicted projects (pb ∩ rb).
    correctly_predicted = real_bundle & predicted_bundle
    score = total_cost(correctly_predicted) / budget_limit
    logger.info(
        "fractional_allocation_score: %d/%d projects correct, FA=%.3f",
        len(correctly_predicted), len(real_bundle | predicted_bundle), score,
    )
    return float(score)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
