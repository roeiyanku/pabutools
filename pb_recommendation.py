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

from pabutools.election import (
    Instance,
    Project,
    ApprovalProfile,
    ApprovalBallot,
    CardinalBallot,
    total_cost,
)
from pabutools.rules import BudgetAllocation

# Model training lives in a separate module, ``pb_model_training`` (the
# professor's note b: separate files for training the model and using it). The
# learning-based predictors below delegate the fitting to its ``train_*``
# functions; ``pb_model_training`` is ballot-agnostic (it takes plain project
# sets), so the dependency is one-directional and there is no import cycle.
from pb_model_training import (
    train_classification,
    train_matrix_factorization,
    train_factorization_machines,
)

# Log the steps of every algorithm at INFO/DEBUG level (see the assignment's
# logging requirement). Configure a handler in your own script to see them, e.g.
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
    scores: dict[Project, int] = {}
    for project in approved:
        scores[project] = APPROVAL
    for project in disapproved:
        scores[project] = DISAPPROVAL
    for project in hidden:
        scores[project] = HIDDEN
    return CardinalBallot(scores)


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
    approved = exposed & full_ballot          # A_v = E_v ∩ full ballot
    disapproved = exposed - full_ballot        # D_v = E_v \ full ballot
    hidden = set(instance) - exposed           # H_v = P \ E_v
    logger.debug(
        "reveal_ballot: |E_v|=%d -> |A_v|=%d, |D_v|=%d, |H_v|=%d",
        len(exposed), len(approved), len(disapproved), len(hidden),
    )
    return partial_ballot(approved=approved, disapproved=disapproved, hidden=hidden)


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
def approval_scores(
    instance: Instance, profile: ApprovalProfile
) -> dict[Project, int]:
    """
    Definition 2.1 (Approval scores): the approval score of a project is the
    number of voters that approve it.

    This is exactly pabutools'
    :py:meth:`~pabutools.election.profile.approvalprofile.AbstractApprovalProfile.approval_scores`,
    so we delegate to the library instead of re-counting the ballots ourselves.
    The only addition is that projects approved by nobody (absent from the
    library's dictionary) are filled in with a score of 0, so every project of
    the instance is present in the result.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance (the set of projects and the budget limit).
        profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The approval profile, i.e. the ballots of the voters.

    Returns
    -------
        dict[:py:class:`~pabutools.election.instance.Project`, int]
            A mapping from each project to its approval score.

    Examples
    --------
    Example 1 from the paper (P = {p1, p2, p3}, three full ballots):

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 2)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> prof = ApprovalProfile([ApprovalBallot([p1, p2]),
    ...                         ApprovalBallot([p1, p3]),
    ...                         ApprovalBallot([p2])])
    >>> s = approval_scores(inst, prof)
    >>> [s[p] for p in (p1, p2, p3)]
    [2, 2, 1]
    """
    # Reuse the library's counting (Definition 2.1), then fill 0 for the
    # projects that nobody approved so the whole universe P is represented.
    library_scores = profile.approval_scores()
    scores = {project: library_scores.get(project, 0) for project in instance}
    logger.debug("approval_scores: %s", scores)
    return scores


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
    # |score - (n - score)| = |2*score - n|.
    n = profile.num_ballots()
    scores = approval_scores(instance, profile)
    consensus = {project: abs(2 * scores[project] - n) for project in instance}
    logger.debug("consensus_levels (n=%d): %s", n, consensus)
    return consensus


def most_popular_projects(
    instance: Instance, profile: ApprovalProfile
) -> list[Project]:
    """
    The paper's ordering sigma (Section 2.2.3): the projects sorted by decreasing
    approval score, ties broken lexicographically by project name. sigma_1 is the
    most popular project, sigma_m the least. Helper used by the greedy rule and by
    offline revealing-by-popularity.

    Examples
    --------
    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> prof = ApprovalProfile([ApprovalBallot([p1, p3]), ApprovalBallot([p1])])
    >>> most_popular_projects(inst, prof)  # scores p1=2, p3=1, p2=0
    [p1, p3, p2]
    """
    scores = approval_scores(instance, profile)
    return sorted(instance, key=lambda project: (-scores[project], str(project)))


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
        Implemented from scratch; it does *not* reuse pabutools'
        :py:func:`~pabutools.rules.greedy_utilitarian_welfare`, which ranks by
        satisfaction **divided by cost** (density). The paper ranks by the
        **raw** approval score, so the two give different bundles.

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
    # Consider the projects by decreasing approval score (sigma) and fund each
    # one whose cost still fits the remaining budget (Section 2.2.5).
    chosen = BudgetAllocation()
    spent = 0
    for project in most_popular_projects(instance, profile):
        if spent + project.cost <= instance.budget_limit:
            chosen.append(project)
            spent += project.cost
            logger.debug("greedy_approval: fund %s (spent %s)", project, spent)
        else:
            logger.debug("greedy_approval: skip %s (would exceed budget)", project)
    logger.info(
        "greedy_approval: funded %d/%d projects, cost %s of %s",
        len(chosen), len(instance), spent, instance.budget_limit,
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
    exposed = set(most_popular_projects(instance, lv_profile)[:k])
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
        remaining = [project for project in instance if project not in exposed]
        # Most controversial = lowest consensus, ties broken by project name.
        project = min(remaining, key=lambda p: (consensus[p], str(p)))
        answered_yes = project in full_ballot  # simulate the voter's answer
        logger.info(
            "online_adaptive_controversial: round %d asks %s -> voter %s",
            round_index, project, "approves" if answered_yes else "disapproves",
        )
        exposed.add(project)
    return exposed


# ---------------------------------------------------------------------------
# Section 2.1 / examples - Prediction module (majority rule over LV).
# ---------------------------------------------------------------------------
def predict_by_majority(
    instance: Instance,
    lv_profile: ApprovalProfile,
    ballot: CardinalBallot,
) -> ApprovalBallot:
    """
    Prediction module (Section 2.1). Completes a single partial TV ballot into a
    full approval ballot: every project in the hidden set H_v is predicted as
    approved iff its approval rate among the LV voters is at least 50%. The
    exposed approvals A_v are kept; the exposed disapprovals D_v stay rejected.

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
    nothing exposed (all three projects hidden) is predicted to approve p1 (100%)
    and p2 (100%) but not p3 (0%).

    >>> p1, p2, p3 = Project("p1", 4), Project("p2", 4), Project("p3", 6)
    >>> inst = Instance([p1, p2, p3], budget_limit=6)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1, p2])])
    >>> partial = partial_ballot(hidden={p1, p2, p3})
    >>> predict_by_majority(inst, lv, partial) == {p1, p2}
    True
    """
    n = lv_profile.num_ballots()
    scores = approval_scores(instance, lv_profile)
    result = set(approved_projects(ballot))  # keep the exposed approvals A_v
    for project in hidden_projects(ballot, instance):
        # LV approval rate >= 50% <=> score >= n/2 <=> 2*score >= n.
        if 2 * scores[project] >= n:
            result.add(project)
            logger.debug(
                "predict_by_majority: predict %s approved (LV %d/%d)",
                project, scores[project], n,
            )
    logger.info("predict_by_majority: completed ballot to %d approvals", len(result))
    return ApprovalBallot(result)


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
    features = model["features"]
    # The TV voter's feature vector: her vote on each exposed project.
    approved = approved_projects(ballot)
    tv_features = [1 if feature in approved else 0 for feature in features]
    result = set(approved)  # keep the exposed approvals A_v
    for project in hidden_projects(ballot, instance):
        kind, payload = model["per_project"][project]
        if kind == "const":
            approve = payload
        else:
            import numpy as np
            approve = int(payload.predict(np.array([tv_features]))[0])
        if approve == 1:
            result.add(project)
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
    result = set(approved_projects(ballot))
    for project in hidden_projects(ballot, instance):
        if scores[project] >= 0.5:
            result.add(project)
    return result


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
def recommend(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_ballots: dict[str, set[Project]],
    k: int,
) -> BudgetAllocation:
    """
    The complete Section 3 pipeline with the offline-popularity sampler: expose
    the k most popular LV projects to each TV voter, predict the hidden votes by
    LV majority, then run greedy approval on the LV plus completed TV ballots.

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
    >>> sorted(recommend(inst, lv, {"v4": {p1, p2}}, k=1), key=str)
    [p1, p2]
    """
    # Sampling: the offline-popularity sampler exposes the same k projects to
    # every Target Voter.
    exposed = offline_popularity(instance, lv_profile, k)
    # Prediction: complete each TV ballot; start the combined profile from the LV
    # ballots (which are already full).
    completed_ballots = list(lv_profile)
    for voter_id, full_ballot in tv_ballots.items():
        partial = reveal_ballot(instance, full_ballot, exposed)
        completed = predict_by_majority(instance, lv_profile, partial)
        logger.debug("recommend: TV %s completed to %d approvals", voter_id, len(completed))
        completed_ballots.append(completed)
    # Voting rule: greedy approval on the LV + completed TV ballots.
    combined = ApprovalProfile(completed_ballots)
    logger.info(
        "recommend: predicting bundle from %d LV + %d TV ballots",
        lv_profile.num_ballots(), len(tv_ballots),
    )
    return greedy_approval(instance, combined)


# ---------------------------------------------------------------------------
# Section 5.2 - Bundle evaluation metrics.
# ---------------------------------------------------------------------------
def fractional_allocation_score(
    real_bundle: set[Project], predicted_bundle: set[Project], budget_limit: int
) -> float:
    """
    Definition 5.1 (Fractional Allocation score): the total cost of the
    projects predicted correctly (those in both bundles) divided by the budget
    limit, FA = lambda / B with lambda = sum of cost(p) over p in pb ∩ rb.

    Parameters
    ----------
        real_bundle : set[:py:class:`~pabutools.election.instance.Project`]
            The bundle obtained from the real (full) ballots.
        predicted_bundle : set[:py:class:`~pabutools.election.instance.Project`]
            The bundle obtained from the predicted ballots.
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
