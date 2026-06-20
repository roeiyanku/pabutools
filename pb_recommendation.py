"""
An implementation of the algorithms in:
"A Recommendation System for Participatory Budgeting",
by Gil Leibiker and Nimrod Talmon (2023), https://optlearnmas23.github.io/files/p17.pdf

Programmer: Roei Yanku
Date: 2026-06-20.
"""

from __future__ import annotations

from pabutools.election import (
    Instance,
    Project,
    ApprovalProfile,
    ApprovalBallot,
)
from pabutools.rules import BudgetAllocation


# ---------------------------------------------------------------------------
# Section 2.3 - Partial ballots (three-state approval ballots).
# ---------------------------------------------------------------------------
class PartialApprovalBallot:
    """
    A three-state approval ballot (Section 2.3), splitting the projects into the
    approval set A_v (``approved``), the disapproval set D_v (``disapproved``)
    and the unknown set H_v (``hidden``). They partition P (A_v ∪ D_v ∪ H_v = P)
    and the exposed set is E_v = A_v ∪ D_v. pabutools' ApprovalBallot cannot
    express this (no disapproved/unknown distinction), so this helper class is
    added here.

    Parameters
    ----------
        approved : Iterable[:py:class:`~pabutools.election.instance.Project`], optional
            The projects the voter approves (A_v). Defaults to ``()``.
        disapproved : Iterable[:py:class:`~pabutools.election.instance.Project`], optional
            The projects the voter disapproves (D_v). Defaults to ``()``.
        hidden : Iterable[:py:class:`~pabutools.election.instance.Project`], optional
            The projects whose preference is unknown (H_v). Defaults to ``()``.

    Attributes
    ----------
        approved : set[:py:class:`~pabutools.election.instance.Project`]
            The approval set A_v.
        disapproved : set[:py:class:`~pabutools.election.instance.Project`]
            The disapproval set D_v.
        hidden : set[:py:class:`~pabutools.election.instance.Project`]
            The hidden set H_v.

    Examples
    --------
    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> b = PartialApprovalBallot(approved={p1}, disapproved={p2}, hidden={p3})
    >>> b.exposed == {p1, p2}
    True
    >>> (b.approved | b.disapproved | b.hidden) == {p1, p2, p3}
    True
    """

    def __init__(self, approved=(), disapproved=(), hidden=()) -> None:
        self.approved: set[Project] = set()  # Empty implementation
        self.disapproved: set[Project] = set()  # Empty implementation
        self.hidden: set[Project] = set()  # Empty implementation

    @property
    def exposed(self) -> set[Project]:
        """The exposed set E_v = A_v ∪ D_v (the projects the voter answered)."""
        return set()  # Empty implementation

    def as_approval_ballot(self) -> ApprovalBallot:
        """
        The pabutools approval ballot made of the approved projects A_v, so that
        a (completed) partial ballot can be fed to approval-based voting rules.
        """
        return ApprovalBallot()  # Empty implementation

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, PartialApprovalBallot)
            and self.approved == other.approved
            and self.disapproved == other.disapproved
            and self.hidden == other.hidden
        )

    def __repr__(self) -> str:
        return (
            f"PartialApprovalBallot(approved={sorted(map(str, self.approved))}, "
            f"disapproved={sorted(map(str, self.disapproved))}, "
            f"hidden={sorted(map(str, self.hidden))})"
        )


def reveal_ballot(
    instance: Instance,
    true_approval: set[Project],
    exposed: set[Project],
) -> PartialApprovalBallot:
    """
    Build the partial ballot exposing a set of projects of a voter whose true
    approval set is known (Section 2.3): A_v = exposed ∩ true,
    D_v = exposed \\ true, H_v = P \\ exposed.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance (used as the universe of projects P).
        true_approval : set[:py:class:`~pabutools.election.instance.Project`]
            The projects the voter truly approves.
        exposed : set[:py:class:`~pabutools.election.instance.Project`]
            The projects revealed for this voter (E_v).

    Returns
    -------
        PartialApprovalBallot
            The corresponding three-state ballot.

    Examples
    --------
    Example 2.4 from the paper: P = {p1, p2, p3, p4}, the voter truly approves
    {p1, p2}, and {p1, p3} is exposed. Then p1 is approved, p3 disapproved, and
    p2, p4 remain hidden.

    >>> p1, p2, p3, p4 = (Project("p1", 1), Project("p2", 1),
    ...                   Project("p3", 1), Project("p4", 1))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=4)
    >>> b = reveal_ballot(inst, {p1, p2}, {p1, p3})
    >>> b.approved == {p1}, b.disapproved == {p3}, b.hidden == {p2, p4}
    (True, True, True)
    """
    return PartialApprovalBallot()  # Empty implementation


# ---------------------------------------------------------------------------
# Section 2.2.3 - Popularity and consensus (primitives used by every module).
# ---------------------------------------------------------------------------
def approval_scores(
    instance: Instance, profile: ApprovalProfile
) -> dict[Project, int]:
    """
    Definition 2.1 (Approval scores): the approval score of a project is the
    number of voters that approve it.

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
    return {}  # Empty implementation


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
    return {}  # Empty implementation


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
    return BudgetAllocation()  # Empty implementation


# ---------------------------------------------------------------------------
# Section 3.1.1 - Random setup.
# ---------------------------------------------------------------------------
def random_setup(
    instance: Instance,
    tv_ballots: dict[str, set[Project]],
    k: int,
    seed: int | None = None,
) -> dict[str, set[Project]]:
    """
    Algorithm 1 - Random setup (Section 3.1.1): expose k projects chosen
    uniformly at random from P for each TV voter (the rest is predicted later).

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        tv_ballots : dict[str, set[:py:class:`~pabutools.election.instance.Project`]]
            The true (hidden) approval set of every TV voter, keyed by voter id.
            Projects outside the set are disapproved by that voter.
        k : int
            The number of projects to expose per voter.
        seed : int, optional
            Seed for the random generator, for reproducibility.

    Returns
    -------
        dict[str, set[:py:class:`~pabutools.election.instance.Project`]]
            For each TV voter, the set of k exposed projects.

    Examples
    --------
    Example 5 from the paper: a single TV voter v4, k = 2. Whatever the draw,
    exactly two projects are exposed and they are real projects of the instance.

    >>> p1, p2, p3, p4 = (Project("p1", 2), Project("p2", 2),
    ...                   Project("p3", 3), Project("p4", 3))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=6)
    >>> exposed = random_setup(inst, {"v4": {p1, p2}}, k=2, seed=0)
    >>> len(exposed["v4"]) == 2 and exposed["v4"] <= {p1, p2, p3, p4}
    True
    """
    return {}  # Empty implementation


# ---------------------------------------------------------------------------
# Section 3.1.2 - Offline setup (popularity / consensus / controversiality).
# ---------------------------------------------------------------------------
def offline_popularity(
    instance: Instance, lv_profile: ApprovalProfile, k: int
) -> list[Project]:
    """
    Algorithm 2 - Offline revealing by popularity (Section 3.1.2). Exposes, for
    every TV voter, the k most approved projects among the LV voters, i.e.
    {sigma_1, ..., sigma_k}.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters.
        k : int
            The number of projects to expose.

    Returns
    -------
        list[:py:class:`~pabutools.election.instance.Project`]
            The k most popular projects, ordered by score (ties by name).

    Examples
    --------
    Example 6 from the paper: LV scores p1 = 3, p2 = 1, p3 = 1, so the single
    most popular project is p1.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p3]),
    ...                       ApprovalBallot([p1, p2]),
    ...                       ApprovalBallot([p1])])
    >>> offline_popularity(inst, lv, k=1)
    [p1]
    """
    return []  # Empty implementation


def offline_consensus(
    instance: Instance, lv_profile: ApprovalProfile, k: int
) -> list[Project]:
    """
    Algorithm 3 - Offline revealing by consensus (Section 3.1.2). Exposes the k
    projects with the highest consensus level among the LV voters, i.e.
    {gamma_1, ..., gamma_k}.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters.
        k : int
            The number of projects to expose.

    Returns
    -------
        list[:py:class:`~pabutools.election.instance.Project`]
            The k projects most in consensus, ordered by level (ties by name).

    Examples
    --------
    Example 7 from the paper: p1 and p3 both have consensus 4 (p1 unanimously
    approved, p3 unanimously rejected); the tie is broken by name towards p1.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1]),
    ...                       ApprovalBallot([p1, p2]), ApprovalBallot([p1])])
    >>> offline_consensus(inst, lv, k=1)
    [p1]
    """
    return []  # Empty implementation


def offline_controversiality(
    instance: Instance, lv_profile: ApprovalProfile, k: int
) -> list[Project]:
    """
    Algorithm 4 - Offline revealing by controversiality (Section 3.1.2): expose
    the k projects *least* in consensus among the LV voters,
    {gamma_{m-k+1}, ..., gamma_m} (the hardest to predict, so asked directly).

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters.
        k : int
            The number of projects to expose.

    Returns
    -------
        list[:py:class:`~pabutools.election.instance.Project`]
            The k most controversial projects, ordered by increasing consensus.

    Examples
    --------
    Example 8 from the paper (same data as Example 7): p2 is split exactly in
    half (consensus 0) and is therefore the single most controversial project.

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> inst = Instance([p1, p2, p3], budget_limit=3)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1]),
    ...                       ApprovalBallot([p1, p2]), ApprovalBallot([p1])])
    >>> offline_controversiality(inst, lv, k=1)
    [p2]
    """
    return []  # Empty implementation


# ---------------------------------------------------------------------------
# Section 3.1.3 - Online setup (adaptive controversial).
# ---------------------------------------------------------------------------
def online_adaptive_controversial(
    instance: Instance,
    lv_profile: ApprovalProfile,
    tv_ballot: set[Project],
    k: int,
) -> list[Project]:
    """
    Algorithm 5 - Online adaptive-controversial setup (Section 3.1.3): in each of
    k iterations recompute the most controversial project given the LV ballots
    and the answers already revealed, then ask about it (each answer affects the
    next question).

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters.
        tv_ballot : set[:py:class:`~pabutools.election.instance.Project`]
            The true (hidden) approval set of the TV voter being queried.
        k : int
            The number of iterations / projects to expose.

    Returns
    -------
        list[:py:class:`~pabutools.election.instance.Project`]
            The k exposed projects, in the order they were asked.

    Examples
    --------
    Example 9 from the paper: 4 LV voters, one TV voter v5, k = 2. p1, p2, p3
    are all tied as most controversial; the first question (tie broken by name)
    is p1, and after the voter's answer the second question is p2.

    >>> p1, p2, p3, p4 = (Project("p1", 1), Project("p2", 1),
    ...                   Project("p3", 1), Project("p4", 1))
    >>> inst = Instance([p1, p2, p3, p4], budget_limit=4)
    >>> lv = ApprovalProfile([ApprovalBallot([p1, p2]), ApprovalBallot([p1, p3]),
    ...                       ApprovalBallot([p2, p3]), ApprovalBallot([])])
    >>> online_adaptive_controversial(inst, lv, {p1, p2}, k=2)
    [p1, p2]
    """
    return []  # Empty implementation


# ---------------------------------------------------------------------------
# Section 2.1 / examples - Prediction module (majority rule over LV).
# ---------------------------------------------------------------------------
def predict_by_majority(
    instance: Instance,
    lv_profile: ApprovalProfile,
    ballot: PartialApprovalBallot,
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
        ballot : PartialApprovalBallot
            The partial ballot of the TV voter to complete.

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
    >>> partial = PartialApprovalBallot(hidden={p1, p2, p3})
    >>> predict_by_majority(inst, lv, partial) == {p1, p2}
    True
    """
    return ApprovalBallot()  # Empty implementation


def predict_by_classification(
    instance: Instance,
    lv_profile: ApprovalProfile,
    ballot: PartialApprovalBallot,
) -> ApprovalBallot:
    """
    Prediction module - binary classification (Section 2.1.1): predict each
    hidden project with a per-project binary classifier trained on the LV ballots
    (features = votes on the exposed projects, label = vote on the target), then
    applied to the TV voter. Exposed approvals A_v are kept and exposed
    disapprovals D_v stay rejected. Backed by the external ``xgboost`` library
    (:py:class:`xgboost.XGBClassifier`, class-weighted loss for the imbalanced
    data), imported inside the implementation.

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters, used as the training data.
        ballot : PartialApprovalBallot
            The partial ballot of the TV voter to complete.

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
    >>> partial = PartialApprovalBallot(hidden={p1, p2, p3})
    >>> predict_by_classification(inst, lv, partial) == {p1, p2}
    True
    """
    return ApprovalBallot()  # Empty implementation


def predict_by_matrix_factorization(
    instance: Instance,
    lv_profile: ApprovalProfile,
    ballot: PartialApprovalBallot,
) -> ApprovalBallot:
    """
    Prediction module - collaborative filtering via Matrix Factorization
    (Section 2.1.2): build the sparse user-item matrix from the LV ballots and
    exposed TV votes (approve=1, disapprove=0), factorise it, and predict a
    hidden project as approved iff its reconstructed score is >= 0.5. Exposed
    approvals A_v are kept and exposed disapprovals D_v stay rejected. Backed by
    the external ``scikit-surprise`` library (:py:class:`surprise.SVD`).

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters, used as the training data.
        ballot : PartialApprovalBallot
            The partial ballot of the TV voter to complete.

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
    >>> partial = PartialApprovalBallot(hidden={p1, p2, p3, p4})
    >>> predict_by_matrix_factorization(inst, lv, partial) == {p1, p2}
    True
    """
    return ApprovalBallot()  # Empty implementation


def predict_by_factorization_machines(
    instance: Instance,
    lv_profile: ApprovalProfile,
    ballot: PartialApprovalBallot,
) -> ApprovalBallot:
    """
    Prediction module - hybrid Factorization Machines (Section 2.1.2): like MF
    but with a linear term plus pairwise latent interactions and optional side
    features, predicting a hidden project as approved iff the FM score is >= 0.5.
    Exposed approvals A_v are kept and exposed disapprovals D_v stay rejected.
    Backed by an external FM library (e.g. ``lightfm`` / ``fastFM``).

    Parameters
    ----------
        instance : :py:class:`~pabutools.election.instance.Instance`
            The PB instance.
        lv_profile : :py:class:`~pabutools.election.profile.approvalprofile.ApprovalProfile`
            The full ballots of the LV voters, used as the training data.
        ballot : PartialApprovalBallot
            The partial ballot of the TV voter to complete.

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
    >>> partial = PartialApprovalBallot(hidden={p1, p2, p3})
    >>> predict_by_factorization_machines(inst, lv, partial) == {p1, p2}
    True
    """
    return ApprovalBallot()  # Empty implementation


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
            The true (hidden) approval set of every TV voter, keyed by voter id.
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
    return BudgetAllocation()  # Empty implementation


# ---------------------------------------------------------------------------
# Section 5.2 - Bundle evaluation metrics.
# ---------------------------------------------------------------------------
def symmetric_distance(
    real_bundle: set[Project], predicted_bundle: set[Project]
) -> int:
    """
    Section 5.2.1 - Symmetric distance between the real winning bundle and the
    predicted one: the size of their symmetric difference.

    Parameters
    ----------
        real_bundle : set[:py:class:`~pabutools.election.instance.Project`]
            The bundle obtained from the real (full) ballots.
        predicted_bundle : set[:py:class:`~pabutools.election.instance.Project`]
            The bundle obtained from the predicted ballots.

    Returns
    -------
        int
            The number of projects in exactly one of the two bundles.

    Examples
    --------
    Example 10 (a perfect prediction) and Example 11 (a complete failure):

    >>> p1, p2, p3 = Project("p1", 1), Project("p2", 1), Project("p3", 1)
    >>> symmetric_distance({p1, p2}, {p1, p2})
    0
    >>> symmetric_distance({p1}, {p3})
    2
    """
    return 0  # Empty implementation


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
    return 0.0  # Empty implementation


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
