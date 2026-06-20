"""
Unit tests for `pb_recommendation`, the implementation of the algorithms in
"A Recommendation System for Participatory Budgeting" (Leibiker & Talmon, 2023).

Run with:  pytest test_pb_recommendation.py -v

Each test is the specification its function must satisfy. They come in three
kinds:
    * small, hand-checked instances (the examples from the paper);
    * edge cases (empty profile, single project, full ballots, k bounds);
    * large inputs, checked against an independent reference or via invariants.

Programmer: Roei Yanku
Date: 2026-06-20.
"""

import random
from collections import Counter

import pytest

from pabutools.election import (
    Instance,
    Project,
    ApprovalProfile,
    ApprovalBallot,
)

from pb_recommendation import (
    PartialApprovalBallot,
    reveal_ballot,
    approval_scores,
    consensus_levels,
    greedy_approval,
    random_setup,
    offline_popularity,
    offline_consensus,
    offline_controversiality,
    online_adaptive_controversial,
    predict_by_majority,
    predict_by_classification,
    predict_by_matrix_factorization,
    predict_by_factorization_machines,
    recommend,
    symmetric_distance,
    fractional_allocation_score,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
def make_projects(specs):
    """specs: list of (name, cost) -> dict name -> Project."""
    return {name: Project(name, cost) for name, cost in specs}


@pytest.fixture
def example1():
    """P = {p1, p2, p3}, costs 1,1,2, B = 3; v1={p1,p2}, v2={p1,p3}, v3={p2}."""
    p = make_projects([("p1", 1), ("p2", 1), ("p3", 2)])
    inst = Instance(p.values(), budget_limit=3)
    prof = ApprovalProfile(
        [
            ApprovalBallot([p["p1"], p["p2"]]),
            ApprovalBallot([p["p1"], p["p3"]]),
            ApprovalBallot([p["p2"]]),
        ]
    )
    return p, inst, prof


@pytest.fixture
def consensus_data():
    """4 LV voters: p1 approved 4/0, p2 split 2/2, p3 rejected 0/4."""
    p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
    inst = Instance(p.values(), budget_limit=3)
    lv = ApprovalProfile(
        [
            ApprovalBallot([p["p1"], p["p2"]]),
            ApprovalBallot([p["p1"]]),
            ApprovalBallot([p["p1"], p["p2"]]),
            ApprovalBallot([p["p1"]]),
        ]
    )
    return p, inst, lv


def random_instance(n_projects, n_voters, budget, seed):
    """Build a random instance + approval profile for property testing."""
    rng = random.Random(seed)
    projects = [Project(f"p{i}", rng.randint(1, 10)) for i in range(n_projects)]
    inst = Instance(projects, budget_limit=budget)
    ballots = []
    for _ in range(n_voters):
        approved = [p for p in projects if rng.random() < 0.3]
        ballots.append(ApprovalBallot(approved))
    return projects, inst, ApprovalProfile(ballots)


def padded_projects(m, cost):
    """m unit-named projects p000, p001, ... so str-order == index-order."""
    width = len(str(m - 1))
    return [Project(f"p{i:0{width}d}", cost) for i in range(m)]


def manual_scores(projects, profile):
    """Independent (Counter-based) reimplementation of the approval scores."""
    counts = Counter()
    for ballot in profile:
        for proj in ballot:
            counts[proj] += 1
    return {proj: counts[proj] for proj in projects}


def reference_greedy_approval(projects, profile, budget):
    """
    Independent reference implementation of greedy approval, used to cross-check
    `greedy_approval` on random inputs (the "compare to another algorithm"
    strategy). Projects are taken in decreasing score, ties broken by name, and
    funded while they fit the budget.
    """
    counts = manual_scores(projects, profile)
    ordered = sorted(projects, key=lambda p: (-counts[p], str(p)))
    chosen, spent = [], 0
    for p in ordered:
        if spent + p.cost <= budget:
            chosen.append(p)
            spent += p.cost
    return chosen


# ---------------------------------------------------------------------------
# approval_scores
# ---------------------------------------------------------------------------
class TestApprovalScores:
    def test_example1(self, example1):
        p, inst, prof = example1
        scores = approval_scores(inst, prof)
        assert scores[p["p1"]] == 2
        assert scores[p["p2"]] == 2
        assert scores[p["p3"]] == 1

    def test_empty_profile(self):
        p = make_projects([("p1", 1), ("p2", 1)])
        inst = Instance(p.values(), budget_limit=2)
        scores = approval_scores(inst, ApprovalProfile([]))
        assert all(scores[proj] == 0 for proj in p.values())

    def test_large_structured_all_approve_all(self):
        # "Complete-approval" analogue: 200 projects, 500 voters, everyone
        # approves everything -> every score is exactly 500.
        projects = padded_projects(200, cost=1)
        inst = Instance(projects, budget_limit=10)
        prof = ApprovalProfile([ApprovalBallot(projects)] * 500)
        scores = approval_scores(inst, prof)
        assert all(scores[p] == 500 for p in projects)

    def test_matches_manual_count_random(self):
        # Cross-check against an independent counting implementation.
        projects, inst, prof = random_instance(40, 120, 100, seed=31)
        assert approval_scores(inst, prof) == manual_scores(projects, prof)


# ---------------------------------------------------------------------------
# consensus_levels
# ---------------------------------------------------------------------------
class TestConsensusLevels:
    def test_example7(self, consensus_data):
        p, inst, lv = consensus_data
        c = consensus_levels(inst, lv)
        assert c[p["p1"]] == 4
        assert c[p["p2"]] == 0
        assert c[p["p3"]] == 4

    def test_nonnegative_and_bounded_random(self):
        projects, inst, prof = random_instance(15, 40, 80, seed=2)
        c = consensus_levels(inst, prof)
        assert all(0 <= c[p] <= len(prof) for p in projects)

    def test_large_structured_even_split_is_zero(self):
        # Exactly half approve and half reject each project -> consensus 0.
        projects = padded_projects(20, cost=1)
        inst = Instance(projects, budget_limit=10)
        prof = ApprovalProfile(
            [ApprovalBallot(projects)] * 50 + [ApprovalBallot([])] * 50
        )
        c = consensus_levels(inst, prof)
        assert all(c[p] == 0 for p in projects)


# ---------------------------------------------------------------------------
# greedy_approval (voting rule)
# ---------------------------------------------------------------------------
class TestGreedyApproval:
    def test_example1(self, example1):
        p, inst, prof = example1
        bundle = set(greedy_approval(inst, prof))
        assert bundle == {p["p1"], p["p2"]}

    def test_skips_unaffordable_but_funds_affordable(self):
        # A project that exceeds the budget is skipped, while a cheaper one that
        # fits is funded.
        p = make_projects([("cheap", 10), ("dear", 100)])
        inst = Instance(p.values(), budget_limit=10)
        prof = ApprovalProfile([ApprovalBallot([p["cheap"], p["dear"]])])
        assert set(greedy_approval(inst, prof)) == {p["cheap"]}

    def test_large_structured_known_bundle(self):
        # 100 unit-cost projects, all approved by everyone, budget 30 -> exactly
        # the 30 name-first projects are funded (tie-break decides everything).
        projects = padded_projects(100, cost=1)
        inst = Instance(projects, budget_limit=30)
        prof = ApprovalProfile([ApprovalBallot(projects)] * 20)
        bundle = set(greedy_approval(inst, prof))
        assert bundle == set(projects[:30])

    def test_matches_reference_random(self):
        # Cross-check against the independent greedy reference, several seeds.
        for seed in range(5):
            projects, inst, prof = random_instance(30, 50, 40, seed=200 + seed)
            got = set(greedy_approval(inst, prof))
            expected = set(reference_greedy_approval(projects, prof, inst.budget_limit))
            assert got == expected


# ---------------------------------------------------------------------------
# random_setup
# ---------------------------------------------------------------------------
class TestRandomSetup:
    def test_exposes_exactly_k(self):
        p = make_projects([("p1", 2), ("p2", 2), ("p3", 3), ("p4", 3)])
        inst = Instance(p.values(), budget_limit=6)
        exposed = random_setup(inst, {"v4": set(p.values())}, k=2, seed=0)
        assert len(exposed["v4"]) == 2
        assert exposed["v4"] <= set(p.values())

    def test_all_tv_voters_covered(self):
        projects, inst, _ = random_instance(10, 1, 30, seed=5)
        tv = {f"v{i}": set() for i in range(5)}
        exposed = random_setup(inst, tv, k=3, seed=7)
        assert set(exposed.keys()) == set(tv.keys())
        assert all(len(exposed[v]) == 3 for v in tv)

    def test_k_zero(self):
        p = make_projects([("p1", 1), ("p2", 1)])
        inst = Instance(p.values(), budget_limit=2)
        exposed = random_setup(inst, {"v1": set()}, k=0, seed=0)
        assert exposed["v1"] == set()


# ---------------------------------------------------------------------------
# offline_popularity / consensus / controversiality
# ---------------------------------------------------------------------------
class TestOfflineSamplers:
    def test_popularity_example6(self):
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
        inst = Instance(p.values(), budget_limit=3)
        lv = ApprovalProfile(
            [
                ApprovalBallot([p["p1"], p["p3"]]),
                ApprovalBallot([p["p1"], p["p2"]]),
                ApprovalBallot([p["p1"]]),
            ]
        )
        assert offline_popularity(inst, lv, k=1) == [p["p1"]]

    def test_consensus_example7(self, consensus_data):
        p, inst, lv = consensus_data
        assert offline_consensus(inst, lv, k=1) == [p["p1"]]

    def test_controversiality_example8(self, consensus_data):
        p, inst, lv = consensus_data
        assert offline_controversiality(inst, lv, k=1) == [p["p2"]]

    def test_popularity_matches_manual_topk_random(self):
        # Cross-check the popularity sampler against an independent top-k.
        projects, inst, lv = random_instance(20, 40, 50, seed=33)
        k = 5
        counts = manual_scores(projects, lv)
        expected = sorted(projects, key=lambda p: (-counts[p], str(p)))[:k]
        assert offline_popularity(inst, lv, k=k) == expected


# ---------------------------------------------------------------------------
# online_adaptive_controversial
# ---------------------------------------------------------------------------
class TestOnlineAdaptive:
    def test_example9(self):
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1), ("p4", 1)])
        inst = Instance(p.values(), budget_limit=4)
        lv = ApprovalProfile(
            [
                ApprovalBallot([p["p1"], p["p2"]]),
                ApprovalBallot([p["p1"], p["p3"]]),
                ApprovalBallot([p["p2"], p["p3"]]),
                ApprovalBallot([]),
            ]
        )
        assert online_adaptive_controversial(
            inst, lv, {p["p1"], p["p2"]}, k=2
        ) == [p["p1"], p["p2"]]

    def test_returns_k_distinct(self):
        projects, inst, lv = random_instance(15, 30, 60, seed=9)
        result = online_adaptive_controversial(inst, lv, set(projects[:5]), k=4)
        assert len(result) == 4
        assert len(set(result)) == 4
        assert set(result) <= set(projects)


# ---------------------------------------------------------------------------
# PartialApprovalBallot + reveal_ballot (three-state ballots, Section 2.3)
# ---------------------------------------------------------------------------
class TestPartialBallot:
    def test_states_partition_projects(self):
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
        b = PartialApprovalBallot(
            approved={p["p1"]}, disapproved={p["p2"]}, hidden={p["p3"]}
        )
        assert b.approved == {p["p1"]}
        assert b.disapproved == {p["p2"]}
        assert b.hidden == {p["p3"]}
        assert b.exposed == {p["p1"], p["p2"]}
        assert (b.approved | b.disapproved | b.hidden) == set(p.values())

    def test_as_approval_ballot_keeps_only_approvals(self):
        p = make_projects([("p1", 1), ("p2", 1)])
        b = PartialApprovalBallot(approved={p["p1"]}, disapproved={p["p2"]})
        ab = b.as_approval_ballot()
        assert isinstance(ab, ApprovalBallot)
        assert set(ab) == {p["p1"]}

    def test_reveal_example2_4(self):
        # Example 2.4: true approvals {p1,p2}, exposed {p1,p3}.
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1), ("p4", 1)])
        inst = Instance(p.values(), budget_limit=4)
        b = reveal_ballot(inst, {p["p1"], p["p2"]}, {p["p1"], p["p3"]})
        assert b.approved == {p["p1"]}
        assert b.disapproved == {p["p3"]}
        assert b.hidden == {p["p2"], p["p4"]}
        assert b.exposed == {p["p1"], p["p3"]}


# ---------------------------------------------------------------------------
# predict_by_majority
# ---------------------------------------------------------------------------
class TestPredictByMajority:
    def test_example11(self):
        p = make_projects([("p1", 4), ("p2", 4), ("p3", 6)])
        inst = Instance(p.values(), budget_limit=6)
        lv = ApprovalProfile(
            [ApprovalBallot([p["p1"], p["p2"]]), ApprovalBallot([p["p1"], p["p2"]])]
        )
        partial = PartialApprovalBallot(hidden=set(p.values()))
        pred = predict_by_majority(inst, lv, partial)
        assert set(pred) == {p["p1"], p["p2"]}

    def test_exposed_approvals_are_kept(self):
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
        inst = Instance(p.values(), budget_limit=3)
        # LV would reject p3 (0%), but the voter explicitly approved it.
        lv = ApprovalProfile([ApprovalBallot([p["p1"]]), ApprovalBallot([p["p1"]])])
        partial = PartialApprovalBallot(
            approved={p["p3"]}, hidden={p["p1"], p["p2"]}
        )
        pred = predict_by_majority(inst, lv, partial)
        assert p["p3"] in pred

    def test_exposed_disapprovals_stay_rejected(self):
        p = make_projects([("p1", 1), ("p2", 1)])
        inst = Instance(p.values(), budget_limit=2)
        # LV approves both p1 and p2 unanimously, but the voter explicitly
        # disapproved p1; p2 is hidden and should be predicted as approved.
        lv = ApprovalProfile([ApprovalBallot([p["p1"], p["p2"]])] * 3)
        partial = PartialApprovalBallot(disapproved={p["p1"]}, hidden={p["p2"]})
        pred = predict_by_majority(inst, lv, partial)
        assert p["p1"] not in pred  # the explicit disapproval is honoured
        assert p["p2"] in pred      # hidden + LV majority -> approved


# ---------------------------------------------------------------------------
# Library-backed predictors: classification (XGBoost), MF, FM (Section 2.1).
# They share the contract of predict_by_majority, so the same behavioural
# invariants are checked for each one.
# ---------------------------------------------------------------------------
LIBRARY_PREDICTORS = [
    predict_by_classification,
    predict_by_matrix_factorization,
    predict_by_factorization_machines,
]


class TestLibraryPredictors:
    @pytest.mark.parametrize("predictor", LIBRARY_PREDICTORS)
    def test_strong_pattern_recovered(self, predictor):
        # LV unanimously approve {p1, p2} and reject p3; a TV voter with nothing
        # exposed must be completed to {p1, p2} by any correct predictor.
        p = make_projects([("p1", 4), ("p2", 4), ("p3", 6)])
        inst = Instance(p.values(), budget_limit=6)
        lv = ApprovalProfile([ApprovalBallot([p["p1"], p["p2"]])] * 4)
        partial = PartialApprovalBallot(hidden=set(p.values()))
        pred = predictor(inst, lv, partial)
        assert set(pred) == {p["p1"], p["p2"]}

    @pytest.mark.parametrize("predictor", LIBRARY_PREDICTORS)
    def test_exposed_votes_are_respected(self, predictor):
        # The exposed approval p3 is kept and the exposed disapproval p1 stays
        # rejected, regardless of what the model would otherwise predict.
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
        inst = Instance(p.values(), budget_limit=3)
        lv = ApprovalProfile([ApprovalBallot([p["p1"]])] * 3)
        partial = PartialApprovalBallot(
            approved={p["p3"]}, disapproved={p["p1"]}, hidden={p["p2"]}
        )
        pred = predictor(inst, lv, partial)
        assert p["p3"] in pred       # exposed approval kept
        assert p["p1"] not in pred   # exposed disapproval rejected


# ---------------------------------------------------------------------------
# recommend (full pipeline)
# ---------------------------------------------------------------------------
class TestRecommend:
    def test_example10_perfect(self):
        p = make_projects([("p1", 3), ("p2", 3), ("p3", 4), ("p4", 4)])
        inst = Instance(p.values(), budget_limit=6)
        lv = ApprovalProfile([ApprovalBallot([p["p1"], p["p2"]])] * 3)
        bundle = set(recommend(inst, lv, {"v4": {p["p1"], p["p2"]}}, k=1))
        assert bundle == {p["p1"], p["p2"]}

    def test_pipeline_respects_budget_random(self):
        # Cheap, widely-approved projects so the pipeline must fund something.
        p = make_projects([("p1", 2), ("p2", 2), ("p3", 9), ("p4", 9)])
        inst = Instance(p.values(), budget_limit=6)
        lv = ApprovalProfile([ApprovalBallot([p["p1"], p["p2"]])] * 4)
        tv = {"v1": {p["p1"], p["p2"]}, "v2": {p["p1"]}}
        bundle = list(recommend(inst, lv, tv, k=2))
        assert set(bundle) <= set(p.values())
        assert sum(proj.cost for proj in bundle) <= inst.budget_limit
        assert len(bundle) > 0


# ---------------------------------------------------------------------------
# symmetric_distance
# ---------------------------------------------------------------------------
class TestSymmetricDistance:
    def test_identical_bundles(self):
        p = make_projects([("p1", 1), ("p2", 1)])
        assert symmetric_distance(set(p.values()), set(p.values())) == 0
        # bundles differing by one project have distance 1.
        assert symmetric_distance(set(p.values()), {p["p1"]}) == 1

    def test_paper_toy_examples(self):
        a, b, c, d, e = (Project(x, 1) for x in "abcde")
        assert symmetric_distance({a, b, c}, {b, a, c}) == 0
        # NB: the paper's text prints 1 here, but the symmetric difference of
        # {a,b,c} and {a,c,d} is {b,d}, i.e. 2 (a typo in the paper).
        assert symmetric_distance({a, b, c}, {a, c, d}) == 2
        assert symmetric_distance({a, b, c}, {a, d, e}) == 4

    def test_large_structured_known_value(self):
        # rb = first 60, pb = last 60 of 100 -> overlap 20, symmetric diff 80.
        a = padded_projects(100, cost=1)
        assert symmetric_distance(set(a[:60]), set(a[40:])) == 80


# ---------------------------------------------------------------------------
# fractional_allocation_score
# ---------------------------------------------------------------------------
class TestFractionalAllocation:
    def test_example11_zero_score(self):
        p = make_projects([("p1", 4), ("p3", 6)])
        # disjoint bundles -> 0.0; a full overlap of cost 6 over budget 6 -> 1.0.
        assert fractional_allocation_score({p["p3"]}, {p["p1"]}, budget_limit=6) == 0.0
        assert fractional_allocation_score({p["p3"]}, {p["p3"]}, budget_limit=6) == 1.0

    def test_partial_overlap(self):
        p = make_projects([("p1", 2), ("p2", 3), ("p3", 5)])
        # overlap is {p2}, cost 3, budget 10 -> 0.3
        score = fractional_allocation_score(
            {p["p1"], p["p2"]}, {p["p2"], p["p3"]}, budget_limit=10
        )
        assert score == pytest.approx(0.3)

    def test_value_matches_overlap_random(self):
        projects, inst, _ = random_instance(12, 1, 40, seed=14)
        rng = random.Random(15)
        rb = set(rng.sample(projects, 6))
        pb = set(rng.sample(projects, 6))
        score = fractional_allocation_score(rb, pb, budget_limit=inst.budget_limit)
        expected = sum(p.cost for p in (rb & pb)) / inst.budget_limit
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
