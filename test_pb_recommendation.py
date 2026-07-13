"""
Unit tests for `pb_recommendation`, the implementation of the algorithms in
"A Recommendation System for Participatory Budgeting",
by Gil Leibiker and Nimrod Talmon (2023), https://optlearnmas23.github.io/files/p17.pdf

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
    # Partial-ballot helpers + the +1/-1/0 convention constants.
    partial_ballot,
    reveal_ballot,
    approved_projects,
    disapproved_projects,
    exposed_projects,
    hidden_projects,
    as_approval_ballot,
    consensus_levels,
    greedy_approval,
    random_setup,
    offline_popularity,
    offline_consensus,
    offline_controversiality,
    online_adaptive_controversial,
    predict_by_classification,
    predict_by_matrix_factorization,
    predict_by_factorization_machines,
    run_pipeline,
    classification_metrics,
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
# consensus_levels
# (Approval scores, Definition 2.1, have no function of our own - the code uses
# pabutools' profile.approval_scores() directly, which the library itself tests.)
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
        exposed = random_setup(inst, k=2, seed=0)
        assert len(exposed) == 2
        assert exposed <= set(p.values())

    def test_returns_a_set_subset_of_projects(self):
        projects, inst, _ = random_instance(10, 1, 30, seed=5)
        exposed = random_setup(inst, k=3, seed=7)
        assert isinstance(exposed, set)
        assert len(exposed) == 3
        assert exposed <= set(projects)

    def test_k_equals_all_exposes_everything(self):
        # Edge case: exposing k = |P| projects reveals the whole instance.
        p = make_projects([("p1", 1), ("p2", 1)])
        inst = Instance(p.values(), budget_limit=2)
        assert random_setup(inst, k=2, seed=0) == set(p.values())


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
        assert offline_popularity(inst, lv, k=1) == {p["p1"]}

    def test_consensus_example7(self, consensus_data):
        p, inst, lv = consensus_data
        assert offline_consensus(inst, lv, k=1) == {p["p1"]}

    def test_controversiality_example8(self, consensus_data):
        p, inst, lv = consensus_data
        assert offline_controversiality(inst, lv, k=1) == {p["p2"]}

    def test_popularity_matches_manual_topk_random(self):
        # Cross-check the popularity sampler against an independent top-k.
        projects, inst, lv = random_instance(20, 40, 50, seed=33)
        k = 5
        counts = manual_scores(projects, lv)
        expected = set(sorted(projects, key=lambda p: (-counts[p], str(p)))[:k])
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
        ) == {p["p1"], p["p2"]}

    def test_returns_k_distinct(self):
        projects, inst, lv = random_instance(15, 30, 60, seed=9)
        result = online_adaptive_controversial(inst, lv, set(projects[:5]), k=4)
        assert isinstance(result, set)
        assert len(result) == 4
        assert result <= set(projects)


# ---------------------------------------------------------------------------
# Partial (three-state) ballots, Section 2.3. Represented as a CardinalBallot
# under the convention: +1 approved, -1 disapproved, 0 (or absent) hidden.
# ---------------------------------------------------------------------------
class TestPartialBallot:
    def test_scores_follow_convention(self):
        # The whole point of the encoding: +1 = approve, -1 = disapprove,
        # 0 = unknown. Pin it down so the convention can't silently drift.
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
        b = partial_ballot(
            approved={p["p1"]}, disapproved={p["p2"]}, hidden={p["p3"]}
        )
        assert b[p["p1"]] == 1
        assert b[p["p2"]] == -1
        assert b[p["p3"]] == 0

    def test_states_partition_projects(self):
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
        inst = Instance(p.values(), budget_limit=3)
        b = partial_ballot(
            approved={p["p1"]}, disapproved={p["p2"]}, hidden={p["p3"]}
        )
        assert approved_projects(b) == {p["p1"]}
        assert disapproved_projects(b) == {p["p2"]}
        assert hidden_projects(b, inst) == {p["p3"]}
        assert exposed_projects(b) == {p["p1"], p["p2"]}
        assert (
            approved_projects(b) | disapproved_projects(b) | hidden_projects(b, inst)
        ) == set(p.values())

    def test_absent_project_is_hidden(self):
        # A project never mentioned in the ballot is hidden, just like an
        # explicit 0 - both belong to H_v.
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
        inst = Instance(p.values(), budget_limit=3)
        b = partial_ballot(approved={p["p1"]}, disapproved={p["p2"]})  # p3 absent
        assert hidden_projects(b, inst) == {p["p3"]}

    def test_as_approval_ballot_keeps_only_approvals(self):
        p = make_projects([("p1", 1), ("p2", 1)])
        b = partial_ballot(approved={p["p1"]}, disapproved={p["p2"]})
        ab = as_approval_ballot(b)
        assert isinstance(ab, ApprovalBallot)
        assert set(ab) == {p["p1"]}

    def test_reveal_example2_4(self):
        # Example 2.4: full ballot {p1,p2}, exposed set {p1,p3}.
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1), ("p4", 1)])
        inst = Instance(p.values(), budget_limit=4)
        b = reveal_ballot(inst, {p["p1"], p["p2"]}, {p["p1"], p["p3"]})
        assert approved_projects(b) == {p["p1"]}
        assert disapproved_projects(b) == {p["p3"]}
        assert hidden_projects(b, inst) == {p["p2"], p["p4"]}
        assert exposed_projects(b) == {p["p1"], p["p3"]}


# ---------------------------------------------------------------------------
# The three prediction modules of the paper: classification (XGBoost), MF, FM
# (Section 2.1). They share one contract - keep the exposed votes, predict the
# hidden ones - so the same behavioural invariants are checked for each.
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
        partial = partial_ballot(hidden=set(p.values()))
        pred = predictor(inst, lv, partial)
        assert set(pred) == {p["p1"], p["p2"]}

    @pytest.mark.parametrize("predictor", LIBRARY_PREDICTORS)
    def test_exposed_votes_are_respected(self, predictor):
        # The exposed approval p3 is kept and the exposed disapproval p1 stays
        # rejected, regardless of what the model would otherwise predict.
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
        inst = Instance(p.values(), budget_limit=3)
        lv = ApprovalProfile([ApprovalBallot([p["p1"]])] * 3)
        partial = partial_ballot(
            approved={p["p3"]}, disapproved={p["p1"]}, hidden={p["p2"]}
        )
        pred = predictor(inst, lv, partial)
        assert p["p3"] in pred       # exposed approval kept
        assert p["p1"] not in pred   # exposed disapproval rejected


# ---------------------------------------------------------------------------
# run_pipeline (full pipeline)
# ---------------------------------------------------------------------------
class TestRunPipeline:
    def test_example10_perfect(self):
        p = make_projects([("p1", 3), ("p2", 3), ("p3", 4), ("p4", 4)])
        inst = Instance(p.values(), budget_limit=6)
        lv = ApprovalProfile([ApprovalBallot([p["p1"], p["p2"]])] * 3)
        bundle = set(run_pipeline(inst, lv, {"v4": {p["p1"], p["p2"]}}, k=1,
                                  setup="offline_popularity",
                                  predict=predict_by_matrix_factorization))
        assert bundle == {p["p1"], p["p2"]}

    def test_pipeline_respects_budget_random(self):
        # Cheap, widely-approved projects so the pipeline must fund something.
        p = make_projects([("p1", 2), ("p2", 2), ("p3", 9), ("p4", 9)])
        inst = Instance(p.values(), budget_limit=6)
        lv = ApprovalProfile([ApprovalBallot([p["p1"], p["p2"]])] * 4)
        tv = {"v1": {p["p1"], p["p2"]}, "v2": {p["p1"]}}
        bundle = list(run_pipeline(inst, lv, tv, k=2,
                                   setup="offline_popularity",
                                   predict=predict_by_matrix_factorization))
        assert set(bundle) <= set(p.values())
        assert sum(proj.cost for proj in bundle) <= inst.budget_limit
        assert len(bundle) > 0

    def test_unknown_setup_raises_value_error(self):
        p = make_projects([("p1", 1), ("p2", 1)])
        inst = Instance(p.values(), budget_limit=2)
        lv = ApprovalProfile([ApprovalBallot([p["p1"]])] * 2)
        with pytest.raises(ValueError, match="unknown setup 'by_magic'"):
            run_pipeline(inst, lv, {"v1": {p["p1"]}}, k=1,
                         setup="by_magic",
                         predict=predict_by_matrix_factorization)

    def test_k_out_of_range_raises_value_error(self):
        p = make_projects([("p1", 1), ("p2", 1)])
        inst = Instance(p.values(), budget_limit=2)
        lv = ApprovalProfile([ApprovalBallot([p["p1"]])] * 2)
        for bad_k in (-1, 3):  # below 0 and above the number of projects
            with pytest.raises(ValueError, match="must be between 0 and"):
                run_pipeline(inst, lv, {"v1": {p["p1"]}}, k=bad_k,
                             setup="random",
                             predict=predict_by_matrix_factorization)

    @pytest.mark.parametrize("setup", [
        "random",
        "offline_popularity",
        "offline_consensus",
        "offline_controversiality",
        "online_adaptive_controversial",
    ])
    @pytest.mark.parametrize("predictor", LIBRARY_PREDICTORS)
    def test_two_camp_electorate_recovers_real_bundle(self, setup, predictor):
        """
        The end-to-end criterion of the paper (Section 2.4): the pipeline must
        estimate the *ideal* outcome. A structured two-camp electorate - a 60%
        majority camp approving {p1, p2, p3} and a 40% minority camp approving
        {p4, p5, p6} - fixes the real winning bundle at {p1, p2, p3}. A third
        of the voters become TV with only k=2 of the 6 projects exposed, and
        every setup x predictor combination must still reconstruct the exact
        real bundle (FA = 1.0, SD = 0).
        """
        p = make_projects([(f"p{i}", 1) for i in range(1, 7)])
        camp_a = {p["p1"], p["p2"], p["p3"]}   # 18 of 30 voters (60%)
        camp_b = {p["p4"], p["p5"], p["p6"]}   # 12 of 30 voters (40%)
        inst = Instance(p.values(), budget_limit=3)

        # The real bundle, from all 30 full ballots: camp A's projects score
        # 18 > 12, and the budget funds exactly three unit-cost projects.
        full_profile = ApprovalProfile(
            [ApprovalBallot(camp_a)] * 18 + [ApprovalBallot(camp_b)] * 12
        )
        real_bundle = set(greedy_approval(inst, full_profile))
        assert real_bundle == camp_a  # sanity: the ground truth is as designed

        # LV/TV split preserving the 60/40 mix: 20 LV, 10 TV.
        lv = ApprovalProfile([ApprovalBallot(camp_a)] * 12
                             + [ApprovalBallot(camp_b)] * 8)
        tv = {f"a{i}": set(camp_a) for i in range(6)}
        tv.update({f"b{i}": set(camp_b) for i in range(4)})

        predicted = set(run_pipeline(inst, lv, tv, k=2,
                                     setup=setup, predict=predictor, seed=7))
        assert predicted == real_bundle
        assert fractional_allocation_score(
            real_bundle, predicted, inst.budget_limit) == 1.0
        assert len(real_bundle ^ predicted) == 0  # symmetric distance


# ---------------------------------------------------------------------------
# classification_metrics (Section 5.1)
# ---------------------------------------------------------------------------
class TestClassificationMetrics:
    def test_mixed_hit_miss_false_alarm(self):
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1), ("p4", 1)])
        hidden = set(p.values())
        # really approves {p1, p2}, predicted {p1, p3}: hit p1, miss p2, false p3.
        m = classification_metrics({p["p1"], p["p2"]}, {p["p1"], p["p3"]}, hidden)
        assert m == {"precision": 0.5, "recall": 0.5, "f1": 0.5}

    def test_perfect_prediction(self):
        p = make_projects([("p1", 1), ("p2", 1), ("p3", 1)])
        hidden = set(p.values())
        m = classification_metrics({p["p1"], p["p2"]}, {p["p1"], p["p2"]}, hidden)
        assert m == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_exposed_votes_excluded(self):
        # A wrong prediction on an *exposed* project must not affect the metrics:
        # only the hidden set is scored. Hidden = {p2}, predicted perfectly there.
        p = make_projects([("p1", 1), ("p2", 1)])
        m = classification_metrics(
            real_approved={p["p2"]},          # p1 exposed, p2 hidden+approved
            predicted_approved={p["p1"], p["p2"]},
            hidden={p["p2"]},
        )
        assert m == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_no_approvals_gives_zero(self):
        # No real and no predicted approvals among the hidden projects -> the
        # metrics are undefined and reported as 0.0 by convention.
        p = make_projects([("p1", 1), ("p2", 1)])
        m = classification_metrics(set(), set(), hidden=set(p.values()))
        assert m == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


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
