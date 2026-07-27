"""
Module implementing the algorithms of
"A Recommendation System for Participatory Budgeting",
by Gil Leibiker and Nimrod Talmon (2023), https://optlearnmas23.github.io/files/p17.pdf

:py:mod:`~pabutools.recommendation.recommendation` holds the sampling setups,
the pipeline, the voting rule and the evaluation metrics;
:py:mod:`~pabutools.recommendation.model_training` holds the fitting and
prediction of the three learning-based modules (whose ML libraries are optional
dependencies, installed with ``pip install pabutools[recommendation]``).

Programmer: Roei Yanku
"""

from pabutools.recommendation.recommendation import (
    APPROVAL,
    DISAPPROVAL,
    HIDDEN,
    partial_ballot,
    reveal_ballot,
    approved_projects,
    disapproved_projects,
    exposed_projects,
    hidden_projects,
    as_approval_ballot,
    consensus_levels,
    most_consensual_projects,
    greedy_approval,
    random_setup,
    offline_popularity,
    offline_consensus,
    offline_controversiality,
    online_adaptive_controversial,
    next_adaptive_question,
    SETUPS,
    PREDICTORS,
    exposed_sets,
    plan_sampling,
    split_lv_tv,
    complete_ballots,
    run_pipeline,
    run_experiment,
    Predictor,
    as_predictor,
    elect,
    SAMPLE_DEGREES,
    LV_DEGREES,
    run_all_experiments,
    classification_metrics,
    fractional_allocation_score,
)
from pabutools.recommendation.model_training import (
    train_classification,
    train_matrix_factorization,
    train_factorization_machines,
    predict_by_classification,
    predict_by_matrix_factorization,
    predict_by_factorization_machines,
)

__all__ = [
    # Section 2.3 - partial ballots and the +1/-1/0 score convention.
    "APPROVAL",
    "DISAPPROVAL",
    "HIDDEN",
    "partial_ballot",
    "reveal_ballot",
    "approved_projects",
    "disapproved_projects",
    "exposed_projects",
    "hidden_projects",
    "as_approval_ballot",
    # Section 2.2 - popularity, consensus and the voting rule.
    "consensus_levels",
    "most_consensual_projects",
    "greedy_approval",
    # Section 3.1 - the sampling setups.
    "SETUPS",
    "random_setup",
    "offline_popularity",
    "offline_consensus",
    "offline_controversiality",
    "online_adaptive_controversial",
    "next_adaptive_question",
    "exposed_sets",
    # Section 2.1 - the prediction modules.
    "PREDICTORS",
    "Predictor",
    "as_predictor",
    "train_classification",
    "train_matrix_factorization",
    "train_factorization_machines",
    "predict_by_classification",
    "predict_by_matrix_factorization",
    "predict_by_factorization_machines",
    # Running a process: a real one, or the paper's experiments.
    "plan_sampling",
    "elect",
    "split_lv_tv",
    "complete_ballots",
    "run_pipeline",
    "run_experiment",
    "run_all_experiments",
    "SAMPLE_DEGREES",
    "LV_DEGREES",
    # Section 5 - evaluation.
    "classification_metrics",
    "fractional_allocation_score",
]
