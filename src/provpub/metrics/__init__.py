"""Evaluation metrics and intervals over seeds."""
from .confidence_intervals import bootstrap_interval, t_interval
from .metrics import (
                      detection_rate,
                      expected_flips,
                      false_alarm_rate,
                      flip_probability,
                      per_record_budget,
                      preservation_by_family,
                      timed_estimate,
                      timed_outcome_may_change,
                      value_coincidence_bound,
                      verdict_transitions,
)

__all__ = ["preservation_by_family", "verdict_transitions", "detection_rate", "false_alarm_rate", "flip_probability",
           "expected_flips", "value_coincidence_bound", "timed_estimate", "timed_outcome_may_change",
           "per_record_budget", "t_interval", "bootstrap_interval"]
