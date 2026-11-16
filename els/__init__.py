"""Extraneous Load Score (ELS) fitting and evaluation.

`features` and `scoring` define the 30 activation features and the fitted scorers;
`splits` defines the row groups and calibration splits. `fit` (held-out base problems),
`calibrate` (few-shot transfer to an unseen axis), `generalization`, `behavior` and
`families` each run one analysis and save its scores and summaries. `validation` and
`verify` recompute those summaries from the saved scores and compare them with the
reference values or with a refit's own summaries.
"""
