# Historical comparison data

`current_training.json` stores the recipe captured from the original runner,
minus obsolete teacher fields. It is read by comparison diagnostics only.
Training and inspection do not read this file. Editing it does not change a
run: edit the Python settings blocks in `experiments/train_current.py` and
`experiments/inspect_current.py` instead.

`reference_manifest.json` records SHA-256 hashes of the original files at the
time this API was created and the three allowed output-path differences.
`Experiment.verify_reference()` compares current files with that snapshot.
Training requires the three numerical implementation files to match unless
explicitly told to skip verification. Changes to an inspector's editable
parameters can change its hash without changing training numerics; its current
selection and plot helpers are covered by the workflow tests.

The defaults reproduce supplied-topology training. A legacy field or class
name containing “Bernoulli” does not mean learned gates are enabled.
