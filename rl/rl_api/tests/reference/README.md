# Frozen sampler reference

`bernoulli_sparse.py` is the numerical reference used by
`test_backup_sampling.py` to check sampler, score, gradient and optimizer
equivalence with the earlier independent-topology recipe. It is a byte-for-byte
copy of the reference module previously kept in the local `rl/BACKUP/` folder.

This is test source, not a model checkpoint or training result. Keep it fixed
when changing the active implementation; update it only when deliberately
changing the reference contract. The other local backup files are not needed
by the library or tests and stay outside version control.
