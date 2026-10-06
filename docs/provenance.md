# Provenance and rewritten history

Every run writes the git commit it ran from into its `manifest.json` (`git_commit`, with `git_dirty`). The frozen configuration was committed before the single lock evaluation, so the commit order shows that the lock split did not shape the configuration.

Before submission, the history was cleaned up on 2026-10-06. Commit dates, authors and the final code were kept.

1. **Planning files removed.** Local planning files (`docs/plans/`, `docs/wayfinder/`, `docs/HACKATHON.md`) were dropped from every commit. Commits that only changed planning files disappeared.
2. **History squashed.** The three initial commits became one, and the rest became 33 feature commits, each of which passes the lint, type and test checks.

The intermediate code states of the original history were not kept. Every recorded commit was remapped instead: run manifests, the run log (`runs.jsonl`), Track A's per-scenario `result.json` files and the feature cache provenance now point at the current history in `git_commit`, keep the old SHA in `git_commit_original`, and say how close the match is in `git_commit_mapping`:

- `exact`: the new commit has the same tree as the old one.
- `same-code`: the code, configs, splits and lockfile are identical; only documentation differs.
- `squashed`: the old commit's changes are part of the new feature commit, together with neighbouring changes. The run used an intermediate code state of that feature.

The lock runs are `exact`: they ran on the freeze commit `68f5301`, now `5f94e31` ("chore: freeze final configuration", 2026-10-02 23:21 +03:00), with the same tree. Dev runs from before the freeze are mostly `squashed`.

The full map is {download}`commit-map.csv <provenance/commit-map.csv>` (`old_sha`, `new_sha`, `mapping`, `date`, `subject`). It covers every commit recorded on the RTX 3090 host and the M4 Pro.
