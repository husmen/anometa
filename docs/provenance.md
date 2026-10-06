# Provenance and rewritten history

Every run writes the git commit it ran from into its `manifest.json` (`git_commit`, with `git_dirty`). The frozen configuration was committed before the single lock evaluation, so the commit order shows that the lock split did not shape the configuration.

The history was rewritten twice on 2026-10-06. Both rewrites kept commit dates, authors and the final code.

1. **Planning files removed.** Local planning files (`docs/plans/`, `docs/wayfinder/`, `docs/HACKATHON.md`) were dropped from every commit, which changed every SHA. 39 commits that only changed planning files disappeared.
2. **History cleaned up.** The resulting 131 commits were squashed into 32 feature commits, each of which passes the lint, type and test checks. The tag `archive/full-history` keeps the 131-commit history, so every intermediate code state stays reachable.

The commit map links the three histories: {download}`commit-map.csv <provenance/commit-map.csv>`.

- `old_sha`: the SHA recorded in run manifests and in the [experiment log](experiment-log.md).
- `archive_sha`: the same commit in `archive/full-history`, with identical code. For a removed planning-only commit, the nearest kept ancestor, which has identical code; its subject says so.
- `clean_sha`: the feature commit of the current history that contains it.

Example: the freeze commit `68f5301` ("chore: freeze final configuration", 2026-10-02 23:21 +03:00) is `e029b24` in the archive and `bce1592` in the current history. All three have the same tree, so the lock runs' code state is preserved exactly.

The map covers every commit recorded in the run manifests of the RTX 3090 host and the M4 Pro. The tuned PatchCore runs record `e2e9c59`, a commit that was later squashed into the current docs commit; its code is identical to `bdf35ca`, and the map lists it with no archive SHA.

To look up the code of a run:

```bash
python -c "import json,sys; print(json.load(open(sys.argv[1]))['git_commit'])" artifacts/<run-id>/manifest.json
grep '^<old_sha>' docs/provenance/commit-map.csv   # then: git checkout <archive_sha>
```
