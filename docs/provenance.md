# Provenance and rewritten history

Every run writes the git commit it ran from into its `manifest.json` (`git_commit`, with `git_dirty`). The frozen configuration was committed before the single lock evaluation, so the commit order proves that the lock split did not shape the configuration.

On 2026-10-06 the repository history was rewritten to remove local planning files (`docs/plans/`, `docs/wayfinder/`, `docs/HACKATHON.md`). The rewrite changed every commit SHA. It kept commit dates, authors, messages and all code.

The commit map links the two histories: {download}`commit-map.csv <provenance/commit-map.csv>`.

- `old_sha`: the SHA recorded in run manifests and in the [experiment log](experiment-log.md).
- `new_sha`: the matching commit in the current history.
- 39 old commits changed only planning files and no longer exist. Their row points to the nearest kept ancestor, which has identical code, and their subject says so.

Example: the freeze commit `68f5301` ("chore: freeze final configuration", 2026-10-02 23:21 +03:00) is now `e029b24`.

The map covers every commit recorded in the run manifests of the RTX 3090 host and the M4 Pro.

To look up the commit of a run:

```bash
python -c "import json,sys; print(json.load(open(sys.argv[1]))['git_commit'])" artifacts/<run-id>/manifest.json
grep '^<old_sha>' docs/provenance/commit-map.csv
```
