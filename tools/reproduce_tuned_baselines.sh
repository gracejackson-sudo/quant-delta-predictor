#!/bin/bash
# Reproduce out/tuned_baselines.json detached, in an isolated worktree, and
# compare it against the committed artifact. Safe to run and walk away:
# nothing writes into the working tree you use day to day.
#
# TWO STEPS, NOT ONE, AND THAT WAS THE BUG. The committed artifact is the
# product of src/tune_baselines.py AND src/tune_baseline_cis.py: the second
# reads the hyperparameter selections the first recorded, adds the
# checkpoint-bootstrap CIs, and merges them back into the SAME file. The
# first version of this script ran only step 1, so it compared a one-step
# rebuild against a two-step artifact and reported DIFFERS. The artifact was
# never the problem; the comparison was. Verified 2026-10-13: run both steps
# and it is IDENTICAL.
#
# ONE NORMALIZED FIELD, NAMED: `seconds`, the top-level wall-clock time
# src/tune_baselines.py records. Nothing else is excused. It is reported
# side by side rather than hidden, because a 20x difference in runtime is
# worth seeing even when every figure matches.
set -euo pipefail

# Derived, never written in: an absolute home path in a tracked file is the
# de-anonymiser this repository's own gates exist to catch, and
# tests/test_all.py::test_no_secret_shaped_strings_in_tracked_tree caught
# exactly that in the first version of this script.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=$(mktemp -d /tmp/qdp-tune.XXXXXX)
echo "scratch: $WORK"

# Isolated checkout of the exact commit under test, so the run cannot see
# or disturb anything uncommitted.
git -C "$REPO" worktree add --detach "$WORK/repo" HEAD >/dev/null
HEAD_SHA=$(git -C "$WORK/repo" rev-parse --short HEAD)
echo "commit : $HEAD_SHA"

# Both scripts derive their paths from their own location, so running the
# worktree's copies reads and writes the worktree's out/.
nohup bash -c "
  set -e
  '$REPO/.venv/bin/python' '$WORK/repo/src/tune_baselines.py'
  echo '--- step 1 done, starting step 2 (tune_baseline_cis.py) ---'
  '$REPO/.venv/bin/python' '$WORK/repo/src/tune_baseline_cis.py'
" > "$WORK/tune.log" 2>&1 &
PID=$!
echo "pid    : $PID"
echo "log    : $WORK/tune.log"

# The comparison runs when the job does, however long it takes. The 2026-10-13
# run took 9,771 seconds on a laptop with a duplicate run competing for the
# same cores, so this waits rather than polling with a deadline.
(
  while kill -0 "$PID" 2>/dev/null; do sleep 30; done
  NEW="$WORK/repo/out/tuned_baselines.json"
  OLD="$REPO/out/tuned_baselines.json"
  {
    echo "commit under test : $HEAD_SHA"
    echo "finished          : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    # VERSIONS IN THE VERDICT, NOT INFERRED LATER. If this comes back
    # DIFFERS the first question is whether the environment moved, so the
    # environment is recorded beside the answer.
    echo "interpreter       : $("$REPO/.venv/bin/python" -c 'import sys;print(sys.version.replace(chr(10)," "))')"
    echo "installed versions:"
    "$REPO/.venv/bin/python" - <<'PYV' | sed 's/^/                    /'
import importlib.metadata as md
for p in ("pandas", "numpy", "scipy", "scikit-learn"):
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p}==ABSENT")
PYV
    echo "requirements.txt  :"
    grep -E '^[A-Za-z]' "$REPO/requirements.txt" | sed 's/^/                    /'
    echo "seed              : random_state=0 (src/tune_baselines.py),"
    echo "                    DRAWS=20000 SEED=0 (src/tune_baseline_cis.py)"
    echo
    if [ ! -f "$NEW" ]; then
      echo "RESULT: NO ARTIFACT -- the run did not write one. Tail of log:"
      tail -40 "$WORK/tune.log"
    else
      "$REPO/.venv/bin/python" - "$OLD" "$NEW" <<'PYC'
import json, sys
old = json.load(open(sys.argv[1]))
new = json.load(open(sys.argv[2]))
# The one normalized field, named rather than globbed.
NORMALIZED = ("seconds",)
print("normalized field  : seconds (wall clock; every other key compared "
      "exactly)")
for k in NORMALIZED:
    print(f"  committed {k:<10}: {old.get(k)}")
    print(f"  rebuilt   {k:<10}: {new.get(k)}")
a = {k: v for k, v in old.items() if k not in NORMALIZED}
b = {k: v for k, v in new.items() if k not in NORMALIZED}
missing = sorted(set(a) - set(b))
added = sorted(set(b) - set(a))
diff = sorted(k for k in set(a) & set(b) if a[k] != b[k])
print(f"keys compared     : {len(set(a) | set(b))}")
if not (missing or added or diff):
    print("\nRESULT: IDENTICAL -- every key outside `seconds` matches "
          "exactly.")
    sys.exit(0)
print("\nRESULT: DIFFERS.")
for k in missing:
    print(f"  only in the committed artifact: {k}")
for k in added:
    print(f"  only in the rebuild           : {k}")
for k in diff:
    print(f"  differs                       : {k}")
    print(f"      committed {old[k]!r}"[:300])
    print(f"      rebuilt   {new[k]!r}"[:300])
print("\nBefore concluding the artifact is stale, check that BOTH steps ran "
      "-- the log above should show step 2. A one-step rebuild compared "
      "against this two-step artifact reports DIFFERS and means nothing.")
sys.exit(1)
PYC
    fi
    echo
    echo "worktree kept at $WORK/repo for inspection. Remove with:"
    echo "  git -C \"$REPO\" worktree remove --force $WORK/repo && rm -rf $WORK"
  } > "$WORK/VERDICT.txt" 2>&1
  echo
  echo "=== VERDICT ==="
  cat "$WORK/VERDICT.txt"
) &

echo
echo "Running detached. Read the verdict when it lands:"
echo "  cat $WORK/VERDICT.txt"
echo "Follow progress with:"
echo "  tail -f $WORK/tune.log"
