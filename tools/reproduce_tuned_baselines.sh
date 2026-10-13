#!/bin/bash
# Reproduce out/tuned_baselines.json detached, in an isolated worktree, and
# byte-compare against the committed artifact. Safe to run and walk away:
# nothing writes into the working tree you use day to day.
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

# The script derives its output path from its own location, so running the
# worktree's copy writes to the worktree's out/.
nohup "$REPO/.venv/bin/python" "$WORK/repo/src/tune_baselines.py" \
      > "$WORK/tune.log" 2>&1 &
PID=$!
echo "pid    : $PID"
echo "log    : $WORK/tune.log"

# The comparison runs when the job does, however long that is. The last
# attempt was killed at a 2-hour limit without finishing, so this waits
# rather than polling with a deadline.
(
  wait_for() { while kill -0 "$1" 2>/dev/null; do sleep 30; done; }
  wait_for "$PID"
  NEW="$WORK/repo/out/tuned_baselines.json"
  OLD="$REPO/out/tuned_baselines.json"
  {
    echo "commit under test : $HEAD_SHA"
    echo "finished          : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    if [ ! -f "$NEW" ]; then
      echo "RESULT: NO ARTIFACT -- the run did not write one. Tail of log:"
      tail -40 "$WORK/tune.log"
    elif cmp -s "$NEW" "$OLD"; then
      echo "RESULT: IDENTICAL -- reproduces byte-for-byte."
      echo "sha256: $(shasum -a 256 "$NEW" | cut -d' ' -f1)"
    else
      echo "RESULT: DIFFERS."
      echo "  committed: $(shasum -a 256 "$OLD" | cut -d' ' -f1)"
      echo "  rebuilt  : $(shasum -a 256 "$NEW" | cut -d' ' -f1)"
      echo "--- first differing lines ---"
      diff "$OLD" "$NEW" | head -40 || true
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
