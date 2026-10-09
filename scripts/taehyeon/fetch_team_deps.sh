#!/usr/bin/env bash
# Extract teammates' code (read-only copies) into .deps/team so that the
# taehyeon stage 5-1/5-2 code can be tested before every branch is merged.
# Nothing in teammates' directories is modified; .deps/ is git-ignored.
#
# Each package comes from its owner's branch:
#   pac_common / pac_planning  <- donghan  (feature/donghan-placement-planner)
#   ahead_dataset_generator    <- jaesung  (feature/jaesung-dataset-generator)
#   pac_simulation             <- jaesung  (feature/jaesung-physics-simulator)
# If a branch is missing, the package is searched in the other branches.
# Source: the team monorepo (TEAM_REPO_URL, default dlwotjd1289-cloud/dlwotjd1289-cloud,
# branch main), fetched as refs/remotes/team/main. pac-mission1-shared is no longer used.
#
# Usage: scripts/taehyeon/fetch_team_deps.sh [PLANNER_REF] [DATASET_REF] [SIM_REF]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEAM_REPO_URL="${TEAM_REPO_URL:-https://github.com/dlwotjd1289-cloud/dlwotjd1289-cloud.git}"
TEAM_BRANCH="${TEAM_BRANCH:-main}"
PLANNER_REF="${1:-team/$TEAM_BRANCH}"
DATASET_REF="${2:-team/$TEAM_BRANCH}"
SIM_REF="${3:-team/$TEAM_BRANCH}"
OUT="$ROOT/.deps/team"

cd "$ROOT"
git fetch --quiet --depth 1 "$TEAM_REPO_URL" \
  "+refs/heads/$TEAM_BRANCH:refs/remotes/team/$TEAM_BRANCH" 2>/dev/null \
  || echo "warning: could not fetch $TEAM_REPO_URL (using refs already present)" >&2

rm -rf "$OUT"
mkdir -p "$OUT"
: > "$OUT/SOURCES.txt"
for pair in "planner_branch:$PLANNER_REF" "dataset_branch:$DATASET_REF" "sim_branch:$SIM_REF"; do
  name="${pair%%:*}"
  ref="${pair#*:}"
  if git rev-parse --verify --quiet "$ref^{commit}" > /dev/null; then
    mkdir -p "$OUT/$name"
    git archive "$ref" | tar -x -C "$OUT/$name"
    echo "$name=$ref $(git rev-parse "$ref")" >> "$OUT/SOURCES.txt"
  else
    echo "$name=$ref MISSING" >> "$OUT/SOURCES.txt"
  fi
done

# Locate a package regardless of the upload prefix, preferring the owner's branch.
find_dir() {  # find_dir <relative path> <branch dirs...>
  local rel="$1"; shift
  for branch in "$@"; do
    [ -d "$OUT/$branch" ] || continue
    local hit
    hit="$(find "$OUT/$branch" -type d -path "*$rel" -print -quit)"
    if [ -n "$hit" ]; then echo "$hit"; return; fi
  done
}
PAC_COMMON="$(find_dir ros2_ws/src/pac_common planner_branch dataset_branch sim_branch)"
PAC_PLANNING="$(find_dir ros2_ws/src/pac_planning planner_branch dataset_branch sim_branch)"
GENERATOR="$(find_dir tools/ahead_dataset_generator dataset_branch planner_branch sim_branch)"
PAC_SIM="$(find_dir ros2_ws/src/pac_simulation sim_branch dataset_branch planner_branch)"
DONGHAN_ROOT=""
if [ -n "$PAC_PLANNING" ]; then
  DONGHAN_ROOT="$(dirname "$(dirname "$(dirname "$PAC_PLANNING")")")"
fi

cat > "$OUT/paths.env" <<ENV
PAC_COMMON_SRC=$PAC_COMMON
PAC_PLANNING_SRC=$PAC_PLANNING
PAC_PLANNING_ROOT=$DONGHAN_ROOT
AHEAD_GENERATOR_ROOT=$GENERATOR
PAC_SIMULATION_SRC=$PAC_SIM
ENV
cat "$OUT/SOURCES.txt"
cat "$OUT/paths.env"
