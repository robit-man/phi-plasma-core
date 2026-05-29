#!/usr/bin/env bash
# Daily snapshot of GitHub Traffic API + repo engagement metrics.
#
# Captures the rolling 14-day window each day so you accumulate a permanent
# record. After ~30 days you can correlate channel events (tweets, paper
# drops, HN/Reddit hits) to clone-spike timing.
#
# Output: ~/.github_traffic_logs/{owner}-{repo}/YYYY-MM-DD.json

set -euo pipefail

REPO="${REPO:-Prime-007-hash/phi-plasma-core}"
OUTDIR="${OUTDIR:-$HOME/.github_traffic_logs/${REPO//\//-}}"
mkdir -p "$OUTDIR"
TODAY=$(date -u +%Y-%m-%d)
OUTFILE="$OUTDIR/$TODAY.json"
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Pull all endpoints to temp files
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

gh api "repos/$REPO" > "$TMPDIR/repo.json"
gh api "repos/$REPO/traffic/views" > "$TMPDIR/views.json"
gh api "repos/$REPO/traffic/clones" > "$TMPDIR/clones.json"
gh api "repos/$REPO/traffic/popular/referrers" > "$TMPDIR/referrers.json"
gh api "repos/$REPO/traffic/popular/paths" > "$TMPDIR/paths.json"
gh api "repos/$REPO/stargazers" > "$TMPDIR/stargazers.json"
gh api "repos/$REPO/issues?state=all&per_page=100" > "$TMPDIR/issues.json"

# Assemble via python (no jq dependency)
python3 - "$TS" "$REPO" "$TMPDIR" "$OUTFILE" <<'PY'
import json, sys
ts, repo, tmpdir, outfile = sys.argv[1:5]

def load(name):
    with open(f"{tmpdir}/{name}.json") as f:
        return json.load(f)

r = load("repo")
v = load("views")
c = load("clones")
ref = load("referrers")
paths = load("paths")
stars = load("stargazers")
issues = load("issues")

snap = {
    "snapshot_ts": ts,
    "repo": repo,
    "engagement": {
        "stars": r.get("stargazers_count", 0),
        "forks": r.get("forks_count", 0),
        "watchers": r.get("subscribers_count", 0),
        "open_issues": r.get("open_issues_count", 0),
        "size_kb": r.get("size", 0),
        "created_at": r.get("created_at", ""),
    },
    "traffic": {
        "views": v,
        "clones": c,
        "referrers": ref,
        "popular_paths": paths,
    },
    "stargazers": [{"login": s["login"], "id": s["id"]} for s in stars],
    "issues": [
        {
            "number": i["number"],
            "title": i["title"],
            "state": i["state"],
            "user": i["user"]["login"] if i.get("user") else None,
            "created_at": i.get("created_at"),
            "closed_at": i.get("closed_at"),
            "comments": i.get("comments", 0),
        }
        for i in issues
    ],
}

with open(outfile, "w") as f:
    json.dump(snap, f, indent=2)

# Quick summary to stderr
e = snap["engagement"]
v = snap["traffic"]["views"]
c = snap["traffic"]["clones"]
print(f"[{ts}] snapshot: {outfile}", file=sys.stderr)
print(
    f"  stars={e['stars']}  views_14d={v.get('count', 0)} ({v.get('uniques', 0)} uniq)  "
    f"clones_14d={c.get('count', 0)} ({c.get('uniques', 0)} uniq)  "
    f"referrers={len(snap['traffic']['referrers'])}  issues={len(snap['issues'])}",
    file=sys.stderr,
)
PY
