#!/usr/bin/env bash
# Create or update this suite's PR comment, found by the hidden marker on its first line,
# so each suite keeps its own comment and reruns edit it in place.
# Usage: upsert-comment.sh <repo> <pr-number> <comment-file>
set -euo pipefail
repo="$1" pr="$2" file="$3"
marker="$(head -n 1 "$file")"
case "$marker" in
  "<!-- llm-evals:report:"*" -->") ;;
  *) echo "first line of $file is not an llm-evals marker: $marker" >&2; exit 1 ;;
esac
id="$(gh api --paginate "repos/$repo/issues/$pr/comments" \
  --jq ".[] | select(.user.type == \"Bot\" and (.body | startswith(\"$marker\"))) | .id" | tail -n 1)"
if [ -n "$id" ]; then
  gh api --method PATCH "repos/$repo/issues/comments/$id" -F "body=@$file" > /dev/null
  echo "updated comment $id"
else
  gh api --method POST "repos/$repo/issues/$pr/comments" -F "body=@$file" > /dev/null
  echo "created a comment"
fi
