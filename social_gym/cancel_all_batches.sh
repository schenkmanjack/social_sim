#!/usr/bin/env bash
set -euo pipefail

API="https://api.anthropic.com/v1/messages/batches"
HEADERS=(-H "x-api-key:$ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01")

while true; do
  # --- 1. fetch ALL batches (paginate) -------------------------------
  next_cursor=""
  open_batches=()
  while :; do
    resp=$(curl -s "${API}?limit=100${next_cursor}" "${HEADERS[@]}")
    # collect ids whose status is NOT ended/canceled
    mapfile -t ids < <(echo "$resp" | jq -r '.data[]
         | select(.processing_status|test("ended|canceled")|not) | .id')
    open_batches+=("${ids[@]}")

    has_more=$(echo "$resp" | jq '.has_more')
    last_id=$(echo "$resp"  | jq -r '.last_id')
    [[ $has_more == true ]] || break
    next_cursor="&starting_after=${last_id}"
  done

  [[ ${#open_batches[@]} -eq 0 ]] && { echo "✔ No open batches"; break; }

  # --- 2. cancel every open batch ------------------------------------
  for BID in "${open_batches[@]}"; do
    echo "→ Canceling $BID"
    curl -s -X POST "$API/$BID/cancel" "${HEADERS[@]}" >/dev/null
  done

  # --- 3. wait a few seconds, then loop again ------------------------
  sleep 5
done