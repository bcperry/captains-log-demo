#!/bin/bash
set -e

# Usage: ralph.sh [target_dir] [max_iterations] [model]
TARGET_DIR=${1:-.}
MAX_ITERATIONS=${2:-30}
MODEL=${3:-"claude-opus-4.5"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$TARGET_DIR" && pwd)"

echo "🚀 Starting Ralph"
echo "   Model: $MODEL"
echo "   Max iterations: $MAX_ITERATIONS"
echo "   Workspace: $WORKSPACE_ROOT"

for i in $(seq 1 $MAX_ITERATIONS); do
  echo ""
  echo "═══════════════════════════════════════"
  echo "═══ Iteration $i of $MAX_ITERATIONS ═══"
  echo "═══════════════════════════════════════"
  
  PROMPT=$(cat "$SCRIPT_DIR/prompt.md")
  
  OUTPUT=$(cd "$WORKSPACE_ROOT" && copilot \
    --prompt "$PROMPT" \
    --model "$MODEL" \
    --allow-all-tools \
    --allow-all-paths \
    --add-dir "$WORKSPACE_ROOT" \
    --silent \
    2>&1 | tee /dev/stderr) || true
  
  if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
    echo ""
    echo "✅ All stories complete!"
    exit 0
  fi
  
  echo ""
  echo "⏳ Waiting before next iteration..."
  sleep 2
done

echo ""
echo "⚠️ Max iterations ($MAX_ITERATIONS) reached"
exit 1
