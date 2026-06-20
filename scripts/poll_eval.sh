#!/bin/bash
for i in $(seq 1 90); do
  if [ -f /workspace/outputs/mvp_side_by_side_h200_v2_metrics.json ]; then
    echo DONE
    exit 0
  fi
  echo poll_$i
  wc -c /workspace/eval_h200_v2.log 2>/dev/null || true
  sleep 20
done
echo TIMEOUT
exit 1
