#!/bin/sh
# Runs a command until it succeeds, at most 6 times, 5 minutes apart. For GPU memory taken by other programs.
n=1
while :; do
  "$@" && exit 0
  rc=$?
  [ "$n" -ge 6 ] && { echo "GAVE UP after $n attempts (rc=$rc): $*"; exit "$rc"; }
  echo "attempt $n failed (rc=$rc); waiting 5 minutes: $*"
  n=$((n + 1))
  sleep 300
done
