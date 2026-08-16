#!/usr/bin/env bash
# Watcher: waits for a free SLURM submit slot (MaxSubmit=3), then seeds the
# self-chaining pipeline at step 1. Exits after seeding (or on ~26h timeout).
cd /home2/home/ankur_d/sm/UltraBreak-Repro
for i in $(seq 1 312); do
  c=$(squeue -h -u "$USER" 2>/dev/null | wc -l)
  if [ "$c" -lt 3 ]; then
    jid=$(sbatch --parsable jobs/pipeline.sh 1 2>&1)
    echo "SEEDED pipeline step 1 as job $jid (queue had $c jobs)"
    exit 0
  fi
  sleep 300
done
echo "seed watcher timed out after ~26h without a free slot"
exit 1
