#!/usr/bin/env bash
# Cron-driven, session-independent seeder for the checklist pipeline.
# Runs every 10 min: when a SLURM submit slot is free (queue < 3), seeds
# jobs/pipeline.sh 1, then removes itself from crontab. Fires at most once
# (guard file). Robust across login sessions, unlike a background shell.
cd /home2/home/ankur_d/sm/UltraBreak-Repro
GUARD="logs/.pipeline_seeded"
unschedule() { crontab -l 2>/dev/null | grep -v 'jobs/cron_seed.sh' | crontab - 2>/dev/null; }
[ -f "$GUARD" ] && { unschedule; exit 0; }
c=$(squeue -h -u "$USER" 2>/dev/null | wc -l)
if [ "$c" -lt 3 ]; then
  jid=$(sbatch --parsable jobs/pipeline.sh 1 2>&1)
  echo "$(date '+%F %T') seeded pipeline step 1 as job ${jid} (queue was ${c})" >> logs/cron_seed.log
  touch "$GUARD"
  unschedule
fi
