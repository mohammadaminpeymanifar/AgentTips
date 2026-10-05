from apscheduler.schedulers.blocking import BlockingScheduler
from run import run_pipeline

scheduler = BlockingScheduler()
scheduler.add_job(run_pipeline, 'cron', hour=9, minute=0)

print("📅 Scheduler active... Press Ctrl+C to stop.")
try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    pass