from apscheduler.schedulers.blocking import BlockingScheduler
from run import run_pipeline

scheduler = BlockingScheduler()

# Run every day at 09:00 AM
scheduler.add_job(run_pipeline, 'cron', hour=9, minute=0)

print("📅 Tips Agent Scheduler running... Press Ctrl+C to exit.")
try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    pass