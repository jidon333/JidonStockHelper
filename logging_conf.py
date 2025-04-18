import logging, sys, os, datetime
from logging.handlers import RotatingFileHandler

# Directory that stores all log files
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Filter: keep logs only from our jd‑ modules or from main.py
class MyProjectFilter(logging.Filter):
    def filter(self, record):
        return record.filename.startswith("jd") or record.filename in {"__main__", "main.py"}

# Log file name: JidonStockHelper_YYYYmmdd_HHMMSS.log
log_filename = os.path.join(
    LOG_DIR,
    datetime.datetime.now().strftime("JidonStockHelper_%Y%m%d_%H%M%S.log")
)

# Log line format
FMT = "%(asctime)s [%(levelname)s] [%(name)s - %(module)s:%(lineno)d %(funcName)s()] %(message)s"
formatter = logging.Formatter(FMT)

# File handler (rolls over at 5 MB, keeps 3 backups)
file_handler = RotatingFileHandler(
    log_filename,
    maxBytes=10_000_000,
    backupCount=10,
    encoding="utf-8",
)
file_handler.setFormatter(formatter)
file_handler.addFilter(MyProjectFilter())

# Console handler (same format and filter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.addFilter(MyProjectFilter())

# Root logger configuration
logging.basicConfig(
    level=logging.DEBUG,          # capture DEBUG and above
    handlers=[file_handler, console_handler],
)


# ------------------------------------------------------------------
# Trim noisy third‑party loggers
# ------------------------------------------------------------------
for lib in ("urllib3", "requests", "yahooquery", "matplotlib"):
    logging.getLogger(lib).propagate = False   # don't bubble to root
    logging.getLogger(lib).setLevel(logging.WARNING)
