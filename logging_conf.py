import datetime
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler 

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
#console_handler = logging.StreamHandler(sys.stdout)

# 기존 console_handler 대신 교체
console_handler = RichHandler(
    rich_tracebacks=True,               # Traceback을 Rich 스타일로
    show_time=False                     # 이미 포맷터에 시간이 있으므로
)

console_handler.setFormatter(formatter)
console_handler.addFilter(MyProjectFilter())
console_handler.setLevel(logging.DEBUG) # DEBUG 이상 로그는 모두 터미널에도 출력

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
