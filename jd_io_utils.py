"""
io_utils.py

입출력(I/O) 관련 함수들을 모아서 정리한 모듈.
주식 데이터 CSV/JSON/Pickle 파일 입출력,
기본 폴더 생성 로직 등을 포함.
"""

import json
import logging
import os
import pickle

import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 폴더 경로 설정
# -----------------------------------------------------------------------------
from jdGlobal import DATA_FOLDER
from jdGlobal import METADATA_FOLDER
from jdGlobal import SAVE_FOLDER
from jdGlobal import SCREENSHOT_FOLDER
from jdGlobal import FILTERED_STOCKS_FOLDER
from jdGlobal import PROFILES_FOLDER

# -----------------------------------------------------------------------------
# 폴더 생성 함수
# -----------------------------------------------------------------------------
def ensure_directories_exist():
    """
    필요한 폴더들이 존재하는지 확인하고,
    없으면 생성합니다.
    """
    for folder in [
        DATA_FOLDER,
        METADATA_FOLDER,
        SAVE_FOLDER,
        FILTERED_STOCKS_FOLDER,
        SCREENSHOT_FOLDER,
        PROFILES_FOLDER
    ]:
        if not os.path.exists(folder):
            os.makedirs(folder)

# -----------------------------------------------------------------------------
# CSV 입출력
# -----------------------------------------------------------------------------
def load_csv_with_date_index(
    ticker: str,
    data_dir: str = DATA_FOLDER,
    start_date=None,
    end_date=None
) -> pd.DataFrame:
    """
    지정한 data_dir 내 티커명(ticker).csv를 읽어
    Date 열을 datetime 인덱스로 설정 후 (start_date ~ end_date)로 슬라이싱한 뒤 반환.
    파일이 없거나 에러가 발생하면 빈 DataFrame을 반환합니다.
    """
    csv_path = os.path.join(data_dir, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        logger.debug("[load_csv_with_date_index] File not found: %s", csv_path)
        return pd.DataFrame()

    try:
        # parse_dates=['Date'] 옵션을 사용해 'Date' 열을 바로 datetime으로 파싱하고, 이후 인덱스로 설정
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)

        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        logger.exception("[load_csv_with_date_index] Failed to read CSV: %s", csv_path)
        return pd.DataFrame()


def save_df_to_csv(
    df: pd.DataFrame,
    ticker: str,
    data_dir: str = DATA_FOLDER,
    encoding: str = "utf-8-sig"
):
    """
    DataFrame을 data_dir 내 ticker.csv 로 저장합니다.
    """
    save_path = os.path.join(data_dir, f"{ticker}.csv")
    try:
        df.to_csv(save_path, encoding=encoding)
        logger.info("%s.csv saved to %s", ticker, save_path)
    except (OSError, ValueError):
        logger.exception("[save_df_to_csv] Failed to save CSV: %s", save_path)

# -----------------------------------------------------------------------------
# JSON 입출력
# -----------------------------------------------------------------------------
def save_to_json(data, filename: str, folder: str = METADATA_FOLDER):
    """
    딕셔너리나 리스트 등 직렬화 가능한 Python 객체를
    folder/filename.json에 저장합니다.
    """
    full_path = os.path.join(folder, f"{filename}.json")
    try:
        with open(full_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        logger.info("[save_to_json] Saved: %s", full_path)
    except (OSError, TypeError):
        logger.exception("[save_to_json] Failed to save JSON: %s", full_path)


def load_from_json(filename: str, folder: str = METADATA_FOLDER):
    """
    folder/filename.json 파일을 로드하여 Python 객체로 반환.
    에러가 발생하면 빈 dict를 반환.
    """
    full_path = os.path.join(folder, f"{filename}.json")
    if not os.path.exists(full_path):
        logger.debug("[load_from_json] File not found: %s", full_path)
        return {}

    try:
        with open(full_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        logger.exception("[load_from_json] Failed to load JSON: %s", full_path)
        return {}

# -----------------------------------------------------------------------------
# Pickle 입출력
# -----------------------------------------------------------------------------
def save_pickle(data, filename: str, folder: str = METADATA_FOLDER):
    """
    data 객체를 folder/filename.pkl 로 직렬화하여 저장합니다.
    """
    full_path = os.path.join(folder, f"{filename}.pkl")
    try:
        with open(full_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("[save_pickle] Saved: %s", full_path)
    except (OSError, pickle.PicklingError):
        logger.exception("[save_pickle] Failed to save pickle: %s", full_path)


def load_pickle(filename: str, folder: str = METADATA_FOLDER):
    """
    folder/filename.pkl 파일을 로드하여 Python 객체로 복원합니다.
    """
    full_path = os.path.join(folder, f"{filename}.pkl")
    if not os.path.exists(full_path):
        logger.debug("[load_pickle] File not found: %s", full_path)
        return None

    try:
        with open(full_path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError):
        logger.exception("[load_pickle] Failed to load pickle: %s", full_path)
        return None


def quarantine_corrupted_csv_files(
    data_dir: str = DATA_FOLDER,
    metadata_dir: str = METADATA_FOLDER,
    quarantine_dir_name: str = "_corrupted",
    sample_bytes: int = 4096,
    strict: bool = True,
):
    """
    StockData 내 CSV를 스캔하여 손상/형식 불일치 파일을 격리합니다.

    - 빈 파일, NUL byte 포함, 헤더에 Date 컬럼이 없는 경우를 '손상'으로 간주합니다.
    - 손상 파일은 data_dir/{quarantine_dir_name}/ 아래로 이동합니다.
    - 격리 내역은 metadata_dir/corrupted_stockdata.json 에 누적 기록됩니다.
    """
    if not os.path.isdir(data_dir):
        logger.warning("[quarantine_corrupted_csv_files] data_dir not found: %s", data_dir)
        return {"checked": 0, "quarantined": []}

    os.makedirs(metadata_dir, exist_ok=True)
    quarantine_dir = os.path.join(data_dir, quarantine_dir_name)
    os.makedirs(quarantine_dir, exist_ok=True)

    def _unique_dest_path(dest_path: str) -> str:
        if not os.path.exists(dest_path):
            return dest_path
        base, ext = os.path.splitext(dest_path)
        for i in range(1, 1000):
            candidate = f"{base}_{i}{ext}"
            if not os.path.exists(candidate):
                return candidate
        return f"{base}_{os.getpid()}{ext}"

    def _detect_issue(csv_path: str) -> str:
        try:
            size = os.path.getsize(csv_path)
        except OSError:
            logger.exception("[quarantine_corrupted_csv_files] stat failed: %s", csv_path)
            return "stat_failed"

        if size == 0:
            return "empty_file"

        try:
            with open(csv_path, "rb") as f:
                sample = f.read(sample_bytes)
        except OSError:
            logger.exception("[quarantine_corrupted_csv_files] read failed: %s", csv_path)
            return "read_failed"

        if b"\x00" in sample:
            return "contains_nul_byte"

        if not strict:
            return ""

        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                header_line = (f.readline() or "").strip()
        except OSError:
            logger.exception("[quarantine_corrupted_csv_files] header read failed: %s", csv_path)
            return "header_read_failed"

        if not header_line:
            return "missing_header"

        columns = [c.strip().strip('"').lstrip("\ufeff") for c in header_line.split(",")]
        if "Date" not in columns:
            return "missing_date_column"

        return ""

    checked = 0
    quarantined = []
    for filename in os.listdir(data_dir):
        if filename == quarantine_dir_name:
            continue
        if not filename.lower().endswith(".csv"):
            continue

        csv_path = os.path.join(data_dir, filename)
        if not os.path.isfile(csv_path):
            continue

        checked += 1
        issue = _detect_issue(csv_path)
        if not issue:
            continue

        ticker = os.path.splitext(filename)[0]
        dest_path = _unique_dest_path(os.path.join(quarantine_dir, filename))
        try:
            os.replace(csv_path, dest_path)
        except OSError:
            logger.exception("[quarantine_corrupted_csv_files] move failed: %s -> %s", csv_path, dest_path)
            continue

        record = {
            "ticker": ticker,
            "filename": filename,
            "issue": issue,
            "src": csv_path,
            "dst": dest_path,
        }
        quarantined.append(record)
        logger.warning("[quarantine_corrupted_csv_files] quarantined %s (%s)", filename, issue)

    report_path = os.path.join(metadata_dir, "corrupted_stockdata.json")
    if quarantined:
        try:
            existing = []
            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    existing = json.load(f) or []
                    if not isinstance(existing, list):
                        existing = []
            existing.extend(quarantined)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("[quarantine_corrupted_csv_files] failed to write report: %s", report_path)

    logger.info(
        "[quarantine_corrupted_csv_files] checked=%s quarantined=%s report=%s",
        checked,
        len(quarantined),
        report_path,
    )
    return {"checked": checked, "quarantined": quarantined, "report_path": report_path}
