"""
io_utils.py

입출력(I/O) 관련 함수들을 모아서 정리한 모듈.
주식 데이터 CSV/JSON/Pickle 파일 입출력,
기본 폴더 생성 로직 등을 포함.
"""

import os
import json
import pickle
import pandas as pd

import logging

# -----------------------------------------------------------------------------
# 폴더 경로 설정
# -----------------------------------------------------------------------------
from jdGlobal import DATA_FOLDER
from jdGlobal import METADATA_FOLDER
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
        print(f"[load_csv_with_date_index] File not found: {csv_path}")
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
    except Exception as e:
        print(f"[load_csv_with_date_index] An error occurred: {e}")
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
        logging.info(f"{ticker}.csv saved to {save_path}")
    except Exception as e:
        logging.info(f"[save_df_to_csv] An error occurred: {e}")

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
        print(f"dictionary data is saved as {full_path}")
    except Exception as e:
        print(f"[save_to_json] An error occurred: {e}")


def load_from_json(filename: str, folder: str = METADATA_FOLDER):
    """
    folder/filename.json 파일을 로드하여 Python 객체로 반환.
    에러가 발생하면 빈 dict를 반환.
    """
    full_path = os.path.join(folder, f"{filename}.json")
    if not os.path.exists(full_path):
        print(f"[load_from_json] File not found: {full_path}")
        return {}

    try:
        with open(full_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"[load_from_json] An error occurred: {e}")
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
        print(f"[save_pickle] Saved: {full_path}")
    except Exception as e:
        print(f"[save_pickle] An error occurred: {e}")


def load_pickle(filename: str, folder: str = METADATA_FOLDER):
    """
    folder/filename.pkl 파일을 로드하여 Python 객체로 복원합니다.
    """
    full_path = os.path.join(folder, f"{filename}.pkl")
    if not os.path.exists(full_path):
        print(f"[load_pickle] File not found: {full_path}")
        return None

    try:
        with open(full_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[load_pickle] An error occurred: {e}")
        return None
