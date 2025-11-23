# jd_filter_utils.py
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
import logging

import numpy as np
import pandas as pd


# ── 기본값 (필요하면 여기만 수정) ──────────────────────────────
DEFAULT_VOL_WINDOW = 50
# ────────────────────────────────────────────────────────────

def precheck(
        cols:        Optional[List[str]] = None,
        min_volume:  Optional[int]       = None,
        min_price:   Optional[float]     = None,
        vol_window:  int = DEFAULT_VOL_WINDOW
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    필터 함수에 붙여서 공통 전처리를 수행하는 파라미터화 데코레이터.

    Parameters
    ----------
    cols        : 반드시 존재하고 NaN 이 아니어야 할 컬럼 목록
    min_volume  : 〈vol_window〉 일 평균 거래량 하한 (None 이면 체크 안 함)
    min_price   : 종가 하한 (None 이면 체크 안 함)
    vol_window  : 거래량 이동평균 창 길이
    """
    cols = cols or []

    # ── 원본 함수를 받아 래퍼를 만들어 주는 '데코레이터'
    def decorator(fn: Callable) -> Callable:
        # 메타데이터 복사
        @wraps(fn)  
        def wrapper(self,
                    stock_dic: Dict[str, pd.DataFrame],
                    n: int = -1,
                    *args, **kwargs):
            passed: Dict[str, pd.DataFrame] = {}

            for tic, df in stock_dic.items():

                try:
                    # 데이터 길이 부족하면 건너뜀
                    if abs(n) >= len(df):
                        continue

                    # 1) 컬럼 존재 & NaN 체크
                    if any(c not in df or pd.isna(df[c].iloc[n]) for c in cols):
                        continue

                    # 2) 거래량 체크
                    if min_volume is not None:
                        vol_ma = df["Volume"].rolling(vol_window).mean().iloc[n]
                        if np.isnan(vol_ma) or vol_ma < min_volume:
                            continue

                    # 3) 종가 체크
                    if min_price is not None and df["Close"].iloc[n] < min_price:
                        continue
                except Exception as e:
                    logging.info(f"[DEBUG] IndexError @ ticker={tic} | len={len(df)} | requested n={n}")
                    continue

                passed[tic] = df

            # ---- 전처리 통과 dict 를 원본 필터로 전달 ----
            return fn(self, passed, n, *args, **kwargs)

        return wrapper

    return decorator          # ← “맞춤형 데코레이터” 반환
