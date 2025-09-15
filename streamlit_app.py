import io
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# -----------------------------
# Utilities
# -----------------------------

KOREAN_BASIC_STOPWORDS: List[str] = [
    "은", "는", "이", "가", "을", "를", "에", "에서", "의", "와", "과", "도", "으로",
    "하고", "한", "하다", "했습니다", "합니다", "그리고", "하지만", "그래서", "그러나",
    "혹은", "또는", "때문", "때문에", "대한", "관련", "합니다.", "입니다", "있어요",
]


def split_text_to_tokens(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    import re

    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", text.lower())
    tokens = [t for t in tokens if t not in KOREAN_BASIC_STOPWORDS and len(t) > 1]
    return tokens


def label_sentiment(compound_score: float) -> str:
    if compound_score >= 0.05:
        return "긍정"
    if compound_score <= -0.05:
        return "부정"
    return "중립"


@st.cache_data(show_spinner=False)
def read_local_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path) and os.path.isfile(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def parse_date_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    series = pd.to_datetime(df[column_name], errors="coerce")
    return series


@st.cache_data(show_spinner=False)
def run_sentiment_analysis(texts: List[str]) -> Tuple[np.ndarray, List[str]]:
    analyzer = SentimentIntensityAnalyzer()
    scores = np.array([analyzer.polarity_scores(t or "").get("compound", 0.0) for t in texts])
    labels = [label_sentiment(s) for s in scores]
    return scores, labels


@st.cache_data(show_spinner=False)
def extract_top_keywords(texts: List[str], top_k: int = 20) -> pd.DataFrame:
    if len(texts) == 0:
        return pd.DataFrame({"keyword": [], "score": []})

    vectorizer = TfidfVectorizer(
        tokenizer=split_text_to_tokens,
        preprocessor=lambda x: x,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
    )
    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame({"keyword": [], "score": []})

    scores = np.asarray(tfidf.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())

    order = np.argsort(scores)[::-1][:top_k]
    return pd.DataFrame({"keyword": terms[order], "score": scores[order]})


def filter_dataframe(
    df: pd.DataFrame,
    date_column: Optional[str],
    category_column: Optional[str],
    date_range: Optional[Tuple[datetime, datetime]],
    categories_selected: Optional[List[str]],
) -> pd.DataFrame:
    filtered = df.copy()

    if date_column and date_range is not None:
        parsed = parse_date_column(filtered, date_column)
        start, end = date_range
        mask = (parsed >= pd.Timestamp(start)) & (parsed <= pd.Timestamp(end))
        filtered = filtered[mask]

    if category_column and categories_selected:
        filtered = filtered[filtered[category_column].isin(categories_selected)]

    return filtered


def ensure_text_column(df: pd.DataFrame) -> Optional[str]:
    candidate_names = [
        "text", "feedback", "review", "comment", "content", "message", "summary",
        "텍스트", "피드백", "리뷰", "코멘트", "내용", "메시지",
    ]
    for name in candidate_names:
        if name in df.columns:
            return name
    # fallback: longest object column
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    return object_cols[0] if object_cols else None


def main() -> None:
    st.set_page_config(page_title="고객 피드백 분석", page_icon="🧠", layout="wide")
    st.title("고객 피드백 분석 (Streamlit)")
    st.caption("CSV 또는 Excel을 업로드하고 감성 분석과 키워드 추출을 실행하세요.")

    with st.sidebar:
        st.header("데이터 업로드")
        uploaded = st.file_uploader(
            "피드백 데이터 파일 업로드 (CSV 또는 Excel)", type=["csv", "xlsx", "xls"]
        )
        st.markdown("---")
        st.caption("로컬의 '@feedback-data.csv' 자동 탐지 시 기본 로드")

    df: Optional[pd.DataFrame] = None
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                import openpyxl  # noqa: F401  # ensure dependency
                df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    else:
        df = read_local_csv_if_exists(os.path.join(os.getcwd(), "@feedback-data.csv"))

    if df is None or df.empty:
        st.info(
            "데이터를 업로드하세요. 예: '@feedback-data.csv' 또는 Excel 파일. "
            "텍스트 컬럼 1개 이상이 필요합니다."
        )
        return

    st.subheader("원본 데이터 미리보기")
    st.dataframe(df.head(200), use_container_width=True)

    with st.sidebar:
        st.header("컬럼 매핑 및 필터")
        text_column_default = ensure_text_column(df)
        text_column = st.selectbox(
            "텍스트 컬럼 선택",
            options=list(df.columns),
            index=list(df.columns).index(text_column_default) if text_column_default in df.columns else 0,
        )

        possible_date_cols = [c for c in df.columns if "date" in c.lower() or "날짜" in c]
        date_column = st.selectbox(
            "날짜 컬럼(선택)", options=["(없음)"] + list(df.columns), index=0
        )
        date_column = None if date_column == "(없음)" else date_column

        category_column = st.selectbox(
            "카테고리/제품 컬럼(선택)", options=["(없음)"] + list(df.columns), index=0
        )
        category_column = None if category_column == "(없음)" else category_column

        date_range: Optional[Tuple[datetime, datetime]] = None
        if date_column:
            date_series = parse_date_column(df, date_column)
            min_date = pd.to_datetime(date_series.min())
            max_date = pd.to_datetime(date_series.max())
            if pd.isna(min_date) or pd.isna(max_date):
                st.warning("선택한 날짜 컬럼에 유효한 값이 적습니다. 필터를 건너뜁니다.")
            else:
                date_range = st.date_input(
                    "기간 필터",
                    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    date_range = (date_range[0], date_range[1])
                else:
                    date_range = None

        categories_selected: Optional[List[str]] = None
        if category_column:
            all_categories = sorted(list(map(str, df[category_column].dropna().unique())))
            categories_selected = st.multiselect("카테고리 선택", options=all_categories)

        st.markdown("---")
        top_k = st.slider("상위 키워드 개수", min_value=10, max_value=50, value=20, step=5)
        run_button = st.button("분석 실행", type="primary")

    if not run_button:
        st.info("좌측 사이드바에서 옵션을 설정하고 '분석 실행'을 클릭하세요.")
        return

    working_df = filter_dataframe(df, date_column, category_column, date_range, categories_selected)
    if working_df.empty:
        st.warning("필터 결과가 없습니다. 조건을 완화해 보세요.")
        return

    texts = working_df[text_column].fillna("").astype(str).tolist()

    with st.spinner("감성 분석 중..."):
        compound_scores, labels = run_sentiment_analysis(texts)

    working_df = working_df.copy()
    working_df["sentiment_score"] = compound_scores
    working_df["sentiment_label"] = labels

    st.subheader("감성 분포")
    label_counts = (
        working_df["sentiment_label"].value_counts().rename_axis("label").reset_index(name="count")
    )
    fig_bar = px.bar(
        label_counts,
        x="label",
        y="count",
        color="label",
        text="count",
        color_discrete_map={"긍정": "#2ca02c", "중립": "#ff7f0e", "부정": "#d62728"},
        title="감성 라벨 분포",
    )
    fig_bar.update_layout(xaxis_title="라벨", yaxis_title="개수")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("상위 키워드")
    with st.spinner("키워드 추출 중..."):
        keyword_df = extract_top_keywords(texts, top_k=top_k)
    if keyword_df.empty:
        st.info("키워드를 추출할 수 없습니다. 데이터가 너무 적거나 일치 토큰이 부족합니다.")
    else:
        fig_kw = px.bar(
            keyword_df.iloc[::-1],
            x="score",
            y="keyword",
            orientation="h",
            title="TF-IDF 기반 상위 키워드",
        )
        fig_kw.update_layout(xaxis_title="점수", yaxis_title="키워드")
        st.plotly_chart(fig_kw, use_container_width=True)

    with st.expander("분석 결과 데이터"):
        st.dataframe(working_df.head(1000), use_container_width=True)
        csv = working_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("결과 CSV 다운로드", data=csv, file_name="feedback_analysis_result.csv", mime="text/csv")

    st.success("분석이 완료되었습니다.")


if __name__ == "__main__":
    main()



