# app.py (데이터 직접 입력 방식으로 수정)

import streamlit as st
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------
# Colab의 핵심 분석 로직을 함수로 정의
# (안정성 향상을 위해 컬럼 위치 대신 이름으로 직접 접근하도록 수정)
# --------------------------------------------------------------------------
def run_keyword_analysis(df: pd.DataFrame):
    try:
        # --- 데이터 컬럼 이름 정의 ---
        # data_editor의 컬럼 이름을 직접 사용하므로, 더 안정적입니다.
        col_keyword = '키워드'
        col_cost = '총비용'
        col_clicks = '클릭수'
        col_conversions = '전환수'
        
        # 데이터 타입 변환 (오류 방지)
        df[col_cost] = pd.to_numeric(df[col_cost], errors='coerce').fillna(0)
        df[col_clicks] = pd.to_numeric(df[col_clicks], errors='coerce').fillna(0)
        df[col_conversions] = pd.to_numeric(df[col_conversions], errors='coerce').fillna(0)
        df = df.dropna(subset=[col_keyword]) # 키워드 없는 행 제거
        df = df[df[col_keyword].astype(str).str.strip() != ''] # 빈 키워드 제거

        # --- 1. 지표 계산 ---
        conv_eff_list = []      # 전환 효율 리스트
        conv_rate_list = []     # 전환율 리스트
        conv_contrib_list = []  # 전환 기여도 리스트
        
        total_conversions = df[col_conversions].sum()

        for i in range(len(df)):
            cost = df.loc[i, col_cost]
            clicks = df.loc[i, col_clicks]
            conversions = df.loc[i, col_conversions]

            conv_eff = conversions / cost if cost != 0 else 0
            conv_rate = (conversions / clicks) * 100 if clicks != 0 else 0
            conv_contrib = (conversions / total_conversions) * 100 if total_conversions != 0 else 0
            
            conv_eff_list.append(conv_eff)
            conv_rate_list.append(conv_rate)
            conv_contrib_list.append(conv_contrib)

        # --- 2. 정규화 및 점수 계산 ---
        def normalize(lst):
            lst = [0 if pd.isna(x) else x for x in lst]
            min_val = min(lst)
            max_val = max(lst)
            range_val = max_val - min_val if max_val != min_val else 1
            return [(x - min_val) / range_val * 100 for x in lst]

        norm_eff_list = normalize(conv_eff_list)
        norm_rate_list = normalize(conv_rate_list)
        norm_contrib_list = normalize(conv_contrib_list)

        total_scores = [
            round(contrib * 0.5 + rate * 0.25 + eff * 0.25, 2)
            for contrib, rate, eff in zip(norm_contrib_list, norm_rate_list, norm_eff_list)
        ]

        # --- 3. 전략 분류 및 최종 결과 생성 ---
        score_series = pd.Series(total_scores)
        score_rank = score_series.rank(ascending=False, method='min')
        total = len(score_series)
        result_list = []

        for i in range(len(df)):
            keyword = df.loc[i, col_keyword]
            cost = df.loc[i, col_cost]
            clicks = df.loc[i, col_clicks]
            conversions = df.loc[i, col_conversions]
            
            cpa = cost / conversions if conversions != 0 else 0
            conv_rate_val = (conversions / clicks) * 100 if clicks != 0 else 0
            contrib = conv_contrib_list[i]
            total_score = total_scores[i]
            rank_percentile = score_rank[i] / total

            if total_score == 0 or pd.isna(total_score):
                strategy = "🔴 중단 전략: 광고 제거, 타겟·콘텐츠 리설정"
            elif rank_percentile <= 0.05:
                strategy = "🔵 강화 전략: 예산 증액, 콘텐츠 확장, 유사 타겟 확장"
            elif rank_percentile <= 0.20:
                strategy = "🟢 유지 전략: 예산 유지, 콘텐츠·타겟 유지"
            elif rank_percentile <= 0.50:
                strategy = "🟡 보완 전략: 일부 콘텐츠 개선, 타겟 정교화"
            else:
                strategy = "🟠 축소 전략: 예산 축소, 리테스트 필요"
            
            result_list.append([
                keyword,
                round(contrib, 2),
                round(conv_rate_val, 2),
                round(cpa, 2) if cpa else 0,
                total_score,
                strategy
            ])
        
        result_df = pd.DataFrame(
            result_list,
            columns=["키워드", "전환 기여도 (%)", "전환율 (%)", "CPA (원)", "종합 점수", "전략 추천"]
        )
        result_df = result_df.sort_values(by="종합 점수", ascending=False).reset_index(drop=True)
        
        return result_df

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
        st.error("입력 데이터의 형식을 확인해주세요. 모든 칸이 올바르게 채워져야 합니다.")
        return None

# --------------------------------------------------------------------------
# Streamlit 웹페이지 UI 구성
# --------------------------------------------------------------------------

# 페이지 레이아웃 넓게 설정
st.set_page_config(layout="wide")

# 제목
st.title('🔑 소재/키워드 성과 분석')
st.write("아래 표에 데이터를 직접 붙여넣거나 입력한 후, '분석 시작하기' 버튼을 누르세요.")
st.write("---")

# --- 데이터 입력 UI (수정된 부분) ---
st.subheader("📋 데이터 입력")
st.info("엑셀이나 구글 시트에서 [키워드, 총비용, 클릭수, 전환수] 데이터를 복사한 후, 아래 표의 첫 번째 칸을 클릭하고 붙여넣기(Ctrl+V) 하세요.")

# 사용자가 데이터를 붙여넣기 쉽도록 예시 데이터가 포함된 빈 데이터프레임 생성
sample_data = {
    '키워드': ['볼보 XC60', '볼보 S90 가격', '전기차 보조금', '수입 SUV 추천', '안전한 차'],
    '총비용': [500000, 350000, 700000, 420000, 150000],
    '클릭수': [1000, 800, 1200, 950, 300],
    '전환수': [20, 10, 30, 25, 5]
}
input_df = pd.DataFrame(sample_data)

# 사용자가 데이터를 편집할 수 있는 인터랙티브 표 (데이터 에디터)
edited_df = st.data_editor(
    input_df,
    num_rows="dynamic", # 사용자가 행을 동적으로 추가/삭제 가능
    height=300, # 표의 높이 지정
    use_container_width=True
)

# --- 분석 시작 버튼 및 로직 (수정된 부분) ---
if st.button('🚀 분석 시작하기'):
    # 입력된 데이터가 있는지 확인
    # '키워드' 열이 비어있지 않은 행만 필터링
    valid_df = edited_df[edited_df['키워드'].astype(str).str.strip() != '']
    
    if valid_df.empty:
        st.warning("분석할 데이터를 입력해주세요.")
    else:
        # 로딩 스피너와 함께 분석 함수 실행
        with st.spinner('데이터를 분석하고 있습니다. 잠시만 기다려주세요...'):
            result_data = run_keyword_analysis(valid_df)
            
            if result_data is not None and not result_data.empty:
                st.success('분석이 완료되었습니다! 🎉')
                st.write("---")
                
                st.subheader('📊 키워드 분석 결과')
                st.dataframe(result_data, height=600, use_container_width=True)
                
                # 다운로드 버튼
                csv = result_data.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="결과 다운로드 (CSV)",
                    data=csv,
                    file_name='keyword_analysis_result.csv',
                    mime='text/csv',
                )