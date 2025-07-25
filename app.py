# app.py (성과 점수 기준 조절 & ROAS 추가 & 키워드 집계 & 오류 처리 강화)

import streamlit as st
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------
# 개선된 키워드 분석 로직 (오류 처리 강화 버전)
# --------------------------------------------------------------------------
def run_keyword_analysis(df: pd.DataFrame, contrib_weight: float, rate_weight: float, cpa_weight: float):
    try:
        # --- 컬럼명 유연하게 매핑 ---
        # 가능한 컬럼명 변형들
        keyword_variants = ['키워드', 'keyword', 'Keyword']
        cost_variants = ['총비용', '비용', 'cost', 'Cost', '총 비용']
        clicks_variants = ['클릭수', '클릭', 'clicks', 'Clicks', '클릭 수']
        conversions_variants = ['전환수', '전환', 'conversions', 'Conversions', '전환 수']
        revenue_variants = ['전환당매출액', '전환당 매출액', '매출액', 'revenue', 'Revenue']
        
        # 실제 컬럼명 찾기
        def find_column(df, variants):
            for col in df.columns:
                if col.strip() in variants:
                    return col
            return None
        
        col_keyword = find_column(df, keyword_variants)
        col_cost = find_column(df, cost_variants)
        col_clicks = find_column(df, clicks_variants)
        col_conversions = find_column(df, conversions_variants)
        col_revenue_per_conv = find_column(df, revenue_variants)
        
        # 필수 컬럼 확인
        missing_cols = []
        if not col_keyword: missing_cols.append("키워드")
        if not col_cost: missing_cols.append("총비용")
        if not col_clicks: missing_cols.append("클릭수")
        if not col_conversions: missing_cols.append("전환수")
        
        if missing_cols:
            st.error(f"필수 컬럼을 찾을 수 없습니다: {', '.join(missing_cols)}")
            st.error("컬럼명이 [키워드, 총비용, 클릭수, 전환수, 전환당매출액] 형식인지 확인해주세요.")
            return None
        
        # 데이터 타입 변환 (더 안전하게)
        df[col_cost] = pd.to_numeric(df[col_cost].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df[col_clicks] = pd.to_numeric(df[col_clicks].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df[col_conversions] = pd.to_numeric(df[col_conversions].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        
        if col_revenue_per_conv:
            df[col_revenue_per_conv] = pd.to_numeric(df[col_revenue_per_conv].astype(str).str.replace(',', ''), errors='coerce')
        else:
            df[col_revenue_per_conv] = None
        
        # 키워드 정리
        df = df.dropna(subset=[col_keyword])
        df[col_keyword] = df[col_keyword].astype(str).str.strip()
        df = df[df[col_keyword] != '']

        # --- 키워드별로 데이터 집계 ---
        # 먼저 간단한 집계
        agg_dict = {
            col_cost: 'sum',
            col_clicks: 'sum',
            col_conversions: 'sum'
        }
        
        df_grouped = df.groupby(col_keyword).agg(agg_dict).reset_index()
        
        # 전환당매출액 가중평균 별도 계산
        if col_revenue_per_conv:
            revenue_dict = {}
            for keyword in df_grouped[col_keyword]:
                keyword_data = df[df[col_keyword] == keyword]
                conversions = keyword_data[col_conversions]
                revenues = keyword_data[col_revenue_per_conv]
                
                # 전환수가 있고 매출액이 있는 경우만 계산
                valid_mask = (conversions > 0) & (~revenues.isna())
                if valid_mask.any():
                    total_conversions = conversions[valid_mask].sum()
                    if total_conversions > 0:
                        weighted_sum = (conversions[valid_mask] * revenues[valid_mask]).sum()
                        revenue_dict[keyword] = weighted_sum / total_conversions
                    else:
                        revenue_dict[keyword] = None
                else:
                    revenue_dict[keyword] = None
            
            # 전환당매출액 추가
            df_grouped[col_revenue_per_conv] = df_grouped[col_keyword].map(revenue_dict)
        else:
            df_grouped[col_revenue_per_conv] = None
        
        # 집계된 데이터로 df 교체
        df = df_grouped.copy()
        
        # 인덱스 리셋
        df = df.reset_index(drop=True)

        # --- 1. 지표 계산 ---
        conv_eff_list = []
        conv_rate_list = []
        conv_contrib_list = []
        cpa_list = []
        roas_list = []
        
        total_conversions = df[col_conversions].sum()

        for idx in df.index:
            cost = df.loc[idx, col_cost]
            clicks = df.loc[idx, col_clicks]
            conversions = df.loc[idx, col_conversions]
            revenue_per_conv = df.loc[idx, col_revenue_per_conv] if col_revenue_per_conv else None

            # 기존 지표들
            conv_eff = conversions / cost if cost != 0 else 0
            conv_rate = (conversions / clicks) * 100 if clicks != 0 else 0
            conv_contrib = (conversions / total_conversions) * 100 if total_conversions != 0 else 0
            cpa = cost / conversions if conversions != 0 else 0
            
            # ROAS 계산
            if revenue_per_conv is None or pd.isna(revenue_per_conv) or revenue_per_conv == 0:
                roas = None
            else:
                revenue = conversions * revenue_per_conv
                roas = (revenue / cost) * 100 if cost != 0 else 0
            
            conv_eff_list.append(conv_eff)
            conv_rate_list.append(conv_rate)
            conv_contrib_list.append(conv_contrib)
            cpa_list.append(cpa)
            roas_list.append(roas)

        # --- 2. 정규화 및 점수 계산 (CPA는 역정규화) ---
        def normalize(lst):
            lst = [0 if pd.isna(x) else x for x in lst]
            min_val = min(lst)
            max_val = max(lst)
            range_val = max_val - min_val if max_val != min_val else 1
            return [(x - min_val) / range_val * 100 for x in lst]
        
        def reverse_normalize(lst):
            """CPA는 낮을수록 좋으므로 역정규화"""
            lst = [0 if pd.isna(x) else x for x in lst]
            min_val = min(lst)
            max_val = max(lst)
            range_val = max_val - min_val if max_val != min_val else 1
            return [(max_val - x) / range_val * 100 for x in lst]

        norm_contrib_list = normalize(conv_contrib_list)
        norm_rate_list = normalize(conv_rate_list)
        norm_cpa_list = reverse_normalize(cpa_list)

        # 가중치 정규화 (합계가 1이 되도록)
        total_weight = contrib_weight + rate_weight + cpa_weight
        if total_weight == 0:
            total_weight = 1
        
        normalized_contrib_weight = contrib_weight / total_weight
        normalized_rate_weight = rate_weight / total_weight
        normalized_cpa_weight = cpa_weight / total_weight

        # 종합 점수 계산 (사용자 정의 가중치 적용)
        total_scores = [
            round(contrib * normalized_contrib_weight + rate * normalized_rate_weight + cpa * normalized_cpa_weight, 2)
            for contrib, rate, cpa in zip(norm_contrib_list, norm_rate_list, norm_cpa_list)
        ]

        # --- 3. 전략 분류 및 최종 결과 생성 ---
        score_series = pd.Series(total_scores)
        score_rank = score_series.rank(ascending=False, method='min')
        total = len(score_series)
        result_list = []

        for idx in df.index:
            keyword = df.loc[idx, col_keyword]
            cost = df.loc[idx, col_cost]
            clicks = df.loc[idx, col_clicks]
            conversions = df.loc[idx, col_conversions]
            
            list_idx = df.index.get_loc(idx)
            cpa = cpa_list[list_idx]
            roas = roas_list[list_idx]
            conv_rate_val = conv_rate_list[list_idx]
            contrib = conv_contrib_list[list_idx]
            total_score = total_scores[list_idx]
            rank_percentile = score_rank[list_idx] / total

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
                round(roas, 2) if roas is not None else "NONE",
                total_score,
                strategy
            ])
        
        result_df = pd.DataFrame(
            result_list,
            columns=["키워드", "전환 기여도 (%)", "전환율 (%)", "CPA (원)", "ROAS", "종합 점수", "전략 추천"]
        )
        result_df = result_df.sort_values(by="종합 점수", ascending=False).reset_index(drop=True)
        
        return result_df

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
        st.error("입력 데이터의 형식을 확인해주세요.")
        st.info("필수 컬럼: 키워드, 총비용, 클릭수, 전환수 (선택: 전환당매출액)")
        return None

# --------------------------------------------------------------------------
# Streamlit 웹페이지 UI 구성
# --------------------------------------------------------------------------

# 페이지 레이아웃 넓게 설정
st.set_page_config(layout="wide", page_title="키워드 성과 분석 도구")

# 제목
st.title('🔑 소재/키워드 성과 분석')
st.write("아래 표에 데이터를 직접 붙여넣거나 입력한 후, 성과 점수 기준을 설정하고 '분석 시작하기' 버튼을 누르세요.")
st.write("---")

# --- 성과 점수 가중치 설정 UI ---
st.subheader("⚖️ 성과 점수 기준 설정")
st.write("각 지표의 중요도를 설정하세요. 비율은 자동으로 계산됩니다.")

col1, col2, col3 = st.columns(3)

with col1:
    contrib_weight = st.slider(
        "전환 기여도 가중치", 
        min_value=0.0, 
        max_value=10.0, 
        value=5.0, 
        step=0.1,
        help="전체 전환에서 해당 키워드가 차지하는 비중"
    )

with col2:
    rate_weight = st.slider(
        "전환율 가중치", 
        min_value=0.0, 
        max_value=10.0, 
        value=2.5, 
        step=0.1,
        help="클릭 대비 전환 비율"
    )

with col3:
    cpa_weight = st.slider(
        "CPA 가중치", 
        min_value=0.0, 
        max_value=10.0, 
        value=2.5, 
        step=0.1,
        help="전환당 광고비 (낮을수록 좋음)"
    )

# 가중치 비율 표시
total_weight = contrib_weight + rate_weight + cpa_weight
if total_weight > 0:
    contrib_ratio = (contrib_weight / total_weight) * 100
    rate_ratio = (rate_weight / total_weight) * 100
    cpa_ratio = (cpa_weight / total_weight) * 100
    
    st.info(f"📊 **현재 가중치 비율**: 전환 기여도 {contrib_ratio:.1f}% | 전환율 {rate_ratio:.1f}% | CPA {cpa_ratio:.1f}%")

st.write("---")

# --- 데이터 입력 UI ---
st.subheader("📋 데이터 입력")

# 데이터 입력 방법 선택
input_method = st.radio(
    "데이터 입력 방법 선택:",
    ["직접 입력/붙여넣기", "CSV 파일 업로드"]
)

if input_method == "직접 입력/붙여넣기":
    st.info("엑셀이나 구글 시트에서 [키워드, 총비용, 클릭수, 전환수, 전환당매출액] 데이터를 복사한 후, 아래 표의 첫 번째 칸을 클릭하고 붙여넣기(Ctrl+V) 하세요.")
    st.warning("💡 **참고**: 동일한 키워드가 여러 행에 있는 경우, 자동으로 합산되어 하나의 키워드로 분석됩니다.")
    
    # 사용자가 데이터를 붙여넣기 쉽도록 예시 데이터가 포함된 빈 데이터프레임 생성
    sample_data = {
        '키워드': ['볼보 XC60', '볼보 S90 가격', '전기차 보조금', '수입 SUV 추천', '안전한 차'],
        '총비용': [500000, 350000, 700000, 420000, 150000],
        '클릭수': [1000, 800, 1200, 950, 300],
        '전환수': [20, 10, 30, 25, 5],
        '전환당매출액': [100000.0, 150000.0, 80000.0, None, 120000.0]
    }
    input_df = pd.DataFrame(sample_data)
    
    # 컬럼 설정
    column_config = {
        "키워드": st.column_config.TextColumn(
            "키워드",
            help="분석할 키워드를 입력하세요",
            max_chars=50,
        ),
        "총비용": st.column_config.NumberColumn(
            "총비용",
            help="해당 키워드에 소요된 총 광고비",
            min_value=0,
            format="%d",
        ),
        "클릭수": st.column_config.NumberColumn(
            "클릭수",
            help="해당 키워드의 총 클릭 수",
            min_value=0,
            format="%d",
        ),
        "전환수": st.column_config.NumberColumn(
            "전환수",
            help="해당 키워드를 통한 총 전환 수",
            min_value=0,
            format="%d",
        ),
        "전환당매출액": st.column_config.NumberColumn(
            "전환당매출액",
            help="전환 1건당 평균 매출액 (ROAS 계산용, 없으면 빈칸)",
            min_value=0,
            format="%.0f",
        ),
    }
    
    # 사용자가 데이터를 편집할 수 있는 인터랙티브 표
    edited_df = st.data_editor(
        input_df,
        column_config=column_config,
        num_rows="dynamic",
        height=300,
        use_container_width=True
    )
    
    analysis_df = edited_df

else:  # CSV 파일 업로드
    uploaded_file = st.file_uploader(
        "CSV 파일을 선택하세요",
        type=['csv'],
        help="컬럼명이 [키워드, 총비용, 클릭수, 전환수, 전환당매출액] 형식인 CSV 파일"
    )
    
    if uploaded_file is not None:
        try:
            analysis_df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            st.success("파일이 성공적으로 업로드되었습니다!")
            st.dataframe(analysis_df.head(10))
        except Exception as e:
            st.error(f"파일 읽기 오류: {e}")
            analysis_df = None
    else:
        analysis_df = None

# --- 분석 시작 버튼 및 로직 ---
if st.button('🚀 분석 시작하기', type='primary'):
    if analysis_df is None or analysis_df.empty:
        st.warning("분석할 데이터를 입력하거나 업로드해주세요.")
    elif total_weight == 0:
        st.warning("최소 하나의 가중치는 0보다 커야 합니다.")
    else:
        # 입력된 데이터가 있는지 확인
        valid_df = analysis_df.copy()
        
        # 로딩 스피너와 함께 분석 함수 실행
        with st.spinner('데이터를 분석하고 있습니다. 잠시만 기다려주세요...'):
            result_data = run_keyword_analysis(
                valid_df, 
                contrib_weight, 
                rate_weight, 
                cpa_weight
            )
            
            if result_data is not None and not result_data.empty:
                st.success('분석이 완료되었습니다! 🎉')
                st.write("---")
                
                # 분석 설정 요약
                st.subheader('📋 분석 설정 요약')
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**전환 기여도 가중치**: {contrib_ratio:.1f}%")
                    st.write(f"**전환율 가중치**: {rate_ratio:.1f}%")
                    st.write(f"**CPA 가중치**: {cpa_ratio:.1f}%")
                with col2:
                    st.write(f"**분석 키워드 수**: {len(result_data)}개")
                    roas_available = len([x for x in result_data['ROAS'] if x != "NONE"])
                    st.write(f"**ROAS 계산 가능**: {roas_available}개 키워드")
                
                st.write("---")
                
                st.subheader('📊 키워드 분석 결과')
                st.dataframe(result_data, height=600, use_container_width=True)
                
                # 다운로드 버튼
                csv = result_data.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 결과 다운로드 (CSV)",
                    data=csv,
                    file_name='keyword_analysis_result.csv',
                    mime='text/csv',
                )
                
                # --- 간단한 인사이트 제공 ---
                st.write("---")
                st.subheader('💡 주요 인사이트')
                
                top_keyword = result_data.iloc[0]
                worst_keyword = result_data.iloc[-1]
                
                # ROAS 평균 계산 (NONE이 아닌 값들만)
                roas_values = [x for x in result_data['ROAS'] if x != "NONE"]
                avg_roas = sum(roas_values) / len(roas_values) if roas_values else 0
                
                insight_col1, insight_col2, insight_col3 = st.columns(3)
                
                with insight_col1:
                    st.metric(
                        label="최고 성과 키워드", 
                        value=top_keyword['키워드'], 
                        delta=f"점수: {top_keyword['종합 점수']}"
                    )
                
                with insight_col2:
                    if avg_roas > 0:
                        st.metric(
                            label="평균 ROAS", 
                            value=f"{avg_roas:.2f}", 
                            delta=f"{len(roas_values)}개 키워드 기준"
                        )
                    else:
                        st.metric(
                            label="평균 ROAS", 
                            value="계산 불가", 
                            delta="전환당매출액 데이터 없음"
                        )
                
                with insight_col3:
                    strong_keywords = len(result_data[result_data['종합 점수'] >= 80])
                    st.metric(
                        label="강화 대상 키워드", 
                        value=f"{strong_keywords}개", 
                        delta="80점 이상"
                    )
