# DAY2 상세 결과

요약은 저장소 최상위 `README.md`에 있다. 이 문서에는 README에서 줄인 표와 수치를 모았다. 아래 수치는 `python outputs/regression_pipeline.py` 실행 시 갱신된다.

## 실행
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python outputs/regression_pipeline.py          # 학습·평가, 결과 저장, README 갱신
python outputs/DS-MINI-Day1-report-builder.py  # 보고서 PDF 재생성 (NanumGothic 폰트 필요)
```

## 데이터 분할과 모델 선택
- Batch 1을 충전 프로토콜 단위로 Train(약 75%)과 Valid(약 25%)로 나눈다. Train 안에서는 프로토콜 단위 5-fold GroupKFold CV를 한다.
- 성능 포맷의 Train·Valid 값은 고정 분할 1개 기준이다.
- 모델 선택은 서로 다른 Hold-out 분할 20개의 Valid MAPE 평균이 가장 낮은 모델로 한다. 0.5%p 이내로 비슷하면 CV 평균으로 정한다. Batch 2·3은 선택에 쓰지 않는다.
- HistGradientBoosting은 학습 셀이 30개 안팎이라 `min_samples_leaf=3`으로 둔다. 기본값 20에서는 분기가 생기지 않는다.

## 전체 후보 비교
<!-- DETAIL_SELECTION:START -->
- **최종 모델 : HistGradientBoosting (DAY1 전략 세트(10))**
- 선택 규칙 : Batch 1 안에서 서로 다른 프로토콜 단위 Hold-out 분할 20개를 만들고, 분할마다 Train으로 학습해 Valid MAPE를 구한 뒤 **평균 Valid MAPE가 가장 낮은 모델**을 고릅니다. 0.5%p 이내 동률이면 평균 CV MAPE로 결정합니다. Batch 2·3은 선택이 끝난 뒤 일반화 성능 보고에만 사용합니다.
- 반복 분할을 쓰는 이유 : Valid가 8~10셀이라 분할 하나만 보고 고르면 분할마다 선택이 바뀝니다. 실제로 분할 하나 기준 규칙을 20개 분할에 적용하면 가장 많이 뽑힌 모델도 8/20회에 그쳤습니다.
- 결과 : HistGradientBoosting (DAY1 전략 세트(10))의 반복 평균 Valid 9.63% ± 2.71%, 평균 CV 10.58%. 반복 평균 Valid MAPE가 다른 후보보다 0.5%p 이상 낮아 바로 선택했습니다.
- 지정 성능 포맷의 Train·Valid 값은 고정 분할 1개(Train 28셀/15개 프로토콜, Valid 8셀/5개 프로토콜, 프로토콜 중복 0개) 기준입니다.

| 선택 | 모델 | Feature 세트 | 반복 Valid 평균 | 반복 CV 평균 | 고정분할 CV | 고정분할 Valid | Batch 2 Test | Batch 3 Test | Batch 2 과대예측(%) |
|---|---|---|---|---|---|---|---|---|---|
| 최종 | HistGradientBoosting | DAY1 전략 세트(10) | 9.63 | 10.58 | 10.51 | 8.12 | 28.96 | 20.95 | 79.49 |
|  | Random Forest | DAY1 전략 세트(10) | 10.71 | 11.09 | 10.09 | 11.13 | 38.00 | 19.91 | 82.05 |
|  | HistGradientBoosting | 전체 세트(18) | 10.75 | 11.52 | 8.94 | 13.22 | 38.12 | 19.56 | 82.05 |
|  | Random Forest | 전체 세트(18) | 11.06 | 11.17 | 9.28 | 12.39 | 41.95 | 19.37 | 87.18 |
|  | Ridge | 전체 세트(18) | 11.84 | 10.78 | 11.15 | 12.19 | 60.27 | 23.41 | 87.18 |
|  | Ridge | DAY1 전략 세트(10) | 12.66 | 11.61 | 12.23 | 12.69 | 61.47 | 23.44 | 87.18 |
|  | Dummy 평균 기준선 | - | 16.93 | 17.12 | 14.31 | 22.53 | 56.70 | 25.62 | 76.92 |

단위: MAPE(%). 반복 평균은 Batch 1 Hold-out 분할 20개 평균, 고정분할은 지정 성능 포맷에 쓴 분할입니다. Dummy는 평균 예측 기준선입니다.

- DAY1 전략 세트(10) vs 전체 세트(18) 반복 평균 Valid : Ridge 12.66% vs 11.84%, Random Forest 10.71% vs 11.06%, HistGradientBoosting 9.63% vs 10.75%. 3개 알고리즘 중 2개에서 전략 세트가 같거나 더 낮아, 중복 Feature를 줄인 DAY1 전략이 성능을 해치지 않음을 확인했습니다.
<!-- DETAIL_SELECTION:END -->

## Feature 중요도
<!-- DETAIL_IMPORTANCE:START -->
최종 모델이 실제로 어떤 Feature에 의존하는지 Batch 1 안에서만 확인했습니다. 프로토콜 단위 GroupKFold로 학습에 쓰지 않은 fold에서 Feature 값을 무작위로 섞었을 때 MAPE가 얼마나 나빠지는지(Permutation Importance)를 계산했습니다.

| Feature | MAPE 증가(%p) | 표준편차 |
|---|---|---|
| log_delta_var | 8.19 | 4.15 |
| QD_std_100 | 0.78 | 0.66 |
| avg_charging_speed_to_80_C | 0.54 | 0.98 |
| QD_mean_100 | 0.48 | 1.01 |
| second_C | 0.31 | 0.74 |
| QD_fade_100 | 0.28 | 0.59 |
| Tavg_mean_100 | 0.10 | 0.27 |
| C_rate_change | -0.01 | 0.08 |

- EDA에서 핵심 신호로 고른 log_delta_var가 1위입니다. ΔQ 계열 Feature(log_delta_var)를 섞었을 때의 MAPE 증가 합은 8.19%p로, 나머지 Feature 전체(2.48%p)보다 3.3배 큽니다. DAY1 EDA의 결론(ΔQ가 배치가 바뀌어도 유지되는 핵심 신호)이 모델에서도 확인됩니다.
- 값이 0 근처이거나 음수인 Feature는 섞어도 성능이 변하지 않아, 최종 모델에 실질적으로 기여하지 않습니다.

![importance](outputs/day2_feature_importance.png)
<!-- DETAIL_IMPORTANCE:END -->

## 분할 안정성
<!-- DETAIL_STABILITY:START -->
지정 성능 포맷은 고정 분할 1개의 값이므로, 같은 최종 모델 구성(HistGradientBoosting (DAY1 전략 세트(10)))을 서로 다른 프로토콜 단위 Hold-out 분할 20개에서 다시 학습해 결과가 분할에 좌우되는지 확인했습니다.

| 구분 | 평균 | 표준편차 | 최소 | 최대 |
|---|---|---|---|---|
| Train (Batch 1 CV) | 10.58 | 1.68 | 7.31 | 12.90 |
| Valid (Batch 1 Hold-out) | 9.63 | 2.71 | 5.53 | 14.34 |
| Test (Batch 2) | 31.02 | 5.89 | 22.19 | 43.14 |
| Test (Batch 3) | 19.06 | 1.03 | 17.74 | 21.78 |

- 고정 분할의 Batch 2 Test 28.96%는 반복 분할 범위(22.2~43.1%, 평균 31.02%) 안에 있어 특정 분할에서 우연히 나온 값이 아닙니다.
- Valid MAPE는 분할에 따라 5.5~14.3%로 흔들립니다. 그래서 모델 선택은 고정 분할 1개가 아니라 반복 분할 평균으로 했습니다.
- Batch 3 Test는 표준편차 1.03%p로 안정적이고, Batch 2 Test는 5.89%p로 더 흔들립니다. Batch 2의 단수명 셀 예측이 학습에 포함된 Batch 1 셀 구성에 민감한 것으로 보입니다.
<!-- DETAIL_STABILITY:END -->

## 오류 분석 상세
<!-- DETAIL_ERRORS:START -->
- 선택 모델의 Batch 2·3 예측값은 **547~880 Cycle**에 머뭅니다. 학습에 쓴 Batch 1 수명이 534~1054 Cycle이라, 이 범위 밖의 수명은 구조적으로 맞히기 어렵습니다.
- Batch 2에서 학습 최솟값(534)보다 짧은 셀 30개의 MAPE는 33.1%, 과대예측 비율은 100%입니다.
- 반대로 학습 범위 안에 있는 Batch 2 셀 7개의 MAPE는 8.9%입니다.
- Batch 3에서 학습 최댓값보다 긴 셀 17개는 MAPE 39.3%, 과대예측 비율 0%로 대부분 **짧게** 예측됩니다.
- 가장 크게 틀린 셀의 공통점 : 오차 상위 10개 중 10개가 학습 수명 범위 밖의 셀입니다 (학습 최솟값보다 짧아 과대예측된 셀 4개, 학습 최댓값보다 길어 과소예측된 셀 6개). 배치로는 Batch 2가 4개, Batch 3가 6개입니다.

**오차 상위 5개 셀**

| batch | cell_id | 실제 수명 | 예측 수명 | APE(%) | 방향 |
|---|---|---|---|---|---|
| Batch 2 | 6 | 393 | 669 | 70.2 | 과대예측 |
| Batch 3 | 38 | 1935 | 808 | 58.3 | 과소예측 |
| Batch 3 | 7 | 1836 | 809 | 55.9 | 과소예측 |
| Batch 3 | 45 | 1801 | 831 | 53.8 | 과소예측 |
| Batch 3 | 16 | 1638 | 785 | 52.1 | 과소예측 |
<!-- DETAIL_ERRORS:END -->

## 결과 파일 (outputs/)
| 파일 | 내용 |
|---|---|
| day2_regression_reporting.csv | Regression 성능 포맷 (6행) |
| day2_batch3_reporting.csv | Batch 3 포함 성능 포맷 (9행) |
| final_model_comparison.csv | 후보 모델 × Feature 세트별 성능 |
| final_model_predictions.csv | 셀별 실제·예측 수명 |
| day2_repeated_holdout_all_models.csv | 분할 20개 × 후보 모델별 CV·Valid (모델 선택 근거) |
| day2_split_stability.csv | 최종 모델의 분할별 Valid·Batch 2·Batch 3 MAPE |
| day2_feature_importance.csv / .png | Permutation Importance (Batch 1 out-of-fold) |
| day2_error_analysis_top15.csv | 오차 상위 15개 셀 |
| day2_error_by_life_range.csv | 학습 수명 범위 안·밖 오차 |
| day2_prediction_range.csv | 학습 수명 범위와 예측값 범위 |
| day2_split_summary.csv | 분할 셀 수·프로토콜 수, Batch 3 노이즈 셀 제외 결과 |
| day2_selected_model.txt, day2_performance.png | 최종 모델명, 성능 그래프 |
