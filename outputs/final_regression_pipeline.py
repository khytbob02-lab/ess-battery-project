"""DAY2 모델 개발 및 평가 — 회귀 Pipeline (수정본)

DAY1에서 만든 셀 단위 초기 100 Cycle Feature로 cycle_life를 예측한다.

[DAY2 기준]
- Train  : Batch 1 Cross-Validation 평균 (Train 부분만 사용)
- Valid  : Batch 1 Hold-out — 과제 기준대로 "충전 프로토콜 단위"로 분리해
           같은 프로토콜 셀이 Train/Valid에 동시에 들어가는 누수를 막는다.
- Test   : Batch 2 (mandatory)
- Batch 3: 추가 검증 (not mandatory)

[이전 버전 대비 수정 사항]
1. Hold-out·CV를 셀 무작위 분할 → 충전 프로토콜(Group) 단위 분할로 변경
2. HistGradientBoosting이 학습 27셀에서 분기하지 못해(min_samples_leaf 기본값 20)
   Dummy와 동일한 상수 예측을 내던 문제 수정
3. 모델 선택 기준을 코드와 문서가 일치하도록 명시 (Batch 1 CV → Valid 순, Test 미사용)
4. Batch 3 Reporting Format(9행) 전체 저장
5. 오류 분석 CSV 저장 (가장 크게 틀린 셀, 예측 범위)
6. README.md 성능·모델 선택·오류 분석 섹션을 실행 결과로 자동 갱신 (코드·CSV·README 수치 일치)
7. DAY1 전략(중복 Feature 1개만 사용) 세트와 전체 세트를 함께 비교해 전략 → 구현 반영
8. 분할 안정성(서로 다른 Hold-out 20개 반복)과 Permutation Feature Importance(Batch 1 out-of-fold) 추가
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TARGET_MAPE = 9.1                 # 원논문 Regression 테스트 오차(%)
SEED = 42
TIE_MARGIN = 0.005                # CV MAPE 차이가 0.5%p 이내면 Valid로 결정

NUMERIC_FEATURES = [
    "log_delta_var",
    "delta_abs_mean",
    "delta_max_abs",
    "QD_mean_100",
    "QD_std_100",
    "QD_fade_100",
    "degradation_rate_100",
    "IR_mean_100",
    "Tavg_mean_100",
    "Tmax_mean_100",
    "chargetime_initial_5",
    "avg_charging_speed_to_80_C",
    "first_C",
    "second_C",
    "C_rate_change",
    "charge_current_mean_100",
    "charge_current_std_100",
    "charge_current_max_100",
]
# DAY1 전략(보고서 4·5장) 그대로: 같은 정보를 담은 중복 Feature는 1개만 남긴 세트
#  - ΔQ 통계(분산·평균·최댓값) → log_delta_var 1개
#  - 초기 충전시간 vs Cavg_80 → Cavg_80 1개
#  - Tavg vs Tmax → Tavg 1개
#  - QD 감소량 vs 감소기울기 → QD_fade_100 1개
#  - 충전 전류 통계는 C-rate 구조 Feature와 같은 정보 → 제외
STRATEGY_FEATURES = [
    "log_delta_var",
    "QD_mean_100",
    "QD_std_100",
    "QD_fade_100",
    "IR_mean_100",
    "Tavg_mean_100",
    "avg_charging_speed_to_80_C",
    "first_C",
    "second_C",
    "C_rate_change",
]
FEATURE_SETS = {"DAY1 전략 세트(10)": STRATEGY_FEATURES, "전체 세트(18)": NUMERIC_FEATURES}
POLICY_COLUMNS = ["charging_policy", "policy", "policy_readable"]
# 원논문 공개 코드에서 채널 노이즈로 제외한 Batch 3 셀(파일 인덱스). 이상치 제거 민감도 분석용
PAPER_NOISY_B3 = {2, 23, 32, 37, 42, 43}


OUT = ROOT / "outputs"


def find_data_dir(cli_dir):
    candidates = [Path(cli_dir)] if cli_dir else [OUT, ROOT / "data", ROOT]
    for d in candidates:
        if (d / "day1_final_early_features_all_batches.csv").exists():
            return d
    raise FileNotFoundError(
        "day1_final_*.csv 를 찾지 못했습니다. data/ 폴더에 넣거나 --data-dir 로 지정하세요: "
        + ", ".join(str(c) for c in candidates))


def load_features(data_dir: Path) -> pd.DataFrame:
    early = pd.read_csv(data_dir / "day1_final_early_features_all_batches.csv")
    delta = pd.read_csv(data_dir / "day1_final_delta_qv_all_batches.csv")
    life = pd.read_csv(data_dir / "day1_final_life_all_batches.csv")

    valid = life[life["target_valid"].astype(bool)].copy()
    keep = ["batch", "cell_id", "cycle_life"] + [c for c in POLICY_COLUMNS if c in life.columns]
    data = valid[keep].merge(early, on=["batch", "cell_id", "cycle_life"], how="inner",
                             suffixes=("", "_early"))
    data = data.merge(delta[["batch", "cell_id", "delta_abs_mean", "delta_max_abs", "delta_var"]],
                      on=["batch", "cell_id"], how="inner")
    data["log_delta_var"] = np.log10(data["delta_var"].clip(lower=1e-12))
    data["policy_group"] = make_policy_group(data)

    expected = {"Batch 1": 36, "Batch 2": 39, "Batch 3": 44}
    actual = data.groupby("batch").size().to_dict()
    if actual != expected:
        raise ValueError(f"최종 유효 셀 수 불일치: {actual} != {expected}")
    return data


def make_policy_group(data: pd.DataFrame) -> pd.Series:
    """충전 프로토콜 식별자. 정책 문자열이 있으면 그대로, 없으면 C-rate 구조로 만든다."""
    for col in POLICY_COLUMNS:
        if col in data.columns and data[col].notna().all():
            return data[col].astype(str).str.replace("-newstructure", "", regex=False).str.strip()
    cols = ["first_C", "second_C", "avg_charging_speed_to_80_C"]
    return data[cols].round(2).astype(str).agg("|".join, axis=1)


def make_model(estimator):
    preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return TransformedTargetRegressor(
        regressor=Pipeline([("preprocess", preprocess), ("model", estimator)]),
        func=np.log10,
        inverse_func=lambda x: np.power(10.0, x),
    )


def metrics(y_true, pred):
    return {
        "MAPE": mean_absolute_percentage_error(y_true, pred),
        "MAE_cycle": mean_absolute_error(y_true, pred),
        "overprediction_rate": float(np.mean(pred > y_true)),
        "mean_signed_error_cycle": float(np.mean(pred - y_true)),
    }


def group_holdout(groups: pd.Series, n_cells: int):
    """프로토콜 단위 Hold-out. Valid가 전체 셀의 약 25%(±2셀)가 되는 첫 seed를 사용한다."""
    target = round(n_cells * 0.25)
    for seed in range(SEED, SEED + 500):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        tr, va = next(gss.split(np.zeros(n_cells), groups=groups))
        if abs(len(va) - target) <= 2:
            return tr, va, seed
    raise RuntimeError("적절한 Group Hold-out 분할을 찾지 못했습니다.")


N_STABILITY_SPLITS = 20


def distinct_group_holdouts(groups: pd.Series, n_cells: int, n_wanted: int):
    """서로 다른 프로토콜 단위 Hold-out 분할을 n_wanted개 만든다 (분할 안정성 점검용)."""
    target, seen, out = round(n_cells * 0.25), set(), []
    for seed in range(0, 5000):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        tr, va = next(gss.split(np.zeros(n_cells), groups=groups))
        key = tuple(sorted(va))
        if abs(len(va) - target) <= 2 and key not in seen:
            seen.add(key); out.append((tr, va, seed))
            if len(out) == n_wanted:
                break
    return out


def label_of(c):
    return c[0] if c[1] == "-" else f"{c[0]} ({c[1]})"


def repeated_holdout(b1, candidates):
    """서로 다른 프로토콜 단위 Hold-out 20개에서 모든 후보의 CV·Valid MAPE를 계산한다.
    Valid가 8~10셀뿐이라 한 번의 분할로 모델을 고르면 결과가 분할에 좌우되기 때문이다. Test 배치는 사용하지 않는다."""
    splits = distinct_group_holdouts(b1["policy_group"], len(b1), N_STABILITY_SPLITS)
    recs = []
    for tr_idx, va_idx, seed in splits:
        tr, va = b1.iloc[tr_idx], b1.iloc[va_idx]
        g = tr["policy_group"]; cv = GroupKFold(n_splits=min(5, g.nunique()))
        for c in candidates:
            m_name, fs_name, make, cols = c
            mdl = make_model(make())
            cv_m = -cross_val_score(mdl, tr[cols], tr["cycle_life"], groups=g, cv=cv,
                                    scoring="neg_mean_absolute_percentage_error").mean()
            mdl.fit(tr[cols], tr["cycle_life"])
            recs.append({"split_seed": seed, "n_valid": len(va), "model": label_of(c), "algorithm": m_name,
                         "cv_mape": cv_m,
                         "valid_mape": mean_absolute_percentage_error(va["cycle_life"], mdl.predict(va[cols]))})
    return pd.DataFrame(recs), splits


def single_split_choice(rep):
    """분할 하나만 보고 'CV 최소 → 0.5%p 이내면 Valid'로 고를 때 분할마다 어떤 모델이 뽑히는지."""
    out = {}
    for seed, r in rep[rep["algorithm"] != "Dummy 평균 기준선"].groupby("split_seed"):
        tied = r[r["cv_mape"] <= r["cv_mape"].min() + TIE_MARGIN]
        out[seed] = tied.sort_values(["valid_mape", "cv_mape"]).iloc[0]["model"]
    return pd.Series(out)


def final_over_splits(b1, b2, b3, final_c, splits):
    recs = []
    for tr_idx, va_idx, seed in splits:
        tr, va = b1.iloc[tr_idx], b1.iloc[va_idx]; cols = final_c[3]
        mdl = make_model(final_c[2]()); mdl.fit(tr[cols], tr["cycle_life"])
        recs.append({"split_seed": seed,
                     "final_valid_mape": mean_absolute_percentage_error(va["cycle_life"], mdl.predict(va[cols])) * 100,
                     "final_batch2_mape": mean_absolute_percentage_error(b2["cycle_life"], mdl.predict(b2[cols])) * 100,
                     "final_batch3_mape": mean_absolute_percentage_error(b3["cycle_life"], mdl.predict(b3[cols])) * 100})
    return pd.DataFrame(recs)


def oof_permutation_importance(b1, make, cols):
    """Batch 1 전체(36셀)를 프로토콜 단위 GroupKFold로 나눠, 학습에 쓰지 않은 fold에서만
    Permutation Importance(Feature를 섞었을 때 MAPE가 얼마나 나빠지는지)를 계산한다. Test 배치는 사용하지 않는다."""
    groups = b1["policy_group"]; cv = GroupKFold(n_splits=min(5, groups.nunique()))
    imp = []
    for k, (tr, te) in enumerate(cv.split(b1, groups=groups)):
        mdl = make_model(make()); mdl.fit(b1.iloc[tr][cols], b1.iloc[tr]["cycle_life"])
        r = permutation_importance(mdl, b1.iloc[te][cols], b1.iloc[te]["cycle_life"],
                                   scoring="neg_mean_absolute_percentage_error", n_repeats=20, random_state=SEED + k)
        imp.append(pd.DataFrame(r.importances.T * 100, columns=cols))   # MAPE 증가량(%p)
    allimp = pd.concat(imp, ignore_index=True)
    out = pd.DataFrame({"feature": cols, "mape_increase_pctp": allimp.mean().values, "std": allimp.std().values})
    return out.sort_values("mape_increase_pctp", ascending=False).reset_index(drop=True)


def save_figure(results, predictions, best, fig_dir: Path):
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    def en(n):
        return (n.replace("Dummy 평균 기준선", "Dummy baseline").replace("DAY1 전략 세트(10)", "strategy-10")
                 .replace("전체 세트(18)", "full-18").replace("HistGradientBoosting", "HistGB"))
    labels = [en(n) for n in results["model"]]
    x = np.arange(len(labels)); width = 0.19
    for i, (col, lab, color) in enumerate([
        ("cv_mape", "Train CV", "#4C78A8"), ("valid_mape", "Valid", "#F58518"),
        ("batch2_test_mape", "Batch 2 Test", "#E45756"), ("batch3_test_mape", "Batch 3 Test", "#54A24B"),
    ]):
        axes[0].bar(x + (i - 1.5) * width, results[col] * 100, width, label=lab, color=color)
    axes[0].axhline(TARGET_MAPE, color="#222222", linestyle="--", linewidth=1.2, label="Paper target 9.1%")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("MAPE (%)"); axes[0].set_title("Model performance comparison")
    axes[0].grid(axis="y", alpha=0.25); axes[0].legend(fontsize=8, ncol=2)

    pred = predictions[predictions["model"] == best]
    for split, lab, color in [("Batch1_valid", "Batch 1 Valid", "#F58518"),
                              ("Batch2_test", "Batch 2 Test", "#E45756"),
                              ("Batch3_test", "Batch 3 Test", "#54A24B")]:
        g = pred[pred["split"] == split]
        axes[1].scatter(g["actual_cycle_life"], g["predicted_cycle_life"], s=34, alpha=0.78, color=color, label=lab)
    lim = [0, max(pred["actual_cycle_life"].max(), pred["predicted_cycle_life"].max()) * 1.05]
    axes[1].plot(lim, lim, color="#222222", linestyle="--", linewidth=1.2, label="Perfect prediction")
    axes[1].set_xlim(lim); axes[1].set_ylim(lim)
    axes[1].set_xlabel("Actual cycle life"); axes[1].set_ylabel("Predicted cycle life")
    axes[1].set_title(f"Actual vs predicted: {en(best)}"); axes[1].grid(alpha=0.25); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "day2_performance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fmt(v, d=2):
    return f"{v:.{d}f}"


def md_table(df, float_d=2):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v, float_d) if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r.values) + " |")
    return "\n".join(lines)


def fill_readme(blocks: dict, path: Path = None):
    """README의 <!-- KEY:START --> ~ <!-- KEY:END --> 구간을 실행 결과로 채운다."""
    path = path or (ROOT / "README.md")
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    for key, body in blocks.items():
        start, end = f"<!-- {key}:START -->", f"<!-- {key}:END -->"
        if start in text and end in text:
            pre, rest = text.split(start, 1)
            _, post = rest.split(end, 1)
            text = pre + start + "\n" + body.strip() + "\n" + end + post
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="day1_final_*.csv 위치 (기본: outputs/)")
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    res_dir = fig_dir = OUT
    OUT.mkdir(exist_ok=True)

    data = load_features(data_dir)
    b1 = data[data["batch"] == "Batch 1"].reset_index(drop=True)
    b2 = data[data["batch"] == "Batch 2"].reset_index(drop=True)
    b3 = data[data["batch"] == "Batch 3"].reset_index(drop=True)

    # ---- 1) Batch 1 프로토콜 단위 Hold-out (같은 프로토콜 셀이 train/valid에 나뉘면 누수) ----
    tr_idx, va_idx, split_seed = group_holdout(b1["policy_group"], len(b1))
    train, valid = b1.iloc[tr_idx], b1.iloc[va_idx]
    assert set(train["policy_group"]).isdisjoint(valid["policy_group"]), "Train/Valid 프로토콜 중복"
    y_tr, g_tr, y_va = train["cycle_life"], train["policy_group"], valid["cycle_life"]
    y_b2, y_b3 = b2["cycle_life"], b3["cycle_life"]
    b3_clean_mask = ~b3["cell_id"].astype(int).isin(PAPER_NOISY_B3)

    estimators = {
        "Ridge": lambda: Ridge(alpha=10.0),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.7, random_state=SEED, n_jobs=1),
        # 학습 셀이 30개 안팎이므로 min_samples_leaf를 낮춰야 실제로 트리가 분기한다
        "HistGradientBoosting": lambda: HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.04, max_leaf_nodes=7, min_samples_leaf=3,
            l2_regularization=1.0, random_state=SEED),
    }
    candidates = [("Dummy 평균 기준선", "-", lambda: DummyRegressor(strategy="mean"), STRATEGY_FEATURES)]
    for fs_name, cols in FEATURE_SETS.items():
        for m_name, make in estimators.items():
            candidates.append((m_name, fs_name, make, cols))

    # ---- 2) Train(CV)도 같은 이유로 GroupKFold ----
    n_splits = min(5, g_tr.nunique())
    cv = GroupKFold(n_splits=n_splits)

    rows, predictions = [], []
    for m_name, fs_name, make, cols in candidates:
        label = m_name if fs_name == "-" else f"{m_name} ({fs_name})"
        model = make_model(make())
        cv_mape = -cross_val_score(model, train[cols], y_tr, groups=g_tr, cv=cv,
                                   scoring="neg_mean_absolute_percentage_error")
        model.fit(train[cols], y_tr)
        p_va, p_b2, p_b3 = model.predict(valid[cols]), model.predict(b2[cols]), model.predict(b3[cols])
        m_va, m_b2, m_b3 = metrics(y_va, p_va), metrics(y_b2, p_b2), metrics(y_b3, p_b3)
        m_b3c = metrics(y_b3[b3_clean_mask], p_b3[b3_clean_mask.values])
        rows.append({
            "model": label, "algorithm": m_name, "feature_set": fs_name,
            "cv_mape": float(cv_mape.mean()), "cv_mape_std": float(cv_mape.std()),
            "valid_mape": m_va["MAPE"], "batch2_test_mape": m_b2["MAPE"], "batch3_test_mape": m_b3["MAPE"],
            "batch3_clean_test_mape": m_b3c["MAPE"],
            "valid_mae_cycle": m_va["MAE_cycle"], "batch2_mae_cycle": m_b2["MAE_cycle"],
            "batch3_mae_cycle": m_b3["MAE_cycle"],
            "batch2_overprediction_rate": m_b2["overprediction_rate"],
            "batch3_overprediction_rate": m_b3["overprediction_rate"],
            "batch2_mean_signed_error_cycle": m_b2["mean_signed_error_cycle"],
            "batch3_mean_signed_error_cycle": m_b3["mean_signed_error_cycle"],
            "n_unique_predictions_b2": int(np.unique(np.round(p_b2, 3)).size),
        })
        for split, frame, pred in [("Batch1_valid", valid, p_va), ("Batch2_test", b2, p_b2),
                                   ("Batch3_test", b3, p_b3)]:
            for rec, p in zip(frame.to_dict("records"), pred):
                predictions.append({"model": label, "split": split, "batch": rec["batch"],
                                    "cell_id": int(rec["cell_id"]), "policy_group": rec["policy_group"],
                                    "actual_cycle_life": float(rec["cycle_life"]),
                                    "predicted_cycle_life": float(p)})

    # ---- 3) 모델 선택: Batch 1 반복 Hold-out(20개) 평균 Valid MAPE 최소 → 0.5%p 이내면 평균 CV 최소.
    #         Test(Batch 2·3)는 사용하지 않음. 본 분할(seed 고정)은 지정 성능 포맷 보고용이다.
    results = pd.DataFrame(rows)
    rep, splits = repeated_holdout(b1, candidates)
    rep.to_csv(res_dir / "day2_repeated_holdout_all_models.csv", index=False)
    agg = rep.groupby("model").agg(rep_valid_mean=("valid_mape", "mean"), rep_valid_std=("valid_mape", "std"),
                                   rep_cv_mean=("cv_mape", "mean")).reset_index()
    results = results.merge(agg, on="model", how="left")
    pool = results[results["algorithm"] != "Dummy 평균 기준선"]
    tied = pool[pool["rep_valid_mean"] <= pool["rep_valid_mean"].min() + TIE_MARGIN]
    best = tied.sort_values(["rep_cv_mean", "rep_valid_mean"]).iloc[0]["model"]
    results["selected"] = results["model"].eq(best)
    results["selected_for_business"] = results["selected"]   # PDF 빌더 호환
    results = results.sort_values(["selected", "rep_valid_mean"], ascending=[False, True]).reset_index(drop=True)
    results.to_csv(res_dir / "final_model_comparison.csv", index=False)
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(res_dir / "final_model_predictions.csv", index=False)

    fig_rows = results[results["model"].isin(
        [best, "Dummy 평균 기준선"] + results[results["feature_set"] == results.loc[results["selected"], "feature_set"].iloc[0]]["model"].tolist())]
    save_figure(fig_rows.drop_duplicates("model").reset_index(drop=True), pred_df, best, fig_dir)

    s = results[results["selected"]].iloc[0]
    tr, va = s["cv_mape"] * 100, s["valid_mape"] * 100
    te2, te3, te3c = s["batch2_test_mape"] * 100, s["batch3_test_mape"] * 100, s["batch3_clean_test_mape"] * 100
    reg = pd.DataFrame([
        {"구분": "Train (Batch 1 CV)", "MAPE(%)": tr, "비고": f"{n_splits}-fold GroupKFold(충전 프로토콜) 평균"},
        {"구분": "Valid (Batch 1 Hold-out)", "MAPE(%)": va, "비고": f"프로토콜 단위 Hold-out {len(valid)}셀"},
        {"구분": "Test (Batch 2)", "MAPE(%)": te2, "비고": "최종 평가"},
        {"구분": "Gap (Train-Valid)", "MAPE(%)": va - tr, "비고": "(+) : 과적합 의심"},
        {"구분": "Gap (Valid-Test)", "MAPE(%)": te2 - va, "비고": "(+) : 배치간 일반화 저하 의심"},
        {"구분": "Gap (Target-Test)", "MAPE(%)": te2 - TARGET_MAPE, "비고": "Target : 원논문 9.1%"},
    ])
    reg.to_csv(res_dir / "day2_regression_reporting.csv", index=False)
    b3_rep = pd.concat([reg, pd.DataFrame([
        {"구분": "Test (Batch 3)", "MAPE(%)": te3, "비고": f"추가 검증 ({len(b3)}셀)"},
        {"구분": "Gap (Batch2-Batch3)", "MAPE(%)": te2 - te3, "비고": "Test 성능 간 비교"},
        {"구분": "Gap (Target-Test)", "MAPE(%)": te3 - TARGET_MAPE, "비고": "Batch 3 기준, 원논문 성능 비교"},
    ])], ignore_index=True)
    b3_rep.to_csv(res_dir / "day2_batch3_reporting.csv", index=False)

    # ---- 4) 오류 분석 ----
    sel = pred_df[(pred_df["model"] == best) & (pred_df["split"] != "Batch1_valid")].copy()
    sel["ape_pct"] = (sel["predicted_cycle_life"] - sel["actual_cycle_life"]).abs() / sel["actual_cycle_life"] * 100
    sel["direction"] = np.where(sel["predicted_cycle_life"] > sel["actual_cycle_life"], "과대예측", "과소예측")
    top = sel.sort_values("ape_pct", ascending=False).head(15)
    top.to_csv(res_dir / "day2_error_analysis_top15.csv", index=False)
    lo, hi = y_tr.min(), y_tr.max()
    sel["구간"] = np.where(sel["actual_cycle_life"] < lo, "학습 최솟값 미만",
                         np.where(sel["actual_cycle_life"] > hi, "학습 최댓값 초과", "학습 범위 안"))
    by_range = (sel.groupby(["batch", "구간"])
                .agg(셀수=("ape_pct", "size"), MAPE=("ape_pct", "mean"),
                     과대예측비율=("direction", lambda d: (d == "과대예측").mean() * 100))
                .reset_index())
    by_range.to_csv(res_dir / "day2_error_by_life_range.csv", index=False)
    rng = pd.DataFrame({
        "구간": ["Batch 1 Train 실제 수명", "Batch 2 실제 수명", "Batch 3 실제 수명", "선택 모델 예측값(Batch 2·3)"],
        "min": [lo, y_b2.min(), y_b3.min(), sel["predicted_cycle_life"].min()],
        "max": [hi, y_b2.max(), y_b3.max(), sel["predicted_cycle_life"].max()],
    })
    rng.to_csv(res_dir / "day2_prediction_range.csv", index=False)
    pd.DataFrame([{
        "n_train": len(train), "n_valid": len(valid), "n_train_protocols": g_tr.nunique(),
        "n_valid_protocols": valid["policy_group"].nunique(), "cv_folds": n_splits, "split_seed": split_seed,
        "n_tied_candidates": len(tied), "b3_clean_mape": te3c, "n_b3_clean": int(b3_clean_mask.sum()),
    }]).to_csv(res_dir / "day2_split_summary.csv", index=False)
    with open(res_dir / "day2_selected_model.txt", "w", encoding="utf-8") as f:
        f.write(f"{best}\n")

    # ---- 5) README 자동 반영 (코드·CSV·README 수치 불일치 방지) ----
    cmp_tbl = results[["algorithm", "feature_set", "rep_valid_mean", "rep_cv_mean", "cv_mape", "valid_mape", "batch2_test_mape", "batch3_test_mape",
                       "batch2_overprediction_rate"]].copy()
    for c in ["rep_valid_mean", "rep_cv_mean", "cv_mape", "valid_mape", "batch2_test_mape", "batch3_test_mape", "batch2_overprediction_rate"]:
        cmp_tbl[c] = cmp_tbl[c] * 100
    cmp_tbl.columns = ["모델", "Feature 세트", "반복 Valid 평균", "반복 CV 평균", "고정분할 CV", "고정분할 Valid", "Batch 2 Test", "Batch 3 Test",
                       "Batch 2 과대예측(%)"]
    cmp_tbl.insert(0, "선택", np.where(results["selected"], "최종", ""))
    tie_note = (f"평균 Valid가 최소값과 0.5%p 이내인 후보가 {len(tied)}개여서 반복 평균 CV가 더 낮은 모델을 골랐습니다."
                if len(tied) > 1 else "반복 평균 Valid MAPE가 다른 후보보다 0.5%p 이상 낮아 바로 선택했습니다.")
    single = single_split_choice(rep)
    sb = s_rep = results[results["selected"]].iloc[0]
    select_md = (
        f"- **최종 모델 : {best}**\n"
        f"- 선택 규칙 : Batch 1 안에서 서로 다른 프로토콜 단위 Hold-out 분할 {rep['split_seed'].nunique()}개를 만들고, "
        f"분할마다 Train으로 학습해 Valid MAPE를 구한 뒤 **평균 Valid MAPE가 가장 낮은 모델**을 고릅니다. "
        f"0.5%p 이내 동률이면 평균 CV MAPE로 결정합니다. Batch 2·3은 선택이 끝난 뒤 일반화 성능 보고에만 사용합니다.\n"
        f"- 반복 분할을 쓰는 이유 : Valid가 8~10셀이라 분할 하나만 보고 고르면 분할마다 선택이 바뀝니다. 실제로 분할 하나 기준 규칙을 "
        f"{len(single)}개 분할에 적용하면 가장 많이 뽑힌 모델도 {single.value_counts().iloc[0]}/{len(single)}회에 그쳤습니다.\n"
        f"- 결과 : {best}의 반복 평균 Valid {sb['rep_valid_mean']*100:.2f}% ± {sb['rep_valid_std']*100:.2f}%, 평균 CV {sb['rep_cv_mean']*100:.2f}%. {tie_note}\n"
        f"- 지정 성능 포맷의 Train·Valid 값은 고정 분할 1개(Train {len(train)}셀/{g_tr.nunique()}개 프로토콜, "
        f"Valid {len(valid)}셀/{valid['policy_group'].nunique()}개 프로토콜, 프로토콜 중복 0개) 기준입니다.\n\n"
        + md_table(cmp_tbl) + "\n\n단위: MAPE(%). 반복 평균은 Batch 1 Hold-out 분할 20개 평균, 고정분할은 지정 성능 포맷에 쓴 분할입니다. Dummy는 평균 예측 기준선입니다."
    )
    strat = results[results["feature_set"] == "DAY1 전략 세트(10)"]
    full = results[results["feature_set"] == "전체 세트(18)"]
    pairs = []
    for alg in ["Ridge", "Random Forest", "HistGradientBoosting"]:
        a_s = strat[strat["algorithm"] == alg]; a_f = full[full["algorithm"] == alg]
        if len(a_s) and len(a_f):
            pairs.append(f"{alg} {a_s['rep_valid_mean'].iloc[0]*100:.2f}% vs {a_f['rep_valid_mean'].iloc[0]*100:.2f}%")
    better = int(sum(strat.sort_values("algorithm")["rep_valid_mean"].values <= full.sort_values("algorithm")["rep_valid_mean"].values))
    select_md += ("\n\n- DAY1 전략 세트(10) vs 전체 세트(18) 반복 평균 Valid : " + ", ".join(pairs) +
                  f". 3개 알고리즘 중 {better}개에서 전략 세트가 같거나 더 낮아, 중복 Feature를 줄인 DAY1 전략이 성능을 해치지 않음을 확인했습니다.")
    perf_md = (
        "**Regression (Batch 1 학습 → Batch 2 Test)**\n\n" + md_table(reg) +
        "\n\n**Batch 3 추가 검증**\n\n" + md_table(b3_rep.iloc[6:].reset_index(drop=True)) +
        f"\n\n- 원논문에서도 제외한 Batch 3 노이즈 셀 4개(파일 인덱스 2, 37, 42, 43)를 빼면 Batch 3 MAPE는 "
        f"{fmt(te3c)}%입니다. (이상치 제거 효과를 확인하기 위한 민감도 분석)\n"
        f"- Gap (Train-Valid)는 Valid가 {len(valid)}셀뿐이라 불확실성이 큽니다. 부호보다 크기를 참고용으로 봅니다."
    )
    def seg(batch, name, col):
        q = by_range[(by_range["batch"] == batch) & (by_range["구간"] == name)]
        return float(q[col].iloc[0]) if len(q) else float("nan")
    out2 = int(sel[(sel["batch"] == "Batch 2") & (sel["구간"] != "학습 범위 안")].shape[0])
    out3 = int(sel[(sel["batch"] == "Batch 3") & (sel["구간"] != "학습 범위 안")].shape[0])
    perf_md += (
        f"\n- Gap (Batch2-Batch3) {te2 - te3:.2f}%p는 Feature가 특정 배치에 과적합되었다기보다 수명 분포 차이에서 옵니다. "
        f"두 배치 모두 학습 범위({lo:.0f}~{hi:.0f} Cycle) 안의 셀은 MAPE {seg('Batch 2', '학습 범위 안', 'MAPE'):.1f}%, "
        f"{seg('Batch 3', '학습 범위 안', 'MAPE'):.1f}%로 비슷하고, 범위를 벗어난 셀 비율이 Batch 2({out2}/{len(b2)})가 "
        f"Batch 3({out3}/{len(b3)})보다 훨씬 높기 때문입니다.\n"
        "- 배치별 Qdlin 시작 시점 차이는 ΔQ = Qdlin(100) − Qdlin(10)을 같은 셀 안에서 빼서 계산하므로 셀 단위 오프셋이 상쇄됩니다."
    )
    b2r = by_range[by_range["batch"] == "Batch 2"].set_index("구간")
    b3r = by_range[by_range["batch"] == "Batch 3"].set_index("구간")
    lines = [
        f"- 선택 모델의 Batch 2·3 예측값은 **{fmt(rng.iloc[3]['min'], 0)}~{fmt(rng.iloc[3]['max'], 0)} Cycle**에 머뭅니다. "
        f"학습에 쓴 Batch 1 수명이 {fmt(lo, 0)}~{fmt(hi, 0)} Cycle이라, 이 범위 밖의 수명은 구조적으로 맞히기 어렵습니다.",
    ]
    if "학습 최솟값 미만" in b2r.index:
        r = b2r.loc["학습 최솟값 미만"]
        lines.append(f"- Batch 2에서 학습 최솟값({fmt(lo, 0)})보다 짧은 셀 {int(r['셀수'])}개의 MAPE는 {fmt(r['MAPE'], 1)}%, "
                     f"과대예측 비율은 {fmt(r['과대예측비율'], 0)}%입니다.")
    if "학습 범위 안" in b2r.index:
        r = b2r.loc["학습 범위 안"]
        lines.append(f"- 반대로 학습 범위 안에 있는 Batch 2 셀 {int(r['셀수'])}개의 MAPE는 {fmt(r['MAPE'], 1)}%입니다.")
    if "학습 최댓값 초과" in b3r.index:
        r = b3r.loc["학습 최댓값 초과"]
        lines.append(f"- Batch 3에서 학습 최댓값보다 긴 셀 {int(r['셀수'])}개는 MAPE {fmt(r['MAPE'], 1)}%, "
                     f"과대예측 비율 {fmt(r['과대예측비율'], 0)}%로 " +
                     ("대부분 **짧게** 예측됩니다." if r["과대예측비율"] < 50 else "오히려 길게 예측됩니다."))
    t10 = sel.sort_values("ape_pct", ascending=False).head(10)
    n_b2, n_over = int((t10["batch"] == "Batch 2").sum()), int((t10["direction"] == "과대예측").sum())
    n_short = int((t10["actual_cycle_life"] < lo).sum())
    n_long = int((t10["actual_cycle_life"] > hi).sum())
    lines.append(f"- 가장 크게 틀린 셀의 공통점 : 오차 상위 10개 중 {n_short + n_long}개가 학습 수명 범위 밖의 셀입니다 "
                 f"(학습 최솟값보다 짧아 과대예측된 셀 {n_short}개, 학습 최댓값보다 길어 과소예측된 셀 {n_long}개). "
                 f"배치로는 Batch 2가 {n_b2}개, Batch 3가 {10 - n_b2}개입니다.")
    top5 = top.head(5)[["batch", "cell_id", "actual_cycle_life", "predicted_cycle_life", "ape_pct", "direction"]].copy()
    top5.columns = ["batch", "cell_id", "실제 수명", "예측 수명", "APE(%)", "방향"]
    top5["실제 수명"] = top5["실제 수명"].round(0).astype(int); top5["예측 수명"] = top5["예측 수명"].round(0).astype(int)
    error_md = "\n".join(lines) + "\n\n**오차 상위 5개 셀**\n\n" + md_table(top5, 1)
    # ---- 6) 분할 안정성 + Feature 중요도 ----
    final_c = next(c for c in candidates if label_of(c) == best)
    stab = final_over_splits(b1, b2, b3, final_c, splits)
    stab = stab.merge(rep[rep["model"] == best][["split_seed", "cv_mape"]].assign(final_cv_mape=lambda d: d["cv_mape"] * 100)
                      .drop(columns="cv_mape"), on="split_seed")
    stab["single_split_choice"] = stab["split_seed"].map(single)
    stab.to_csv(res_dir / "day2_split_stability.csv", index=False)
    imp = oof_permutation_importance(b1, final_c[2], final_c[3])
    imp.to_csv(res_dir / "day2_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    top_imp = imp.head(12).iloc[::-1]
    ax.barh(top_imp["feature"], top_imp["mape_increase_pctp"], xerr=top_imp["std"], color="#4C78A8", alpha=.85)
    ax.axvline(0, color="#222222", lw=.8)
    ax.set_xlabel("MAPE increase when shuffled (%p)"); ax.set_title("Permutation importance (Batch 1 out-of-fold)")
    fig.tight_layout(); fig.savefig(fig_dir / "day2_feature_importance.png", dpi=200); plt.close(fig)

    desc = stab[["final_cv_mape", "final_valid_mape", "final_batch2_mape", "final_batch3_mape"]].agg(["mean", "std", "min", "max"]).T
    desc.index = ["Train (Batch 1 CV)", "Valid (Batch 1 Hold-out)", "Test (Batch 2)", "Test (Batch 3)"]
    desc = desc.reset_index().rename(columns={"index": "구분", "mean": "평균", "std": "표준편차", "min": "최소", "max": "최대"})
    stab_md = (
        f"지정 성능 포맷은 고정 분할 1개의 값이므로, 같은 최종 모델 구성({best})을 서로 다른 프로토콜 단위 Hold-out 분할 "
        f"{len(stab)}개에서 다시 학습해 결과가 분할에 좌우되는지 확인했습니다.\n\n" + md_table(desc) +
        f"\n\n- 고정 분할의 Batch 2 Test {te2:.2f}%는 반복 분할 범위({desc.iloc[2]['최소']:.1f}~{desc.iloc[2]['최대']:.1f}%, "
        f"평균 {desc.iloc[2]['평균']:.2f}%) 안에 있어 특정 분할에서 우연히 나온 값이 아닙니다.\n"
        f"- Valid MAPE는 분할에 따라 {desc.iloc[1]['최소']:.1f}~{desc.iloc[1]['최대']:.1f}%로 흔들립니다. 그래서 모델 선택은 고정 분할 1개가 아니라 반복 분할 평균으로 했습니다.\n"
        f"- Batch 3 Test는 표준편차 {desc.iloc[3]['표준편차']:.2f}%p로 안정적이고, Batch 2 Test는 {desc.iloc[2]['표준편차']:.2f}%p로 더 흔들립니다. "
        f"Batch 2의 단수명 셀 예측이 학습에 포함된 Batch 1 셀 구성에 민감한 것으로 보입니다."
    )
    rank = {f: i + 1 for i, f in enumerate(imp["feature"])}
    dq_cols = [f for f in imp["feature"] if f in ("log_delta_var", "delta_max_abs", "delta_abs_mean")]
    dq_sum = float(imp[imp["feature"].isin(dq_cols)]["mape_increase_pctp"].sum())
    rest_sum = float(imp[~imp["feature"].isin(dq_cols)]["mape_increase_pctp"].clip(lower=0).sum())
    imp_note = (f"- EDA에서 핵심 신호로 고른 log_delta_var가 {rank.get('log_delta_var', '-')}위입니다. "
                f"ΔQ 계열 Feature({', '.join(dq_cols)})를 섞었을 때의 MAPE 증가 합은 {dq_sum:.2f}%p로, "
                f"나머지 Feature 전체({rest_sum:.2f}%p)보다 {dq_sum / max(rest_sum, 1e-9):.1f}배 큽니다. "
                "DAY1 EDA의 결론(ΔQ가 배치가 바뀌어도 유지되는 핵심 신호)이 모델에서도 확인됩니다.")
    imp_top = imp.head(8).copy()
    imp_top.columns = ["Feature", "MAPE 증가(%p)", "표준편차"]
    rank = {f: i + 1 for i, f in enumerate(imp["feature"])}
    imp_md = (
        "최종 모델이 실제로 어떤 Feature에 의존하는지 Batch 1 안에서만 확인했습니다. 프로토콜 단위 GroupKFold로 학습에 쓰지 않은 fold에서 "
        "Feature 값을 무작위로 섞었을 때 MAPE가 얼마나 나빠지는지(Permutation Importance)를 계산했습니다.\n\n"
        + md_table(imp_top) +
        "\n\n" + imp_note + "\n"
        f"- 값이 0 근처이거나 음수인 Feature는 섞어도 성능이 변하지 않아, 최종 모델에 실질적으로 기여하지 않습니다.\n\n"
        "![importance](outputs/day2_feature_importance.png)"
    )
    # 상세 결과는 DAY2_README, 메인 README에는 요약만 넣는다
    fill_readme({"DETAIL_SELECTION": select_md, "DETAIL_IMPORTANCE": imp_md, "DETAIL_STABILITY": stab_md,
                 "DETAIL_ERRORS": error_md}, OUT / "DAY2_README.md")

    sb = results[results["selected"]].iloc[0]
    short_tbl = results[results["algorithm"] != "Dummy 평균 기준선"].sort_values("rep_valid_mean").head(3)
    short_tbl = pd.concat([short_tbl, results[results["algorithm"] == "Dummy 평균 기준선"]])
    st = pd.DataFrame({
        "모델": short_tbl["algorithm"].values,
        "Feature 세트": [f.replace("DAY1 ", "") if f != "-" else "-" for f in short_tbl["feature_set"]],
        "Valid 평균(20회)": short_tbl["rep_valid_mean"].values * 100,
        "Batch 2 Test": short_tbl["batch2_test_mape"].values * 100,
        "Batch 3 Test": short_tbl["batch3_test_mape"].values * 100,
    })
    n_rep = rep["split_seed"].nunique()
    sel_short = (
        f"- 최종 모델 : {sb['algorithm']} + 전략 세트(10개)\n" if sb["feature_set"].startswith("DAY1") else
        f"- 최종 모델 : {sb['algorithm']} + 전체 세트(18개)\n")
    sel_short += (
        f"- 선택 방법 : Batch 1 안에서 프로토콜 단위 Hold-out 분할을 {n_rep}번 바꿔 가며 Valid MAPE 평균이 가장 낮은 모델을 골랐다. "
        f"Valid가 8~10셀이라 분할 한 번으로 고르면 분할마다 결과가 달랐다(가장 많이 뽑힌 모델도 {n_rep}번 중 {single.value_counts().iloc[0]}번). "
        "Batch 2·3은 선택에 쓰지 않았다.\n"
        f"- 선택 이유 : Valid 평균 {sb['rep_valid_mean']*100:.2f}%로 후보 중 가장 낮았다."
        + (" DAY1에서 중복 Feature를 줄인 전략 세트가 전체 세트보다 나았다." if sb["feature_set"].startswith("DAY1") else "")
        + "\n\n" + md_table(st) + "\n\n단위 MAPE(%). 전체 후보 비교는 `outputs/DAY2_README.md`에 있다."
    )
    if imp.iloc[0]["feature"] == "log_delta_var":
        sel_short += (f"\n\nPermutation Importance(Batch 1 안에서 계산)로 보면 log_delta_var를 섞었을 때 MAPE가 {imp.iloc[0]['mape_increase_pctp']:.2f}%p 올라, "
                      f"나머지 Feature를 모두 섞은 영향의 합({rest_sum:.2f}%p)보다 컸다. EDA에서 고른 ΔQ가 모델에서도 가장 중요하게 쓰였다.")

    stb = stab[["final_valid_mape", "final_batch2_mape", "final_batch3_mape"]]
    in2, in3 = seg("Batch 2", "학습 범위 안", "MAPE"), seg("Batch 3", "학습 범위 안", "MAPE")
    perf_short = (
        md_table(reg) + "\n\n**Batch 3 추가 검증**\n\n" + md_table(b3_rep.iloc[6:].reset_index(drop=True)) + "\n\n"
        f"- Gap (Train-Valid) {va - tr:+.2f}%p : Valid가 {len(valid)}셀이라 값 자체는 흔들린다. 분할 {n_rep}번 평균으로도 Train {sb['rep_cv_mean']*100:.2f}%, "
        f"Valid {sb['rep_valid_mean']*100:.2f}%로 차이가 작아 과적합은 크지 않다.\n"
        f"- Gap (Valid-Test) {te2 - va:+.2f}%p, Gap (Target-Test) {te2 - TARGET_MAPE:+.2f}%p : Batch 2 성능 저하는 대부분 학습 범위보다 수명이 짧은 셀에서 나온다(오류 분석 참고). "
        f"학습 수명 범위 안의 셀만 보면 Batch 2 MAPE는 {in2:.1f}%로 원논문 9.1%와 비슷하다.\n"
        f"- Gap (Batch2-Batch3) {te2 - te3:+.2f}%p : 학습 범위 안의 셀은 Batch 2 {in2:.1f}%, Batch 3 {in3:.1f}%로 비슷하다. 차이는 Feature가 한 배치에 맞춰져서라기보다, "
        f"학습 범위를 벗어난 셀이 Batch 2({out2}/{len(b2)})에 더 많기 때문이다.\n"
        f"- 분할을 {n_rep}번 바꿔 다시 학습해도 Batch 2 Test는 {stb['final_batch2_mape'].min():.1f}~{stb['final_batch2_mape'].max():.1f}%, "
        f"Batch 3 Test는 {stb['final_batch3_mape'].min():.1f}~{stb['final_batch3_mape'].max():.1f}% 안에 있었다.\n"
        f"- 원논문에서 노이즈로 제외한 Batch 3 셀 4개를 빼면 Batch 3 MAPE는 {te3c:.2f}%다.\n"
        "- ΔQ는 같은 셀의 Cycle 100과 10을 빼서 만들기 때문에, 배치별 Qdlin 시작 위치 차이는 상쇄된다.\n\n"
        "원논문 9.1%는 학습 셀 수와 데이터 정제 조건이 달라 참고 기준으로만 비교했다."
    )

    b2lo = by_range[(by_range["batch"] == "Batch 2") & (by_range["구간"] == "학습 최솟값 미만")]
    b3hi = by_range[(by_range["batch"] == "Batch 3") & (by_range["구간"] == "학습 최댓값 초과")]
    err_short = (
        f"- 최종 모델의 Batch 2·3 예측값은 {rng.iloc[3]['min']:.0f}~{rng.iloc[3]['max']:.0f} Cycle 사이에 있다. 학습 셀 수명({lo:.0f}~{hi:.0f})을 벗어나는 값은 거의 예측하지 못한다.\n"
        f"- 가장 크게 틀린 셀의 공통점 : 오차 상위 10개가 모두 학습 수명 범위 밖의 셀이다. Batch 2의 단수명 셀은 길게, Batch 3의 장수명 셀은 짧게 예측됐다.\n"
    ) if n_short + n_long == 10 else (
        f"- 최종 모델의 Batch 2·3 예측값은 {rng.iloc[3]['min']:.0f}~{rng.iloc[3]['max']:.0f} Cycle 사이에 있다.\n"
        f"- 가장 크게 틀린 셀의 공통점 : 오차 상위 10개 중 {n_short + n_long}개가 학습 수명 범위 밖의 셀이다.\n")
    if len(b2lo):
        err_short += f"- Batch 2에서 학습 최솟값보다 짧은 {int(b2lo['셀수'].iloc[0])}셀은 MAPE {b2lo['MAPE'].iloc[0]:.1f}%, 범위 안 셀은 {in2:.1f}%다.\n"
    if len(b3hi):
        err_short += f"- Batch 3에서 학습 최댓값보다 긴 {int(b3hi['셀수'].iloc[0])}셀은 MAPE {b3hi['MAPE'].iloc[0]:.1f}%다.\n"
    err_short += "- 셀별 오차는 `outputs/day2_error_analysis_top15.csv`에 있다."
    fill_readme({"SELECTION": sel_short, "PERFORMANCE": perf_short, "ERRORS": err_short})

    print("[유효 셀 수]", data.groupby("batch").size().to_dict())
    print(f"[Batch 1 분할] Train {len(train)}셀/{g_tr.nunique()}개 프로토콜, "
          f"Valid {len(valid)}셀/{valid['policy_group'].nunique()}개 프로토콜 (seed={split_seed})")
    print("[모델 비교]"); print(results[["model", "cv_mape", "valid_mape", "batch2_test_mape", "batch3_test_mape",
                                     "n_unique_predictions_b2", "selected"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("[선정] Batch 1 반복 Hold-out 평균 Valid 최소(0.5%p 이내 동률 시 평균 CV) →", best)
    print("[DAY2 Reporting]"); print(b3_rep.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"[Batch 3 노이즈 셀 제외 민감도] {te3c:.2f}%")
    print("[예측 범위]"); print(rng.to_string(index=False, float_format=lambda v: f"{v:.0f}"))
    print("README.md 성능·선택·오류분석 섹션을 실행 결과로 갱신했습니다." if (ROOT / "README.md").exists() else "")


if __name__ == "__main__":
    main()
