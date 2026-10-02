from pathlib import Path
import re
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
FILES = {
    "Batch 1": DATA / "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": DATA / "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": DATA / "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
COLORS = {"Batch 1": "#1677B8", "Batch 2": "#E67E22", "Batch 3": "#2E9D64"}

def scalar(h5, ref):
    d = h5[ref][()]
    return float(np.asarray(d).reshape(-1)[0]) if np.asarray(d).size else np.nan

def text_value(h5, ref):
    d = np.asarray(h5[ref][()]).reshape(-1)
    return "".join(chr(int(x)) for x in d if int(x) > 0)

def first_c(policy):
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)C", policy)
    return float(m.group(1)) if m else np.nan

def policy_rates(policy):
    vals = [float(x) for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)C", policy)]
    return (vals[0] if vals else np.nan, vals[1] if len(vals) > 1 else np.nan)

def avg_speed_to_80(policy):
    """충전 시간의 역수로 SOC 80%까지의 평균 C-rate를 계산한다."""
    segments = re.findall(r"([0-9]+(?:\.[0-9]+)?)C(?:\(([0-9]+)%\))?", policy)
    if not segments:
        return np.nan
    first_rate, first_pct = float(segments[0][0]), segments[0][1]
    if first_pct:
        first_pct = min(float(first_pct), 80.0) / 100.0
        second_rate = float(segments[1][0]) if len(segments) > 1 else first_rate
        charge_hours = first_pct / first_rate + (0.8 - first_pct) / second_rate
        return 0.8 / charge_hours
    return first_rate

life_rows, qd_rows, early_rows, delta_rows, qv_rows, excluded_rows, ir_missing_rows = [], [], [], [], [], [], []
for batch_name, path in FILES.items():
    with h5py.File(path, "r") as h5:
        b = h5["batch"]
        n = b["summary"].shape[0]
        for cell_id in range(n):
            life = scalar(h5, b["cycle_life"][cell_id, 0])
            if not np.isfinite(life) or life <= 0: life = np.nan
            policy = text_value(h5, b["policy_readable"][cell_id, 0])
            first_rate, second_rate = policy_rates(policy)
            avg_speed_80 = avg_speed_to_80(policy)
            cycles = h5[b["cycles"][cell_id, 0]]
            summary = h5[b["summary"][cell_id, 0]]
            cycle = np.asarray(summary["cycle"][()]).reshape(-1).astype(float)
            qd = np.asarray(summary["QDischarge"][()]).reshape(-1).astype(float)
            valid = np.isfinite(cycle) & np.isfinite(qd) & (qd > 0)
            cycle, qd = cycle[valid], qd[valid]
            order = np.argsort(cycle)
            cycle, qd = cycle[order], qd[order]
            # 과제 기준: 0.88 Ah EOL에 도달하기 전에 기록이 끝난 셀은
            # 실제 cycle_life를 알 수 없으므로 회귀·EDA에서 제외한다.
            # 마지막 관측 방전용량이 EOL 기준(약 0.88 Ah)에 충분히
            # 도달했는지 확인한다. 0.90 Ah 초과로 기록이 끝난 셀은
            # 과제 기준의 "수명 미확정" 10셀(Batch 1)로 분류한다.
            reached_eol = qd.size > 0 and qd[-1] <= 0.90
            target_valid = np.isfinite(life) and life >= 100 and reached_eol
            life_rows.append({"batch": batch_name, "cell_id": cell_id, "cycle_life": life, "policy": policy, "first_C": first_rate, "second_C": second_rate, "avg_charging_speed_to_80_C": avg_speed_80, "target_valid": target_valid, "reached_eol_0.88Ah": reached_eol})
            if not target_valid:
                reason = "수명 미확정: 0.88 Ah 도달 전 기록 종료" if np.isfinite(life) and not reached_eol else "cycle_life 결측 또는 100 Cycle 미만"
                excluded_rows.append({"batch": batch_name, "cell_id": cell_id, "cycle_life": life, "reason": reason})
                continue
            for c, q in zip(cycle, qd): qd_rows.append({"batch": batch_name, "cell_id": cell_id, "cycle": c, "QD": q, "cycle_life": life})
            early = cycle <= 100
            if early.sum() >= 3:
                ecycle, eqd = cycle[early], qd[early]
                ir_values = np.asarray(summary["IR"][()]).reshape(-1).astype(float)
                ir_values[ir_values <= 0] = np.nan
                ir_early = ir_values[valid][early]
                ir_missing = not np.isfinite(ir_early).any()
                if ir_missing:
                    ir_missing_rows.append({"batch": batch_name, "cell_id": cell_id, "cycle_life": life, "reason": "IR=0 sentinel 또는 유효 저항 기록 없음"})
                current_values = []
                for ci in range(min(100, cycles["I"].shape[0])):
                    current = np.asarray(h5[cycles["I"][ci, 0]][()]).reshape(-1).astype(float)
                    positive = current[np.isfinite(current) & (current > 0)]
                    if positive.size:
                        current_values.append(positive)
                current_values = np.concatenate(current_values) if current_values else np.array([np.nan])
                qd_slope = np.polyfit(ecycle, eqd, 1)[0]
                early_rows.append({"batch": batch_name, "cell_id": cell_id, "cycle_life": life,
                    "QD_mean_100": np.mean(eqd), "QD_std_100": np.std(eqd, ddof=1),
                    "QD_slope_100": qd_slope, "QD_fade_100": eqd[0] - eqd[-1], "degradation_rate_100": -qd_slope,
                    "first_C": first_rate, "second_C": second_rate, "C_rate_change": second_rate-first_rate,
                    "avg_charging_speed_to_80_C": avg_speed_80,
                    "charge_current_mean_100": np.nanmean(current_values),
                    "charge_current_std_100": np.nanstd(current_values),
                    "charge_current_max_100": np.nanmax(current_values),
                    "policy": policy,
                    "IR_mean_100": np.nanmean(ir_early) if not ir_missing else np.nan,
                    "IR_missing": ir_missing,
                    "Tavg_mean_100": np.nanmean(np.asarray(summary["Tavg"][()]).reshape(-1)[valid][early]),
                    "Tmax_mean_100": np.nanmean(np.asarray(summary["Tmax"][()]).reshape(-1)[valid][early]),
                    "chargetime_mean_100": np.nanmean(np.asarray(summary["chargetime"][()]).reshape(-1)[valid][early]),
                    "chargetime_initial_5": np.nanmedian(np.asarray(summary["chargetime"][()]).reshape(-1)[valid][early][:5])})
            if cycles["Qdlin"].shape[0] >= 100:
                q10 = np.asarray(h5[cycles["Qdlin"][9, 0]][()]).reshape(-1).astype(float)
                q100 = np.asarray(h5[cycles["Qdlin"][99, 0]][()]).reshape(-1).astype(float)
                m = np.isfinite(q10) & np.isfinite(q100)
                if m.sum():
                    d = q100[m] - q10[m]
                    delta_rows.append({"batch": batch_name, "cell_id": cell_id, "cycle_life": life,
                        "delta_mean": np.mean(d), "delta_abs_mean": np.mean(np.abs(d)), "delta_max_abs": np.max(np.abs(d)),
                        "delta_var": np.var(d)})
                    # 데이터 설명에 명시된 Qdlin 공통 전압축: 1,000 points, 2V~3.6V
                    vgrid = np.linspace(2.0, 3.6, len(q10))
                    for voltage, a10, a100 in zip(vgrid, q10, q100):
                        qv_rows.append({"batch": batch_name, "cell_id": cell_id, "cycle_life": life,
                                        "voltage": voltage, "Qdlin_C10": a10, "Qdlin_C100": a100,
                                        "delta_QV": a100 - a10})

life = pd.DataFrame(life_rows)
qd = pd.DataFrame(qd_rows)
early = pd.DataFrame(early_rows)
delta = pd.DataFrame(delta_rows)
# 과제의 데이터 품질 기준에 따른 최종 셀 수를 코드에서 검증한다.
expected_valid_counts = {"Batch 1": 36, "Batch 2": 39, "Batch 3": 44}
actual_valid_counts = life[life["target_valid"]].groupby("batch").size().to_dict()
if actual_valid_counts != expected_valid_counts:
    raise ValueError(f"유효 셀 수 불일치: 실제={actual_valid_counts}, 기준={expected_valid_counts}")
life.to_csv(OUT / "day1_final_life_all_batches.csv", index=False)
qd.to_csv(OUT / "day1_final_qd_cycle_level_all_batches.csv", index=False)
early.to_csv(OUT / "day1_final_early_features_all_batches.csv", index=False)
delta.to_csv(OUT / "day1_final_delta_qv_all_batches.csv", index=False)
pd.DataFrame(qv_rows).to_csv(OUT / "day1_final_qv_curves_all_batches.csv", index=False)
pd.DataFrame(excluded_rows).to_csv(OUT / "day1_final_excluded_cells.csv", index=False)
pd.DataFrame(ir_missing_rows).to_csv(OUT / "day1_final_ir_missing_cells.csv", index=False)

# Q1: target distribution + long/short ratio
v = life[life["target_valid"]].dropna(subset=["cycle_life"]).copy()
q1 = v.groupby("batch")["cycle_life"].agg(["count", "mean", "median", "std", "min", "max"]).round(2)
q1["long_gt_1000"] = v.groupby("batch")["cycle_life"].apply(lambda x: int((x > 1000).sum()))
 # 참고 보고서의 배치 분포 비교 기준과 맞춰 550 Cycle 미만을 단수명
 # 위험군으로 표시한다. 500 Cycle 미만은 원자료의 별도 요구 기준이므로
 # 혼동하지 않도록 결과 파일에 두 기준을 모두 저장한다.
q1["short_lt_550"] = v.groupby("batch")["cycle_life"].apply(lambda x: int((x < 550).sum()))
q1["short_lt_500"] = v.groupby("batch")["cycle_life"].apply(lambda x: int((x < 500).sum()))
q1["long_ratio"] = (q1["long_gt_1000"] / q1["count"]).round(3)
q1["short_550_ratio"] = (q1["short_lt_550"] / q1["count"]).round(3)
q1["short_ratio"] = (q1["short_lt_500"] / q1["count"]).round(3)
q1.to_csv(OUT / "day1_final_q1_summary.csv")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for b, g in v.groupby("batch"): ax[0].hist(g.cycle_life, bins=np.arange(150,2351,50), alpha=.55, label=b, color=COLORS[b], edgecolor="white")
ax[0].axvline(500, color="#C0392B", ls=":", label="과제 기준 단수명 < 500")
ax[0].axvline(550, color="#C0392B", ls="--", label="참고 기준 단수명 < 550")
ax[0].axvline(1000, color="#8E44AD", ls="--", label="장수명 > 1,000")
ax[0].set(xlim=(150,2300), xlabel="Cycle Life", ylabel="셀 개수", title="Batch별 Cycle Life 분포"); ax[0].legend(fontsize=8)
ax[1].boxplot([v[v.batch==b].cycle_life for b in ["Batch 1","Batch 2","Batch 3"]], tick_labels=["Batch 1","Batch 2","Batch 3"], patch_artist=True, boxprops=dict(facecolor="#DDEFF5")); ax[1].set(title="Batch별 분포 비교", ylabel="Cycle Life")
fig.tight_layout(); fig.savefig(OUT / "day1_final_q1.png", dpi=220); plt.close(fig)

# Q2: individual QD curves and per-cell Knee point
def cell_knee(g):
    g = g[(g["QD"] >= 0.7) & (g["QD"] <= 1.2)].sort_values("cycle")
    if len(g) < 30:
        return np.nan
    x = g["cycle"].to_numpy(dtype=float)
    y = g["QD"].to_numpy(dtype=float)
    y_smooth = pd.Series(y).rolling(15, center=True, min_periods=1).median().to_numpy()
    x0, y0, x1, y1 = x[0], y_smooth[0], x[-1], y_smooth[-1]
    distance = np.abs((x1-x0)*(y0-y_smooth) - (x0-x)*(y1-y0)) / np.hypot(x1-x0, y1-y0)
    usable = np.arange(len(x))[5:-5]
    return float(x[usable[np.argmax(distance[usable])]]) if len(usable) else np.nan

knee_cell = qd.groupby(["batch", "cell_id"]).apply(cell_knee, include_groups=False).reset_index(name="knee_cycle")
knee_cell = knee_cell.merge(v[["batch", "cell_id", "cycle_life"]], on=["batch", "cell_id"], how="left")
knee_rows = []
for b, g in knee_cell.groupby("batch"):
    knee_rows.append({"batch":b, "valid_cells":int(g["knee_cycle"].notna().sum()), "knee_cycle_median":float(g["knee_cycle"].median()), "knee_life_ratio_median":float((g["knee_cycle"]/g["cycle_life"]).median()), "knee_before_100_count":int((g["knee_cycle"]<100).sum())})
pd.DataFrame(knee_rows).to_csv(OUT / "day1_final_q2_knee_summary.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for a, b in zip(axes, ["Batch 1", "Batch 2", "Batch 3"]):
    g = qd[(qd.batch == b) & (qd["QD"] >= 0.7) & (qd["QD"] <= 1.2)]
    norm, cmap = plt.Normalize(g["cycle_life"].min(), g["cycle_life"].max()), plt.cm.viridis_r
    for (batch, cell_id), cell in g.groupby(["batch", "cell_id"]):
        a.plot(cell["cycle"], cell["QD"], color=cmap(norm(cell["cycle_life"].iloc[0])), lw=.65, alpha=.55)
    a.axvline(100, color="#C0392B", ls=":", lw=1.2)
    a.axhline(0.88, color="#666666", ls="--", lw=1.1)
    a.set_title(b); a.set_xlabel("Cycle"); a.grid(alpha=.2)
axes[0].set_ylabel("방전용량 QD (Ah)")
fig.suptitle("배치별 사이클에 따른 방전용량 (색 = Cycle Life)", fontsize=14, fontweight="bold")
fig.text(0.5, 0.01, "빨간 점선: 100번째 Cycle | 회색 점선: EOL 0.88 Ah | 유효 Target 셀만 표시", ha="center", fontsize=9)
fig.tight_layout(rect=[0, 0.03, 1, 0.96]); fig.savefig(OUT/"day1_final_q2.png", dpi=220); plt.close(fig)

# Q3: Delta Q(V) curve, log variance and log life
d=delta.dropna(subset=["cycle_life"]).copy()
d["log_delta_var"] = np.log10(d["delta_var"].clip(lower=1e-12))
d["log_cycle_life"] = np.log10(d["cycle_life"])
q3_rows=[]
for b,g in d.groupby("batch"):
    q3_rows.append({"batch":b,"n":len(g),"corr_delta_var_cycle_life":g["delta_var"].corr(g["cycle_life"]),"corr_log_delta_var_log_cycle_life":g["log_delta_var"].corr(g["log_cycle_life"])})
q3_rows.append({"batch":"전체","n":len(d),"corr_delta_var_cycle_life":d["delta_var"].corr(d["cycle_life"]),"corr_log_delta_var_log_cycle_life":d["log_delta_var"].corr(d["log_cycle_life"])})
pd.DataFrame(q3_rows).round(4).to_csv(OUT/"day1_final_q3_summary.csv",index=False)
qv = pd.DataFrame(qv_rows)
qv = qv.merge(d[["batch", "cell_id", "cycle_life", "log_delta_var"]], on=["batch", "cell_id", "cycle_life"], how="left")
fig = plt.figure(figsize=(14, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
for j, b in enumerate(["Batch 1", "Batch 2", "Batch 3"]):
    ax = fig.add_subplot(gs[0, j])
    g = qv[qv.batch == b]
    norm = plt.Normalize(g["cycle_life"].min(), g["cycle_life"].max())
    cmap = plt.cm.viridis_r
    for cell_id, cell in g.groupby("cell_id"):
        ax.plot(cell["voltage"], cell["delta_QV"], color=cmap(norm(cell["cycle_life"].iloc[0])), lw=.8, alpha=.55)
    ax.axhline(0, color="black", lw=.8)
    ax.set_title(b); ax.set_xlabel("Voltage (V)"); ax.set_ylabel("델타Q(V) (Ah)"); ax.grid(alpha=.2)
ax = fig.add_subplot(gs[1, :])
for b, g in d.groupby("batch"):
    ax.scatter(g["log_delta_var"], g["log_cycle_life"], label=b, color=COLORS[b], s=30, alpha=.75)
valid = d[["log_delta_var", "log_cycle_life"]].dropna()
if len(valid) > 1:
    coef = np.polyfit(valid["log_delta_var"], valid["log_cycle_life"], 1)
    xx = np.linspace(valid["log_delta_var"].min(), valid["log_delta_var"].max(), 200)
    ax.plot(xx, np.polyval(coef, xx), color="black", ls=":", lw=2.2, label="전체 Batch 선형 적합")
ax.set_xlabel("log10 델타Q(V) 분산"); ax.set_ylabel("log10 Cycle Life"); ax.set_title("log 델타Q(V) 분산과 log Cycle Life: 세 Batch 전체 회귀선"); ax.legend(); ax.grid(alpha=.2)
fig.suptitle("Batch별 델타Q(V) 곡선과 변화 분산·수명 관계", fontsize=15, fontweight="bold")
fig.savefig(OUT/"day1_final_q3.png",dpi=220, bbox_inches="tight"); plt.close(fig)

# Q4: 충전 C-rate·프로토콜·전류 패턴과 열화속도
v2=v.dropna(subset=["first_C"]).copy(); v2["C-rate 구간"]=pd.cut(v2.first_C,bins=[-np.inf,4.8,6,np.inf],labels=["낮음 ≤4.8C","중간 4.8~6C","높음 >6C"])
q4=v2.groupby("C-rate 구간",observed=True).cycle_life.agg(["mean","std","count"]).round(2); q4.to_csv(OUT/"day1_final_q4_summary.csv")
# Batch 1 그래프와 동일한 집계 기준을 유지한다.
batch1_v2 = v2[v2["batch"] == "Batch 1"].copy()
protocol = batch1_v2.groupby("policy").agg(mean_cycle_life=("cycle_life","mean"), std_cycle_life=("cycle_life","std"), count=("cycle_life","count"), first_C=("first_C","first"), second_C=("second_C","first"), avg_charging_speed_to_80_C=("avg_charging_speed_to_80_C","first")).sort_values("mean_cycle_life", ascending=False).round(2)
protocol.to_csv(OUT/"day1_final_q4_protocol_summary.csv")
f4 = early.dropna(subset=["degradation_rate_100", "first_C", "avg_charging_speed_to_80_C"]).copy()
current_cols = ["first_C", "second_C", "C_rate_change", "charge_current_mean_100", "charge_current_std_100", "charge_current_max_100"]
current_corr = f4[current_cols + ["degradation_rate_100"]].corr()["degradation_rate_100"].drop("degradation_rate_100").sort_values(key=lambda x: x.abs(), ascending=False).to_frame("corr_with_degradation_rate")
current_corr.to_csv(OUT/"day1_final_q4_current_corr.csv")
fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.35, 1]})
batch1_protocol = protocol[protocol["count"] >= 2].copy()
axes[0].scatter(batch1_protocol["mean_cycle_life"], batch1_protocol["avg_charging_speed_to_80_C"], s=55, color="#1677B8", alpha=.85)
for policy_name, row in batch1_protocol.iterrows():
    axes[0].annotate(policy_name, (row["mean_cycle_life"], row["avg_charging_speed_to_80_C"]), xytext=(4, 4), textcoords="offset points", fontsize=6)
axes[0].set_xlabel("Mean Cycle Life"); axes[0].set_ylabel("Avg charging speed to 80% (C)"); axes[0].set_title("Batch 1 충전 방식별 평균 수명과 80%까지 평균 충전속도"); axes[0].grid(alpha=.2)
for b, g in f4.groupby("batch"):
    axes[1].scatter(g["avg_charging_speed_to_80_C"], g["cycle_life"], label=b, color=COLORS[b], s=32, alpha=.8)
valid4=f4[["avg_charging_speed_to_80_C","cycle_life"]].dropna()
if len(valid4)>1:
    coef4=np.polyfit(valid4["avg_charging_speed_to_80_C"],valid4["cycle_life"],1); xx=np.linspace(valid4.avg_charging_speed_to_80_C.min(),valid4.avg_charging_speed_to_80_C.max(),200)
    axes[1].plot(xx,np.polyval(coef4,xx),color="black",ls=":",lw=2,label="세 Batch 전체 선형 적합")
axes[1].set_xlabel("Avg charging speed to 80% (C)"); axes[1].set_ylabel("Cycle Life"); axes[1].set_title("80%까지 평균 충전속도와 수명"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
fig.tight_layout(); fig.savefig(OUT/"day1_final_q4.png",dpi=220,bbox_inches="tight"); plt.close(fig)

# Q5: log target correlations and multicollinearity
f = early.merge(d[["batch", "cell_id", "delta_var", "delta_abs_mean", "delta_max_abs", "log_delta_var"]], on=["batch", "cell_id"], how="left").dropna(subset=["cycle_life"]).copy()
f["log_cycle_life"] = np.log10(f["cycle_life"])
feature_cols = ["log_delta_var", "delta_var", "delta_abs_mean", "delta_max_abs", "avg_charging_speed_to_80_C", "chargetime_initial_5", "QD_fade_100", "QD_mean_100", "IR_mean_100", "Tavg_mean_100", "Tmax_mean_100"]
q5_rows=[]
for col in feature_cols:
    q5_rows.append({"feature":col, "Batch 1":f[f.batch=="Batch 1"][col].corr(f[f.batch=="Batch 1"]["log_cycle_life"]), "전체":f[col].corr(f["log_cycle_life"])})
q5_corr = pd.DataFrame(q5_rows).set_index("feature").sort_values("Batch 1", key=lambda x:x.abs(), ascending=False)
q5_corr.to_csv(OUT/"day1_final_q5_log_corr.csv")
plot_cols = ["log_cycle_life", "log_delta_var", "delta_var", "delta_abs_mean", "delta_max_abs", "avg_charging_speed_to_80_C", "chargetime_initial_5", "QD_fade_100", "QD_mean_100", "IR_mean_100", "Tavg_mean_100", "Tmax_mean_100"]
corr = f[plot_cols].corr(); corr.to_csv(OUT/"day1_final_q5_corr.csv")
fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1); ax.set_xticks(range(len(plot_cols)),plot_cols,rotation=45,ha='right',fontsize=7); ax.set_yticks(range(len(plot_cols)),plot_cols,fontsize=7)
for i in range(len(plot_cols)):
  for j in range(len(plot_cols)): ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha='center',va='center',fontsize=6,color='white' if abs(corr.iloc[i,j])>.5 else 'black')
fig.colorbar(im,ax=ax,shrink=.8); ax.set_title("log 수명과 초기 100 Cycle Feature 상관관계"); fig.tight_layout(); fig.savefig(OUT/"day1_final_q5.png",dpi=220); plt.close(fig)

print("[Q1]\n",q1.to_string())
print("[Q3]\n",pd.DataFrame(q3_rows).to_string(index=False))
print("[Q4]\n",q4.to_string())
print("[Q5 log Cycle Life correlations]\n",q5_corr.to_string())
