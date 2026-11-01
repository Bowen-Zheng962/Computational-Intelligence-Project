"""Report every predeclared method; never rewrite the earlier official test."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
METHODS = ("fixed", "random_search", "genetic_search", "verified_random", "verified_genetic", "verified_neighbor")
NAMES = {"fixed": "固定配置", "random_search": "原始随机搜索", "genetic_search": "原始遗传搜索",
         "verified_random": "多种子复核随机搜索", "verified_genetic": "多种子复核遗传搜索", "verified_neighbor": "复核＋邻域变异遗传搜索"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "results/development_v1")
    args = parser.parse_args()
    out = args.output.resolve()
    meta = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    all_summary = json.loads((out / "audit_summary.json").read_text(encoding="utf-8"))
    summary = all_summary["methods"]
    verification = json.loads((out / "verification.json").read_text(encoding="utf-8"))
    assert verification["status"] == "passed"
    accounting = json.loads((out / "fit_accounting.json").read_text(encoding="utf-8"))
    with (out / "pairwise_comparisons.csv").open(encoding="utf-8", newline="") as f:
        pairs = {(r["method"], r["baseline"]): r for r in csv.DictReader(f)}
    choices = json.loads((out / "search_choices_frozen.json").read_text(encoding="utf-8"))
    concise_rows = []
    for method in METHODS:
        concise_rows.append({"method": method, "name": NAMES[method], **summary[method]})
    with (out / "results_table.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(concise_rows[0]))
        writer.writeheader()
        writer.writerows(concise_rows)
    report = ["# 进化搜索改进：开发实验结果", "",
              "本报告来自实际重新训练、冻结后审计的结果；旧正式测试结果未被改写。EO（Evolutionary Optimization，进化优化）在这里采用 GA（Genetic Algorithm，遗传算法）。", "",
              "## 1. 这次改变了什么", "",
              "第一项改变是候选评价预算分配：原始方法每个候选单种子训练 20 轮；复核方法先让 16 个候选训练 10 轮，再让前 4 个续训并用第二个种子检查，前 2 个加入第三个种子。两种评价方式的逻辑总预算都为 320 轮。", "",
              "第二项改变仅用于最后一组：变异触发后，以 80% 概率尝试相邻维度／正则档位，20% 概率采用原全局变异。普通复核 GA 是它的消融对照。随机搜索也有相同的复核版本，避免把更多复核预算误当成进化操作的优势。", "",
              "## 2. 数据与预先固定的比较", "",
              f"沿用 118 次训练飞行（1,445 个窗口）。原验证集按种子 2041 划为 19 次选型飞行（{meta['selection_windows']} 个窗口）和 20 次审计飞行（{meta['audit_windows']} 个窗口）。所有模型检查点只按选型集选择；全部方法冻结后统一评价审计集。", "",
              "搜索种子为 11、22、33、44、55、66、77、88、99、110，最终训练种子为 511、522、533。每种搜索使用 16 个不同候选，相同搜索种子共享初始 4 个配置。结果平均所有预定运行，不选择最好的种子。", "",
              "**本轮是开发实验，不是新的独立正式盲测。** 旧测试结果此前已经可见，因此新方法与结果必须作为探索性开发证据。新程序不构造原 39 次正式测试飞行的窗口，不能把本表审计数字当成旧测试表的升级数字。", "",
              "## 3. 全部方法的审计结果", "",
              "J 是按训练集能耗标准差缩放、按飞行等权的 10/20/30 秒平均能耗绝对误差，越小越好。SD（Standard Deviation，标准差）按各训练种子或搜索种子的边际均值计算。", "",
              "| 方法 | 审计 J | 训练种子 SD | 搜索种子 SD | 相对固定配置误差下降 |",
              "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        r = summary[method]
        sd = "—" if r["search_seed_sd"] is None else f"{r['search_seed_sd']:.5f}"
        report.append(f"| {NAMES[method]} | {r['energy_score']:.6f} | {r['training_seed_sd']:.5f} | {sd} | {r['improvement_over_fixed_percent']:+.2f}% |")
    report += ["", "| 方法 | 10 秒能耗 MAE（Wh） | 20 秒（Wh） | 30 秒（Wh） | 功率 MAE（W） | 平均参数量 |",
               "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        r = summary[method]
        report.append(f"| {NAMES[method]} | {r['energy_mae_10s_wh']:.5f} | {r['energy_mae_20s_wh']:.5f} | {r['energy_mae_30s_wh']:.5f} | {r['power_mae_w']:.3f} | {r['mean_parameters']:.0f} |")
    report += ["", "MAE（Mean Absolute Error，平均绝对误差）；Wh（watt-hour，瓦时）；W（watt，瓦特）。误差下降不等于无人机实际节省的电量。", "",
               "## 4. 必须看的配对比较", "",
               "| 方法相对对照 | 误差下降 | 按飞行配对重采样 95% 区间 | 误差更低的飞行数 |",
               "|---|---:|---:|---:|"]
    contrasts = [("genetic_search", "random_search"), ("verified_genetic", "verified_random"),
                 ("verified_genetic", "genetic_search"), ("verified_neighbor", "verified_genetic"),
                 ("verified_neighbor", "verified_random"), ("verified_genetic", "fixed"), ("verified_neighbor", "fixed")]
    for key in contrasts:
        r = pairs[key]
        report.append(f"| {NAMES[key[0]]} / {NAMES[key[1]]} | {float(r['relative_improvement_percent']):+.2f}% | [{float(r['paired_flight_95pct_low']):+.2f}%, {float(r['paired_flight_95pct_high']):+.2f}%] | {r['flights_with_lower_error']}/20 |")
    primary = summary["verified_genetic"]["improvement_over_verified_random_percent"]
    neighbor = summary["verified_neighbor"]["improvement_over_verified_genetic_percent"]
    primary_direction = "更低" if primary > 0 else "更高"
    neighbor_direction = "更低" if neighbor > 0 else "更高"
    report += ["", f"主要匹配比较中，复核遗传搜索相对复核随机搜索的误差均值{primary_direction}，相对降幅为 {primary:+.2f}%。新增邻域变异相对普通复核 GA 的误差均值{neighbor_direction}，相对降幅为 {neighbor:+.2f}%。正号表示改善，负号表示退步。", "",
               "区间在已冻结模型上对 20 次审计飞行重采样 10,000 次得到，未作多重比较校正。若区间跨零，只能报告观察到的均值差异，不能声称稳定优势或等效。", "",
               "## 5. 选中配置分布", "",
               "| 方法 | 配置编号与选中次数（10 次搜索） |", "|---|---|"]
    for method in METHODS[1:]:
        freq = Counter(r["config_id"] for r in choices if r["method"] == method)
        report.append(f"| {NAMES[method]} | " + "; ".join(f"{k}: {v} 次" for k, v in sorted(freq.items())) + " |")
    report += ["", "配置编号按潜在维度 4、8、12、16、24、32 与正则强度 0.0001、0.0003、0.001、0.003、0.01 的笛卡尔积从 0 编号。", "",
               "## 6. 资源、复核与解释边界", "",
               f"本轮共有 50 次搜索，逻辑搜索预算 {accounting['logical_search_epochs']:,} 轮；共享缓存后的实际唯一搜索训练为 {accounting['actual_unique_search_epochs']:,} 轮。最终有 {accounting['logical_final_slots']} 个比较位置，但仅 {accounting['unique_final_models']} 个不同模型，复用不计作独立重复。最终模型共训练 {accounting['actual_final_epochs']} 轮。", "",
               f"结果验证通过：重放 {verification['search_runs_replayed']} 次搜索，核验 {verification['validation_checkpoints_checked']} 个选型检查点，重算 {verification['audit_comparison_rows_recomputed']} 条审计对比和 {verification['audit_per_flight_rows_recomputed']} 条逐飞行结果；重载模型的审计预测最大差异为 {verification['max_reloaded_audit_forecast_difference_w']:g} W。", "",
               "此处只验证同一给定航线条件下的功率／能耗预测，没有验证通用世界模型控制、路径规划或实际飞行节能。未来轨迹被视为已知计划，过滤后的数据和单次开发划分限制了结论。", "",
               "本报告不按审计结果继续改参数；全部预定条件均列出。后续若需要形成新的确认性结论，应使用独立数据或预先设计的外层评估。"]
    # A reproduction must not overwrite the shipped historical report.
    (out / "RESULTS_ZH.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})
    labels = ("Fixed", "Random", "Original GA", "Verified\nrandom", "Verified\nGA", "Verified\nneighbor GA")
    colors = ("#8994a6", "#419b8a", "#7b9ec2", "#31776a", "#3a6fa2", "#bb793d")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = np.arange(6)
    axes[0].bar(x, [summary[m]["energy_score"] for m in METHODS], color=colors,
                yerr=[summary[m]["training_seed_sd"] for m in METHODS], capsize=3)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Development-audit normalized energy MAE")
    axes[0].set_title("All predeclared methods; bars: training-seed SD")
    for i, m in enumerate(METHODS):
        axes[1].plot((10, 20, 30), [summary[m][f"energy_mae_{h}s_wh"] for h in (10, 20, 30)], "o-", color=colors[i], label=labels[i].replace("\n", " "))
    axes[1].set(xlabel="Energy horizon (seconds)", ylabel="Energy MAE (Wh)", title="Held-out development audit (20 flights)")
    axes[1].set_xticks((10, 20, 30))
    axes[1].legend(frameon=False, fontsize=7)
    fig.savefig(figures / "audit_comparison.png")
    fig.savefig(figures / "audit_comparison.pdf")
    plt.close(fig)
    concise = {m: {k: summary[m][k] for k in ("energy_score", "improvement_over_fixed_percent", "improvement_over_random_search_percent", "improvement_over_verified_random_percent")}
               for m in METHODS}
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
