"""LLM 输出质量评测 pipeline
维度：① 准确率 Accuracy ② 格式合规率 ③ 幻觉/越界率 ④ 宏平均F1
产出：控制台汇总 + reports/report.md + reports/badcase.csv"""
import json, os, sys, csv, re
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "prompts"))
from variants import VARIANTS

LABELS = {"高风险", "中风险", "低风险"}
ROOT = os.path.join(os.path.dirname(__file__), "..")

USE_REAL = os.getenv("OPENAI_API_KEY") is not None

def call_real_llm(prompt: str) -> str:
    """真实调用（OpenAI 兼容接口）。可改 base_url 接入国内模型。"""
    from openai import OpenAI
    client = OpenAI()  # 读环境变量 OPENAI_API_KEY / OPENAI_BASE_URL
    r = client.chat.completions.create(
        model=os.getenv("EVAL_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return r.choices[0].message.content

def call_llm(prompt, text, variant):
    if USE_REAL:
        return call_real_llm(prompt)
    from mock_llm import mock_llm
    return mock_llm(prompt, text, variant)

def parse_output(raw: str):
    """解析模型输出，返回 (risk, format_ok, hallucination)。"""
    m = re.search(r'\{.*\}', raw, re.S)
    if not m:
        return None, False, False          # 非JSON → 格式不合规
    try:
        obj = json.loads(m.group())
    except Exception:
        return None, False, False
    risk = str(obj.get("risk", "")).strip()
    if risk not in LABELS:
        return risk, True, True             # JSON合规但类别越界 → 幻觉
    return risk, True, False

def load_eval_set():
    rows = []
    with open(os.path.join(ROOT, "data", "eval_set.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    return rows

def macro_f1(gold, pred):
    f1s = []
    for c in LABELS:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
    return sum(f1s) / len(f1s)

def evaluate_variant(name, template, data):
    gold, pred, badcases = [], [], []
    fmt_ok = halluc = 0
    for row in data:
        prompt = template.replace("{text}", row["text"])
        raw = call_llm(prompt, row["text"], name)
        risk, ok, hal = parse_output(raw)
        fmt_ok += ok; halluc += hal
        pred_label = risk if risk in LABELS else "解析失败"
        gold.append(row["label"]); pred.append(pred_label)
        if pred_label != row["label"]:
            badcases.append({
                "variant": name, "id": row["id"], "text": row["text"],
                "gold": row["label"], "pred": pred_label,
                "raw": raw[:80],
                "err_type": ("格式错误" if not ok else "类别幻觉" if hal else "判断错误"),
            })
    n = len(data)
    acc = sum(g == p for g, p in zip(gold, pred)) / n
    return {
        "variant": name, "accuracy": acc, "format_rate": fmt_ok / n,
        "halluc_rate": halluc / n, "macro_f1": macro_f1(gold, pred),
        "badcases": badcases,
    }

def attribute(badcases):
    """badcase 系统性归因：按错误类型 + 混淆方向统计。"""
    by_type, by_confusion = defaultdict(int), defaultdict(int)
    for b in badcases:
        by_type[b["err_type"]] += 1
        by_confusion[f'{b["gold"]}→{b["pred"]}'] += 1
    return by_type, by_confusion

def main():
    data = load_eval_set()
    results = [evaluate_variant(n, t, data) for n, t in VARIANTS.items()]
    results.sort(key=lambda r: r["accuracy"], reverse=True)

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    lines = ["# LLM 输出质量评测报告\n",
             f"评测集：{len(data)} 条信用风险样本（三分类）  |  引擎：{'真实API' if USE_REAL else '规则模拟'}\n",
             "## 一、各 Prompt 变体总览（按准确率排序）\n",
             "| 变体 | 准确率 | 格式合规率 | 幻觉率 | 宏平均F1 |",
             "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['variant']} | {r['accuracy']:.1%} | {r['format_rate']:.1%} "
                     f"| {r['halluc_rate']:.1%} | {r['macro_f1']:.2f} |")

    best, worst = results[0], results[-1]
    lift = best["accuracy"] - worst["accuracy"]
    lines += ["", "## 二、A/B 结论\n",
              f"- 最优变体：**{best['variant']}**，准确率 **{best['accuracy']:.1%}**",
              f"- 最差变体：{worst['variant']}，准确率 {worst['accuracy']:.1%}",
              f"- **准确率提升 {lift:.1%}**（{worst['accuracy']:.1%} → {best['accuracy']:.1%}）\n",
              "## 三、badcase 系统性归因（最优变体）\n"]
    bt, bc = attribute(best["badcases"])
    lines.append("**按错误类型：**")
    for k, v in sorted(bt.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}：{v} 例")
    lines.append("\n**按混淆方向（真实→预测）：**")
    for k, v in sorted(bc.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}：{v} 例")

    with open(os.path.join(ROOT, "reports", "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    all_bad = [b for r in results for b in r["badcases"]]
    with open(os.path.join(ROOT, "reports", "badcase.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["variant","id","gold","pred","err_type","text","raw"])
        w.writeheader(); w.writerows(all_bad)

    print("\n".join(lines))
    print(f"\n✅ 报告已生成：reports/report.md  badcase 明细：reports/badcase.csv")

if __name__ == "__main__":
    main()
