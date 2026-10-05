"""Gate G1 report: runs the four analyses and writes results/<run>/analysis/g1_report.md.

    python -m analysis.g1_gate --config configs/g1.yaml

Criteria (TD-MAD v2, "Cổng G1"; feasibility only, never the sign of an effect):
  1. every model has >= 40 valid trap items (pi(lure) > pi(gold))
  2. split-half reliability of pi >= 0.8 (pi_CoT, gold + lure targets, every model)
  3. annotators recognise the D level on >= 85% of messages and Cohen's kappa >= 0.7
  4. GPU-seconds per call within +-30% of the estimate in the config
A criterion whose inputs are missing is PENDING. The MAD pre-estimate is printed after the
table as exploratory context; it does not enter the decision.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from tdmad.config import ROOT, Config
from tdmad.utils import read_json, write_json


def run(module: str, config: str) -> str | None:
    p = subprocess.run([sys.executable, "-m", module, "--config", config], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if p.returncode != 0:
        return (p.stderr or p.stdout).strip().splitlines()[-1] if (p.stderr or p.stdout) else "failed"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/g1.yaml")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles are not UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = Config(args.config)
    gate = cfg["gate"]
    a = cfg.run_dir / "analysis"
    errors = {m: run(f"analysis.{m}", args.config)
              for m in ("pi_reliability", "mad_kernel", "annotation_agreement", "cost_report")}

    def load(name):
        p = a / name
        return read_json(p) if p.exists() else None

    pi, mad, ann, cost = (load("pi_summary.json"), load("mad_summary.json"),
                          load("annotation_summary.json"), load("cost_summary.json"))
    if errors["annotation_agreement"]:
        ann = None
    rows, status = [], {}

    # 1 + 2
    if pi:
        v = {m: s["valid_traps"] for m, s in pi["per_model"].items()}
        ok1 = all(x >= gate["min_valid_traps"] for x in v.values())
        status[1] = "PASS" if ok1 else "FAIL"
        detail = ", ".join(f"{m}: {x}/{pi['per_model'][m]['traps_checked']} "
                           f"(ô nghịch chặt {pi['per_model'][m]['valid_traps_strict_cell']})"
                           for m, x in v.items())
        rows.append((1, f"Câu bẫy hợp lệ ≥ {gate['min_valid_traps']} mỗi mô hình", detail, status[1]))
        rel = {m: s["reliability"] for m, s in pi["per_model"].items()}
        vals = [r["primary_min"] for r in rel.values() if r["primary_min"] is not None]
        ok2 = bool(vals) and len(vals) == len(rel) and min(vals) >= gate["min_split_half"]
        status[2] = "PASS" if ok2 else "FAIL" if vals else "PENDING"

        def sb(r):
            return f"{r['sb']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]" if r["sb"] is not None else "—"
        detail = "; ".join(f"{m}: đúng {sb(r['cot_gold'])}, bẫy {sb(r['cot_lure'])}"
                           for m, r in rel.items())
        rows.append((2, f"Split-half π (đáp án đúng và đáp án bẫy, mỗi loại) ≥ "
                        f"{gate['min_split_half']}", detail, status[2]))
    else:
        status[1] = status[2] = "PENDING"
        rows += [(1, "Câu bẫy hợp lệ", f"chưa có S1 ({errors['pi_reliability']})", "PENDING"),
                 (2, "Split-half π", "chưa có S1", "PENDING")]
    # 3
    if ann and ann.get("accuracy_A") is not None and ann.get("accuracy_B") is not None:
        ok3 = (min(ann["accuracy_A"], ann["accuracy_B"]) >= gate["min_label_accuracy"]
               and (ann["kappa_A_B"] or 0) >= gate["min_kappa"])
        status[3] = "PASS" if ok3 else "FAIL"
        detail = (f"acc A {ann['accuracy_A']:.1%}, B {ann['accuracy_B']:.1%}; "
                  f"κ(A,B) {ann['kappa_A_B']:.2f}; n = {ann['n_messages']}")
        rows.append((3, f"Nhận đúng mức D ≥ {gate['min_label_accuracy']:.0%}, κ ≥ {gate['min_kappa']}",
                     detail, status[3]))
    else:
        status[3] = "PENDING"
        rows.append((3, "Nhận đúng mức D, κ", f"chưa có nhãn ({errors['annotation_agreement']})",
                     "PENDING"))
    # 4
    if cost and cost.get("criterion4") is not None:
        status[4] = "PASS" if cost["criterion4"] else "FAIL"
        detail = "; ".join(f"{r['model']} {r['stage']}: {r['gpu_s_per_call']:.2f} s "
                           f"({r['ratio']:.2f}×)" for r in cost["rows"] if r["ratio"] is not None)
        rows.append((4, f"Chi phí/lượt trong ±{cost['tolerance']:.0%}", detail, status[4]))
    else:
        status[4] = "PENDING"
        rows.append((4, "Chi phí/lượt", f"chưa có log chi phí ({errors['cost_report']})", "PENDING"))

    decision = ("ĐI TIẾP" if all(s == "PASS" for s in status.values()) else
                "CHƯA ĐỦ DỮ LIỆU" if "FAIL" not in status.values() else "DỪNG / SỬA")
    lines = [f"# Cổng G1 — {cfg['run_name']}", "", f"**Quyết định: {decision}**", "",
             "| # | Tiêu chí | Kết quả | Trạng thái |", "|---|---|---|---|"]
    lines += [f"| {i} | {name} | {det} | {st} |" for i, name, det, st in rows]
    if mad:
        lines += ["", "## Ước tính sơ bộ từ MAD (khám phá, không dùng cho quyết định)", "",
                  "| pool · task | n | acc cá nhân v0→v5 | đa số v0→v5 | Δp₁ = p⁺(1) − p⁻(1) | sổ giữ/sụp/sửa/không sửa |",
                  "|---|---|---|---|---|---|"]
        for key, s in mad.items():
            d = s.get("delta_p1")
            ds = f"{d['est']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" if d and d.get("lo") is not None else "—"
            lg = s["ledger"]
            lines.append(f"| {s['pool']} · {s['task']} | {s['n_debates']} | {s['individual_acc_r0']:.2f}→"
                         f"{s['individual_acc_final']:.2f} | {s['majority_acc_r0']:.2f}→"
                         f"{s['majority_acc_final']:.2f} | {ds} | {lg.get('keep', 0)}/"
                         f"{lg.get('collapse', 0)}/{lg.get('fix', 0)}/{lg.get('no_fix', 0)} |")
    if pi:
        lines += ["", "## Sàng câu (S1)", "", "| model | task | acc 10 mẫu đầu | câu có cả đúng và sai | "
                  "không đọc được | bị cắt | gần mức đoán (C9) |", "|---|---|---|---|---|---|---|"]
        for m, s in pi["per_model"].items():
            for t, x in s["screening"].items():
                lines.append(f"| {m} | {t} | {x['acc_first10']:.2f} | {x['n_keep_mixed']}/{x['n_items']} | "
                             f"{x['invalid_rate']:.1%} | {x['truncated_rate']:.1%} | "
                             f"{'có' if x['near_chance_C9'] else 'không'} |")
        lines += ["", f"Câu bẫy hợp lệ với ≥ nửa pool: {pi.get('traps_valid_for_half_pool')}"]
    errs = {k: v for k, v in errors.items() if v}
    if errs:
        lines += ["", "## Bước chưa chạy được", ""] + [f"- `{k}`: {v}" for k, v in errs.items()]
    (a).mkdir(parents=True, exist_ok=True)
    (a / "g1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(a / "g1_status.json", {"status": status, "decision": decision})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
