import os
import re

results_dir = "results"
rows = []

for filename in sorted(os.listdir(results_dir)):
    if not filename.endswith(".txt"):
        continue
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "r") as f:
        content = f.read()
    
    match = re.search(r"Average Acc:([\d.]+)\s+Average Acc5:([\d.]+)\s+Average Acc_var:([\d.]+)\s+Average Acc_var5:([\d.]+)", content)
    if match:
        rows.append({
            "run": filename.replace(".txt", ""),
            "acc": float(match.group(1)),
            "acc5": float(match.group(2)),
            "acc_var": float(match.group(3)),
            "acc_var5": float(match.group(4)),
        })

# sort by attack accuracy ascending (best defense first)
rows.sort(key=lambda x: x["acc"])

print(f"{'Run':<45} {'Atk Acc':>8} {'Top-5 Acc':>10} {'Acc Var':>10} {'Acc5 Var':>10}")
print("-" * 85)
for r in rows:
    print(f"{r['run']:<45} {r['acc']:>8.2f} {r['acc5']:>10.2f} {r['acc_var']:>10.4f} {r['acc_var5']:>10.4f}")