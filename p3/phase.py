import json
import csv
import math
from collections import defaultdict, Counter
import os

def merge_scored_jsonls_and_to_csv(
    input_files,  # List of (file_path, temperature) tuples
    output_csv
):
    """
    合并多个 scored.jsonl 文件，按温度分组计算 Pass@k 和 maj@1，输出到 CSV。
    
    Args:
        input_files: List of tuples [(file_path, temperature), ...]
        output_csv: 输出 CSV 文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    all_data = []

    # 读取所有文件
    for file_path, temp in input_files:
        print(f"Reading {file_path} with temperature={temp}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                data["temperature"] = temp  # 添加温度字段
                all_data.append(data)

    # 按温度分组
    temp2data = defaultdict(list)
    for data in all_data:
        temp = data.get('temperature', 0.6)
        temp2data[temp].append(data)

    # 计算每个温度的 Pass@k 和 maj@1
    results = []
    for temp, data_list in temp2data.items():
        # 按问题 id 分组
        problem2scores = defaultdict(list)
        problem2preds = defaultdict(list)

        for data in data_list:
            problem_id = data.get('id', 0)
            score = float(data.get('score', 0.0))
            pred = data.get('extracted_pred', None)
            problem2scores[problem_id].append(score)
            problem2preds[problem_id].append(pred)

        # 计算 Pass@k
        k_values = [1, 2, 4, 8, 16]
        pass_at_k = {}
        for k in k_values:
            pass_vals = []
            for problem_id, scores in problem2scores.items():
                n = len(scores)
                c = sum(1 for s in scores if s > 0.0)
                if k > n:
                    pass_at_k_val = 1.0
                else:
                    try:
                        pass_at_k_val = 1.0 - (math.comb(n - c, k) / math.comb(n, k))
                    except (ValueError, ZeroDivisionError):
                        pass_at_k_val = 0.0
                pass_vals.append(pass_at_k_val)
            pass_at_k[k] = sum(pass_vals) / len(pass_vals) * 100.0 if pass_vals else 0.0

        # 计算 Majority Vote @1
        maj_correct = 0
        maj_total = 0
        for problem_id, preds in problem2preds.items():
            maj_total += 1
            counts = Counter(preds)
            max_freq = max(counts.values())
            candidates = [p for p, c in counts.items() if c == max_freq]
            if len(candidates) == 1:
                winner = candidates[0]
            else:
                candidates.sort(key=lambda x: str(x))
                winner = candidates[0]
            # 检查 winner 是否正确
            is_correct = False
            for data in data_list:
                if data.get('id') == problem_id and data.get('extracted_pred') == winner:
                    if float(data.get('score', 0.0)) > 0.0:
                        is_correct = True
                    break
            if is_correct:
                maj_correct += 1
        maj_at_1 = (maj_correct / maj_total) * 100.0 if maj_total > 0 else 0.0

        # 保存结果
        results.append({
            'temperature': temp,
            'pass@1': pass_at_k[1],
            'pass@2': pass_at_k[2],
            'pass@4': pass_at_k[4],
            'pass@8': pass_at_k[8],
            'pass@16': pass_at_k[16],
            'maj@1': maj_at_1
        })

    # 按温度排序
    results.sort(key=lambda x: x['temperature'])

    # 写入 CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['temperature', 'pass@1', 'pass@2', 'pass@4', 'pass@8', 'pass@16', 'maj@1'])
        for row in results:
            writer.writerow([
                row['temperature'],
                round(row['pass@1'], 2),
                round(row['pass@2'], 2),
                round(row['pass@4'], 2),
                round(row['pass@8'], 2),
                round(row['pass@16'], 2),
                round(row['maj@1'], 2)
            ])

    print(f"✅ CSV 文件已保存：{output_csv}")

# ============================
# 使用示例
# ============================
if __name__ == "__main__":
    # 定义三个温度的输入文件和对应的温度值
    input_files = [
        ("outputs/aime25_qwen_instruct_t0.6_scored.jsonl", 0.6),
        ("outputs/aime25_qwen_instruct_t1.0_scored.jsonl", 1.0),
        ("outputs/aime25_qwen_instruct_t1.2_scored.jsonl", 1.2),
    ]
    output_file = "outputs/passk&maj_qwen_instruct.csv"

    merge_scored_jsonls_and_to_csv(input_files, output_file)