import subprocess

models = [
    "Qwen/Qwen2.5-Math-1.5B",
    "Qwen/Qwen2.5-Math-1.5B-Instruct"
]

temperatures = [0.6, 1.0, 1.2]
dataset = "aime"
rollout_n = 64
dp_size = 1
batch_size = 16

for model in models:
    for temp in temperatures:
        model_name = model.split("/")[-1].replace("-", "_")
        output_file = f"outputs/{dataset}_{model_name}_t{temp:.1f}.jsonl"
        
        # Run inference
        cmd_infer = [
            "python", "inference.py",
            "--model", model,
            "--dataset", dataset,
            "--dp-size", str(dp_size),
            "--batch-size", str(batch_size),
            "--rollout-n", str(rollout_n),
            "--temperature", str(temp),
            "--output_file", output_file
        ]
        print(f"Running: {' '.join(cmd_infer)}")
        subprocess.run(cmd_infer, check=True)
        
        # Run evaluation
        scored_file = output_file.replace(".jsonl", "_scored.jsonl")
        cmd_eval = [
            "python", "evaluate(finished).py",
            "--input_file", output_file,
            "--output_file", scored_file
        ]
        print(f"Running: {' '.join(cmd_eval)}")
        subprocess.run(cmd_eval, check=True)