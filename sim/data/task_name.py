import os
import json

input_root = "anns_single"
output_dir = "tasks"
os.makedirs(output_dir, exist_ok=True)


task_counts = {} 

def extract_tasks_from_file(file_path):
    tasks = set()
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            for conv in item.get("conversations", []):
                value = conv.get("value", "")
                if "The task is" in value:
                    start = value.find("The task is \"") + len("The task is \"")
                    end = value.find("\"", start)
                    if end != -1:
                        task = value[start:end]
                        tasks.add(task)
                        count += 1  
    return tasks, count


for subfolder in os.listdir(input_root):
    subfolder_path = os.path.join(input_root, subfolder)
    if os.path.isdir(subfolder_path):
        train_json_path = os.path.join(subfolder_path, "train.json")
        if os.path.isfile(train_json_path):
            tasks, count = extract_tasks_from_file(train_json_path)
            task_counts[subfolder] = count


            output_path = os.path.join(output_dir, f"{subfolder}.txt")
            with open(output_path, "w", encoding="utf-8") as out_file:
                for task in sorted(tasks):
                    out_file.write(task + "\n")

            print(f"Processed: {train_json_path} → {output_path} ({count} task entries)")
        else:
            print(f"Warning: No train.json found in {subfolder_path}")

# 保存任务总条数（未去重）
total_counts_path = os.path.join(output_dir, "task_total_counts.txt")
with open(total_counts_path, "w", encoding="utf-8") as f:
    for task_name, count in sorted(task_counts.items()):
        f.write(f"{task_name}: {count}\n")

print(f"Total task entry counts saved to: {total_counts_path}")