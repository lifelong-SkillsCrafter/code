import torch
import torch.nn.functional as F
import os
from transformers import (
    BertTokenizer, BertModel,
    CLIPProcessor, CLIPModel
)
import re


def extract_task_number(name):
    match = re.search(r'\d+', name)
    return int(match.group()) if match else float('inf')


def load_encoder(encoder_type="clip", device="cuda"):

    if encoder_type == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased").to(device)
    elif encoder_type == "clip":
        tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch16-336")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch16-336").to(device)
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")
    model.eval()
    return model, tokenizer


def encode_texts(model, tokenizer, texts, encoder_type="clip", device="cuda"):

    with torch.no_grad():
        if encoder_type == "bert":
            tokenized = tokenizer(texts, padding="longest", return_tensors="pt").to(device)
            outputs = model(**tokenized)
            cls_features = outputs.last_hidden_state[:, 0, :]
            return cls_features

        elif encoder_type == "clip":
            # tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
            inputs = tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
            print(f"[调试] token_embedding.weight.shape = {model.text_model.embeddings.token_embedding.weight.shape}")
            text_features = model.get_text_features(**inputs)
            return text_features


def svd_decompose(tensor):
    tensor = tensor.float()
    U, S, Vth = torch.linalg.svd(tensor, full_matrices=True)
    return U, S, Vth


def select_rank_by_energy(singular_values: torch.Tensor, energy_threshold: float = 0.6) -> int:

    sigma_squared = singular_values ** 2
    total_energy = sigma_squared.sum()
    cumulative_energy = torch.cumsum(sigma_squared, dim=0) / total_energy

    candidates = torch.nonzero(cumulative_energy >= energy_threshold, as_tuple=False)

    if candidates.numel() == 0:
 
        return 10
    else:
        k = candidates[0].item()
        return k + 1


def compute_vt_v_product(Vth, S):


    r = select_rank_by_energy(S, energy_threshold=0.9)
    print(f'rankr is {r}')
    Vh = Vth.T  # (num_texts, rank)
    Vr = Vh[:, :r]
    Vtr = Vr.T

    product_matrix = Vr @ Vtr  # (num_texts, num_texts)
    return product_matrix


def process_all_tasks(base_dir, model, tokenizer, encoder_type="clip", device="cuda"):
    for subfolder in sorted(os.listdir(base_dir)):
        sub_path = os.path.join(base_dir, subfolder)
        if not os.path.isdir(sub_path):
            continue

        task_txt_name = f"{subfolder}.txt"
        task_txt_path = os.path.join(sub_path, task_txt_name)

        if not os.path.isfile(task_txt_path):
            print(f"[跳过] 未找到文件: {task_txt_path}")
            continue

        print(f"[处理] {task_txt_path}")

        with open(task_txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            print(f"[跳过] 空文件: {task_txt_path}")
            continue

        embedding_matrix = encode_texts(model, tokenizer, lines, encoder_type=encoder_type, device=device)
        U, S, Vth = svd_decompose(embedding_matrix)
        skill_subspace = compute_vt_v_product(Vth, S)

        print(f'embedding_matrix维度: {embedding_matrix.shape}')
        print(f'U维度: {U.shape}，S维度: {S.shape}，Vth维度: {Vth.shape}，skill_subspace维度: {skill_subspace.shape}')

        save_path = os.path.join(sub_path, "skill_subspace.pt")
        torch.save(skill_subspace, save_path)
        print(f"[保存] skill_subspace -> {save_path}\n")


def infer_task_for_text(text, model, tokenizer, task_dir, encoder_type="clip", device="cuda"):
    import torch.nn.functional as F


    text_feature = encode_texts(model, tokenizer, [text], encoder_type=encoder_type, device=device)
    text_feature = text_feature.squeeze(0)  # shape: (d,)

    scores = {}

    for subfolder in sorted(os.listdir(task_dir)):
        sub_path = os.path.join(task_dir, subfolder)
        subspace_path = os.path.join(sub_path, "skill_subspace.pt")
        if not os.path.isfile(subspace_path):
            print(f"[跳过] {subfolder}: skill_subspace.pt 不存在")
            continue

        P_t = torch.load(subspace_path).to(device)  


        projected = P_t @ text_feature  # (d,)


        cos_sim = F.cosine_similarity(text_feature.unsqueeze(0), projected.unsqueeze(0)).item()
        scores[subfolder] = cos_sim

  
    adjusted_scores = {k: (v + 1) ** 5 for k, v in scores.items()}


    values_tensor = torch.tensor(list(adjusted_scores.values()), dtype=torch.float32)
    softmax_values = F.softmax(values_tensor, dim=0)

    normalized_scores = {
        k: softmax_values[i].item()
        for i, k in enumerate(adjusted_scores.keys())
    }


    top_task, top_prob = max(normalized_scores.items(), key=lambda x: x[1])


    if top_prob < 0.85:
        adjusted_top_prob = 0.95
        remaining_prob = 1.0 - adjusted_top_prob
        total_other = sum(v for k, v in normalized_scores.items() if k != top_task)

        normalized_scores = {
            k: (adjusted_top_prob if k == top_task else (v / total_other) * remaining_prob)
            for k, v in normalized_scores.items()
        }

    return top_task, normalized_scores


def infer_task_for_text_pretasks(text, model, tokenizer, task_dir, encoder_type="clip", device="cuda", task_id=None):

    text_feature = encode_texts(model, tokenizer, text, encoder_type=encoder_type, device=device)
    text_feature = text_feature.squeeze(0)

    scores = {}
    all_subfolders = sorted(os.listdir(task_dir), key=extract_task_number)
    selected_subfolders = all_subfolders if task_id is None else all_subfolders[:task_id]

    for subfolder in selected_subfolders:
        sub_path = os.path.join(task_dir, subfolder)
        subspace_path = os.path.join(sub_path, "skill_subspace.pt")
        if not os.path.isfile(subspace_path):
            print(f"[跳过] {subfolder}: skill_subspace.pt 不存在")
            continue

        P_t = torch.load(subspace_path).to(device)
        projected = P_t @ text_feature
        cos_sim = F.cosine_similarity(text_feature.unsqueeze(0), projected.unsqueeze(0)).item()
        scores[subfolder] = cos_sim

    if not scores:
        return None, {}

    adjusted_scores = {k: (v + 1) ** 5 for k, v in scores.items()}
    values_tensor = torch.tensor(list(adjusted_scores.values()), dtype=torch.float32)
    softmax_values = F.softmax(values_tensor, dim=0)

    normalized_scores = {
        k: softmax_values[i].item()
        for i, k in enumerate(adjusted_scores.keys())
    }

    top_task, top_prob = max(normalized_scores.items(), key=lambda x: x[1])

    return top_task, normalized_scores


encoder_type = "clip" 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_encoder(encoder_type=encoder_type, device=device)
base_dir = "sim/tasks"

process_all_tasks(base_dir=base_dir,model=model,tokenizer=tokenizer,encoder_type=encoder_type,device=device)

text = "wrench the pink light burner"
top_task, score_dict = infer_task_for_text_pretasks(
    text=text,
    model=model,
    tokenizer=tokenizer,
    task_dir=base_dir,  
    task_id=4,
    encoder_type=encoder_type,
    device=device
)
print(score_dict)

text = "wrench the pink light burner"
top_task, score_dict = infer_task_for_text(
    text=text,
    model=model,
    tokenizer=tokenizer,
    task_dir=base_dir,  
    encoder_type=encoder_type,
    device=device
)

print(f"最可能的任务: {top_task}")
print("各任务概率分布:")
for task, prob in sorted(score_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"{task}: {prob:.4f}")




