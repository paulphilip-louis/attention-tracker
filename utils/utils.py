# Imports
import math
import re
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import matplotlib.ticker as ticker

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INST_TOKEN = ""
SEP_TOKEN = "Data: "



def generate_dataset(tokenizer, normal_instruction, sentences, attacks, inst_token=INST_TOKEN, sep_token=SEP_TOKEN):
    normal_dataset = []
    attack_dataset = []
    if isinstance(attacks, str):
        attacks = [attacks]*len(sentences)

    for sentence, attack in zip(sentences, attacks):
        normal_message = [
            {'role':'system', 'content':normal_instruction},
            {'role':'user', 'content': sep_token + sentence}
        ]
        injected_message = [
            {'role':'system', 'content':normal_instruction},
            {'role':'user', 'content': sep_token + sentence + '\n' + attack}
        ]
        normal_data = tokenizer.apply_chat_template(normal_message, tokenize=False, add_generation_prompt=True)
        injected_data = tokenizer.apply_chat_template(injected_message, tokenize=False, add_generation_prompt=True)

        normal_dataset.append(normal_data)
        attack_dataset.append(injected_data)


    return normal_dataset, attack_dataset

def get_raw_activations(model, prompt, device):
    tokens = model.to_tokens(prompt).to(device)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    
    raw = {}
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads
    for l in range(n_layers):
        attn = cache["pattern", l]  # shape: [n_heads, seq_len, seq_len]
        for h in range(n_heads):
            raw[(l, h)] = attn[h, -1, :].float().cpu().numpy()  # last token attending to all
    return raw

def get_activations(model, prompt, device):
    tokenized_prompt = model.to_tokens(prompt).to(device)
    
    tokenized_prompt_str = model.to_str_tokens(prompt)
    for i, t in enumerate(tokenized_prompt_str):
        if 'Data' in t:
            end_inst = i
            break

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    layers = [f"blocks.{i}.attn.hook_pattern" for i in range(n_layers)]

    _, cache = model.run_with_cache(tokenized_prompt, remove_batch_dim=True, names_filter = layers)

    attention_scores = torch.zeros(n_layers, n_heads)
    for i in range(n_layers):
        attention_scores[i] = cache[layers[i]][:, -1, :end_inst].sum(dim=1)
        
    return attention_scores

def get_mean_and_std(model, dataset, device=DEVICE):
    s = []
    for prompt in dataset:
        s.append(get_activations(model, prompt, device=DEVICE).unsqueeze(0))
    s = torch.concatenate(s)
    mu = torch.mean(s, dim=0)
    std = torch.std(s, dim=0)
    return mu, std


def score_heads(model, normal_dataset, attack_dataset, k=4, device=DEVICE):
    mu_N, std_N = get_mean_and_std(model, normal_dataset)
    mu_A, std_A = get_mean_and_std(model, attack_dataset)

    scores = (mu_N - k*std_N) - (mu_A + k*std_A)
    return scores

def sus_heads(scores, eps=0):
    """
    scores : torch.Tensor(n_layers, n_heads)

    From the scores of the heads, return the indices of the important heads
    """
    H_i = torch.where(scores > eps)
    indices = []
    for i, j in zip(*H_i):
        indices.append((i.item(), j.item()))
    return indices



def f_s(model, model_family, heads, instruction, data, ):
    """
    Exact replication of paper's focus score for Qwen2-1.5B via HookedTransformer.
    
    heads: list of (layer, head) tuples
    instruction: raw instruction string
    data: raw data string (without "Data: " prefix — added internally)
    get_activations_fn: your existing get_activations function, 
                        returns dict {(layer, head): scalar attention score}
    """
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user",   "content": "Data: " + data}
    ]
    text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Token lengths of raw strings, matching paper exactly
    instruction_len = len(model.tokenizer.encode(instruction))
    data_len        = len(model.tokenizer.encode("Data: " + data))
    
    if model_family=="llama3.2":
        inst_start = 26
    elif model_family=="qwen2.5":
        inst_start = 3
    else:
        raise ValueError("model family invalid")
    
    inst_end   = inst_start + instruction_len
    
    total_len  = len(model.tokenizer.encode(text))
    data_start = total_len - 5 - data_len
    data_end   = total_len - 5
    
    # Get raw attention tensors per layer per head: shape [seq_len] for last token
    # get_activations returns the full attention map — we need raw weights, not the
    # aggregated scalar your current version returns.
    # You may need a separate get_raw_activations that returns per-position weights.
    raw_attn = get_raw_activations(model, prompt=text, device=DEVICE)  # shape: {(l,h): np.array of shape [seq_len]}
    
    epsilon = 1e-8
    scores = []
    for l, h in heads:
        attn = raw_attn[(l, h)]  # attention from last token to all positions
        
        inst_attn = np.sum(attn[inst_start:inst_end])
        data_attn = np.sum(attn[data_start:data_end])
        
        normalized = inst_attn / (inst_attn + data_attn + epsilon)
        scores.append(normalized)
    
    return np.mean(scores)

##############################
# --------- TF-IDF --------- #
##############################

# ── Stop words ──────────────────────────────────────────────────────────────

STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","through",
    "during","before","after","above","below","to","from","up","down","in",
    "out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","can","will","just","don","should","now","also","would",
    "could","may","might","shall","let","like","well","much","get","got",
    "make","made","one","two","thing","things","way","even","still","us",
    "something","anything","everything","many","really","every","go","know",
    "see","think","take","come","back","use","used","using","try","need",
    "want","say","said","new","first","last","right","good","sure","however",
    "since","whether","yet","though","already","rather","quite","often","data"
}


# ── Tokenizer ───────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip non-alpha, remove stop words and short tokens."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    return [w for w in words if w not in STOP_WORDS]



# ── TF-IDF ──────────────────────────────────────────────────────────────────

def compute_tfidf(docs: list[str]) -> dict[str, dict]:
    """
    Parameters
    ----------
    docs : list of strings – each string is one AI reply.

    Returns
    -------
    dict  {word: {"tfidf": float, "count": int}}
          tfidf  = mean across documents of  (tf * idf)
          count  = total raw occurrences across all documents
    """
    n = len(docs)
    if n == 0:
        return {}

    # tokenize every document
    doc_tokens = [_tokenize(d) for d in docs]

    # document frequency (how many docs contain each term)
    df: dict[str, int] = {}
    for tokens in doc_tokens:
        for w in set(tokens):
            df[w] = df.get(w, 0) + 1

    # per-document TF-IDF, then average across docs
    tfidf_sum: dict[str, float] = {}
    global_count: dict[str, int] = {}

    for tokens in doc_tokens:
        tf = Counter(tokens)
        max_tf = max(tf.values()) if tf else 1
        for term, cnt in tf.items():
            normalized_tf = cnt / max_tf          # augmented TF
            idf = math.log(n / df[term])          # inverse doc freq
            tfidf_sum[term] = tfidf_sum.get(term, 0.0) + normalized_tf * idf
            global_count[term] = global_count.get(term, 0) + cnt

    return {
        term: {"tfidf": tfidf_sum[term] / n, "count": global_count[term]}
        for term in tfidf_sum
    }





# ── Plot ────────────────────────────────────────────────────────────────────

def plot_top_k(scores: dict[str, dict], k: int = 15) -> None:
    """
    Select the top-k words by TF-IDF score, then plot a horizontal bar chart
    where bar length = raw word count.

    Parameters
    ----------
    scores : output of compute_tfidf()
    k      : number of top words to display
    """
    if not scores:
        print("No scores to plot.")
        return

    # rank by tfidf, keep top k
    ranked = sorted(scores.items(), key=lambda x: x[1]["tfidf"], reverse=True)[:k]
    # reverse so highest is at the top of the chart
    ranked.reverse()

    words  = [w for w, _ in ranked]
    counts = [v["count"] for _, v in ranked]
    tfidf  = [v["tfidf"] for _, v in ranked]

    # colour gradient: higher tfidf → more saturated
    max_tfidf = max(tfidf) if tfidf else 1
    colors = [plt.cm.YlOrRd(0.3 + 0.65 * (t / max_tfidf)) for t in tfidf]

    fig, ax = plt.subplots(figsize=(10, max(5, k * 0.45)))
    bars = ax.barh(words, counts, color=colors, edgecolor="white", linewidth=0.4)

    # annotate each bar with count + tfidf
    for bar, c, t in zip(bars, counts, tfidf):
        ax.text(
            bar.get_width() + max(counts) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{c}  (tfidf {t:.4f})",
            va="center", fontsize=9, color="#444",
        )

    ax.set_xlabel("Raw Word Count", fontsize=11)
    ax.set_title(f"Top-{k} Words by TF-IDF  (bars = raw count)", fontsize=13, pad=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.18)
    plt.tight_layout()
    plt.savefig("plots/tfidf_top_words_2.png", dpi=150)
    plt.show()
    print("Saved → tfidf_top_words.png")
