#!/usr/bin/env python3
"""
Research & Code Aggregator & Classifier
Funnels papers (arXiv, HuggingFace Daily Papers) and code repositories (GitHub)
across:
- Pre-training (Distributed systems, Megatron, 3D parallelism, scaling)
- Post-training (RLHF, DPO, PPO, GRPO, quantization, fine-tuning)
- Kernel Optimization (GEMM, Attention, MoE, Normalization, FlashInfer/Triton)

Sorts entries by targeted topic sections into structured Markdown/JSON data.
"""

import sys
import os
import json
import re
import datetime
import urllib.request
import xml.etree.ElementTree as ET

CATEGORIES = {
    "Attention & Sequence Kernels": {
        "keywords": [
            "attention", "flashattention", "flash-attention", "ringattention", 
            "linear attention", "mla", "multi-head latent attention", "kv cache", 
            "paged attention", "vllm", "flashinfer", "context extension", "long-context"
        ],
        "items": []
    },
    "MoE (Mixture of Experts) & Routing": {
        "keywords": [
            "moe", "mixture of experts", "mixture-of-experts", "expert routing", 
            "switch transformer", "deepseek-v2", "deepseek-v3", "token dispatch",
            "ep parallelism", "expert parallelism", "fused moe"
        ],
        "items": []
    },
    "GEMM, Tensor Cores & Low-Precision": {
        "keywords": [
            "gemm", "matmul", "matrix multiplication", "tensor core", "wgmma", 
            "cutlass", "cute", "fp8", "fp4", "nvfp4", "mxfp4", "mxfp8", "quantization",
            "awq", "gptq", "bitsandbytes", "block scaling", "sm90", "sm100"
        ],
        "items": []
    },
    "Kernel Compilers & DSLs (Triton, TileLang, CUDA)": {
        "keywords": [
            "triton", "tilelang", "thunderkittens", "cutedsl", "tvm", "halide",
            "autotuning", "kernel generation", "cuda optimization", "warp specialization",
            "tma", "tensor memory accelerator"
        ],
        "items": []
    },
    "Pre-training & Distributed Infrastructure": {
        "keywords": [
            "pre-training", "pretraining", "foundation model", "megatron", "deepspeed", 
            "fsdp", "tensor parallel", "pipeline parallel", "sequence parallel",
            "3d parallelism", "context parallelism", "data parallel", "cluster scheduling",
            "collective communication", "nccl"
        ],
        "items": []
    },
    "Post-training, Alignment & RL": {
        "keywords": [
            "post-training", "post training", "rlhf", "dpo", "ppo", "grpo", 
            "reinforcement learning from human feedback", "instruction tuning", "sft",
            "preference optimization", "reasoning models", "distillation", "lora", "qlora"
        ],
        "items": []
    }
}

ARXIV_QUERIES = [
    '(ti:"attention" OR abs:"attention" OR ti:"kernel" OR abs:"kernel")',
    '(ti:"MoE" OR abs:"mixture of experts" OR ti:"GEMM")',
    '(ti:"pre-training" OR ti:"post-training" OR ti:"RLHF" OR ti:"DPO")'
]

def fetch_arxiv_papers(max_results=30):
    papers = []
    base_url = "http://export.arxiv.org/api/query"
    query = " OR ".join(ARXIV_QUERIES)
    params = f"?search_query={urllib.parse.quote(query)}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    url = base_url + params
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (AI Research Pipeline Bot)'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
            root = ET.fromstring(content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                link = entry.find('atom:id', ns).text.strip()
                published = entry.find('atom:published', ns).text.strip()[:10]
                
                authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
                first_author = authors[0] if authors else "Unknown"
                
                papers.append({
                    "title": title,
                    "summary": summary,
                    "url": link,
                    "date": published,
                    "author": first_author,
                    "type": "Paper"
                })
    except Exception as e:
        print(f"[Warning] Failed to fetch arXiv papers: {e}", file=sys.stderr)
    return papers

def classify_item(title, summary):
    text = (title + " " + summary).lower()
    scores = {}
    for cat_name, cat_data in CATEGORIES.items():
        score = 0
        for kw in cat_data["keywords"]:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                score += 2 if kw in title.lower() else 1
        if score > 0:
            scores[cat_name] = score
            
    if scores:
        return max(scores, key=scores.get)
    return "Kernel Compilers & DSLs (Triton, TileLang, CUDA)"

def main():
    print("🚀 Fetching latest papers and code...")
    papers = fetch_arxiv_papers(max_results=50)
    print(f"Retrieved {len(papers)} candidate entries from arXiv.")
    
    for item in papers:
        cat = classify_item(item["title"], item["summary"])
        CATEGORIES[cat]["items"].append(item)
        
    date_str = datetime.date.today().isoformat()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, "data", "daily")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{date_str}.json")
    md_path = os.path.join(output_dir, f"{date_str}.md")
    
    # Save raw JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(CATEGORIES, f, indent=2, ensure_ascii=False)
        
    # Generate structured Markdown report
    lines = [
        f"# Automated Research & Kernel Funnel ({date_str})\n",
        "> Automated daily CI/CD snapshot indexing fresh Pre-training, Post-training, and GPU Kernel Optimizations.\n"
    ]
    
    for cat_name, cat_data in CATEGORIES.items():
        lines.append(f"## {cat_name}\n")
        if not cat_data["items"]:
            lines.append("_No new tracked entries for this section in the latest batch._\n")
            continue
            
        lines.append("| Title / Paper | Author / Source | Date | Key Highlights |")
        lines.append("|---|---|---|---|")
        for item in cat_data["items"]:
            title_link = f"[{item['title']}]({item['url']})"
            desc_snippet = item['summary'][:160].replace('|', '-') + "..."
            lines.append(f"| {title_link} | {item['author']} | {item['date']} | {desc_snippet} |")
        lines.append("")
        
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"✅ Generated report: {md_path}")
    print(f"✅ Generated dataset: {json_path}")

if __name__ == "__main__":
    main()
