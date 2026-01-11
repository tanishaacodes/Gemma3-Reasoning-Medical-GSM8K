# Gemma3-Reasoning-Medical-GSM8K
A learning experiment that has been fine tuned on a medical dataset(PubMedQA)
This is a professional `README.md` specifically tailored for your **Medical Reasoning** model. It maintains the technical depth of your previous one but pivots the focus to clinical context and PubMed data.

---

# Gemma 3 1B Medical Reasoning (GRPO) 🏥

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/tanicodesallday/gemma3-1b-medical-grpo)
[![PubMedQA Dataset](https://img.shields.io/badge/%F0%9F%8F%A5%20PubMedQA-Dataset-red)](https://huggingface.co/datasets/bigbio/pubmed_qa)

## 📌 Problem Statement
Medical diagnosis and clinical decision-making require more than just factual knowledge; they require **evidence-based reasoning**. While large models excel at this, Small Language Models (SLMs) often provide "hallucinated" or direct answers without explaining the underlying clinical logic.

This project uses **Group Relative Policy Optimization (GRPO)** to fine-tune **Gemma 3 1B** on the **PubMedQA** dataset. The goal was to transform the model from a simple QA bot into a medical reasoning agent that extracts evidence from provided contexts and follows a structured "Chain-of-Thought" before arriving at a "yes/no/maybe" conclusion.

---

## 🧪 Methodologies

### 1. Group Relative Policy Optimization (GRPO)
I implemented GRPO, a reinforcement learning algorithm that optimizes the policy by comparing a group of outputs against each other, rather than using a separate reward-predicting model.

**The GRPO Loss Function:**

<img width="875" height="99" alt="image" src="https://github.com/user-attachments/assets/b9944620-9d47-44c5-9f03-29c6585ebeca" />


### 2. Medical Reward Functions
To ensure clinical accuracy and structural integrity, I used three primary reward pillars:
*   **Clinical Correctness:** High rewards for matching the expert-labeled "Long Answer" conclusion in PubMedQA.
*   **Contextual Grounding:** Rewards for referencing specific data points found in the provided medical context.
*   **Structural XML Reward:** Enforcing the `<thought>` (reasoning trace) and `<answer>` (final verdict) format.

### 3. Efficiency via Unsloth
By using **Unsloth**, I utilized 4-bit quantization and specialized Triton kernels to make high-memory Reinforcement Learning possible on a single consumer-grade GPU.

---

## ⚙️ Experiments & Hardware
Training was conducted on an **NVIDIA Tesla T4 (16GB)**. 

| Parameter | Value |
| :--- | :--- |
| **Model** | Gemma 3 1B Instruct |
| **Dataset** | PubMedQA (pqa_labeled) |
| **Compute Dtype** | `float32` (Ensures Gemma 3 stability on T4 hardware) |
| **Group Size (G)** | 4 |
| **Max Seq Length** | 1024 tokens |

**Technical Challenge:** Since the Tesla T4 does not support `bfloat16`, I manually set the compute dtype to `float32` and disabled vLLM fast inference to prevent "Numerical Instability" (which causes the model to output empty strings or NaN losses).

---

## 📊 Results

The model shifted from providing brief answers to conducting a full "Differential Diagnosis" style reasoning trace.

### Side-by-Side Comparison
**Question:** Do aggressive prompt-gamma-ray checks improve survival in stage III lung cancer?

| Feature | Base Gemma 3 1B | Fine-Tuned (Medical GRPO) |
| :--- | :--- | :--- |
| **Response Type** | Direct Answer | Evidence-based Reasoning |
| **Verification** | Unstructured | Cites provided context in `<thought>` |
| **Verdict** | "Yes/No" | "Yes/No/Maybe" with logic |

### Evaluation Data
We compare the performance of the base model and the GRPO fine-tuned model across 50 evaluation cases.



---

## 💡 What I Learned

1.  **Reasoning vs. Memorization:** In the previous Math project, the model learned logic. In this Medical project, I learned that **context is king**. The model had to learn to ignore its internal "imagination" and only reason based on the medical context provided in the prompt.
2.  **Handling Unbalanced Rewards:** Medical data is nuanced. I learned that if the "Format Reward" is too high, the model writes a beautiful `<thought>` block that is factually wrong. Balancing "Thinking" vs "Correctness" is the hardest part of RL.
3.  **Cross-Domain RL:** Applying the same GRPO logic from Math (GSM8K) to Medicine (PubMedQA) proved that **Reasoning is a generalized skill**. Once a model learns to use `<thought>` tags effectively, it can be pivoted to almost any analytical domain.

---

## 🛠️ How to Use
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("tanicodesallday/gemma3-1b-medical-grpo")
FastLanguageModel.for_inference(model)

# Example Medical Prompt
context = "In a study of 500 patients with X-condition..."
question = "Does treatment Y improve outcomes?"
prompt = f"Context: {context}\nQuestion: {question}"

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

---
*Developed as a research project into SLM alignment for the healthcare domain.*
