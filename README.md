# CRM Instruction Fine-Tuning

Fine-tune code generation models (StarCoder, StarCoder2, or CodeLlama) for CRM-related instruction datasets using LoRA for efficient training on consumer GPUs.

---

## 1. Setup
```bash
git clone <repo_url>
cd <repo_name>
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Dataset Format
Dataset must be JSONL with `messages` list containing `user` and `assistant`.

### Example (`data/train_data_expanded.jsonl`)
```json
{"messages":[
  {"role":"user","content":"Generate CRM code for Healthcare Institution with intent handling."},
  {"role":"assistant","content":"# Auto-generated CRM code for HealthcareInstitution\n# Intent: Auto-generated CRM code for HealthcareInstitution\n# [CODE PLACEHOLDER for HealthcareInstitution CRM]\n# [CODE PLACEHOLDER for HealthcareInstitution Intent]"}
]}
```

Provide train & validation sets:
```
data/train_data_expanded.jsonl
data/val_data_expanded.jsonl
```

---

## 3. Model
Preferred models:
- `bigcode/starcoder`
- `bigcode/starcoder2-7b`
- `meta-llama/CodeLlama-7b-Instruct-hf`

Use any Hugging Face-compatible model or a locally downloaded model folder.

---

## 4. Fine-Tuning
```bash
python fine_tune.py
```
- Detects GPU automatically (uses CPU if unavailable)  
- Saves model to `finetuned_codellama_lora/`

---

## 5. Inference Example
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("finetuned_codellama_lora")
tokenizer = AutoTokenizer.from_pretrained("finetuned_codellama_lora")
input_text = "Generate CRM code for a Retail Store"
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Notes
- Training on GPU is recommended.  
- For 6–8 GB GPUs, use smaller models (1.3B) or 4-bit quantization.  
- Data is flexible; you can use any instructions relevant to CRM.

---
