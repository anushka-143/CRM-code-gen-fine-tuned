import os
import shutil
import zipfile
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from fine_tune import MODEL_DIR, fine_tune

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return model, tokenizer


def generate_code(model, tokenizer, prompt, max_length=2048, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt) :].strip()


def save_code_to_file(output_text, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "generated_crm.py")
    with open(file_path, "w") as f:
        f.write(output_text)
    return file_path


def zip_output(file_path, zip_name):
    zip_path = os.path.join(OUTPUT_DIR, zip_name)
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(file_path, arcname=os.path.basename(file_path))
    return zip_path


def main():
    st.title("AI-powered CRM Code Generator")

    prompt = st.text_area("Enter your CRM generation prompt:", "Generate a CRM for Healthcare")
    max_length = st.slider("Max generation length", 256, 4096, 2048, 256)
    temperature = st.slider("Sampling temperature", 0.1, 1.0, 0.7, 0.05)

    if st.button("Generate Code"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        if not os.path.isdir(MODEL_DIR) or not os.listdir(MODEL_DIR):
            with st.spinner("Fine-tuned model not found. Starting fine-tuning... This may take a while."):
                try:
                    fine_tune()
                except Exception as e:
                    st.error(f"Fine-tuning failed: {e}")
                    return

        with st.spinner("Generating CRM code..."):
            try:
                model, tokenizer = load_model_and_tokenizer()
                output_text = generate_code(model, tokenizer, prompt, max_length, temperature)

                output_file = save_code_to_file(output_text, os.path.join(OUTPUT_DIR, "latest"))
                zip_path = zip_output(output_file, "generated_crm_code.zip")

                st.success("Code generated successfully!")
                st.code(output_text, language="python")

                with open(zip_path, "rb") as f:
                    st.download_button("Download Generated Code ZIP", f, file_name="generated_crm_code.zip")

            except Exception as e:
                st.error(f"Code generation failed: {e}")


if __name__ == "__main__":
    main()
