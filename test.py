import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "./gpt2-math-generator"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

def generate_math_question(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    if torch.cuda.is_available():
        model.cuda()
        inputs = {k: v.cuda() for k,v in inputs.items()}

    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=20,
        num_beams=5,
        do_sample=False,
        no_repeat_ngram_size=5,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Test here:
prompt_text = "Q: Evaluate the derivative of x^3. A:"
response = generate_math_question(prompt_text)
print("==== Prompt ====")
print(prompt_text)
print("==== Model Response ====")
print(response)