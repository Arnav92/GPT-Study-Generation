import os
import shutil
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# NOTE: Change num_train_epochs to a small value like 3 for increased speed during training
# (results will be much worse though; only do to see an example of the result).

# Delete old folder so we start from scratch
out_dir = "./gpt2-math-generator"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

# math dataset
math_dataset = [
    {"text": "Q: What is 2 + 2? A: 4."},
    {"text": "Q: Solve for x if 2x = 10. A: x = 5."},
    {"text": "Q: Evaluate the derivative of x^2. A: 2x."},
    {"text": "Q: What is the integral of 3x^2 dx? A: x^3 + C."},
    {"text": "Q: Simplify the expression (x+2)(x+3). A: x^2 + 5x + 6."},
    {"text": "Q: If f(x) = x^2, what is f(3)? A: 9."},
    {"text": "Q: What are the solutions to x^2 - 1 = 0? A: x = ±1."},
    {"text": "Q: Compute 7 * 8. A: 56."},
    {"text": "Q: Differentiate sin(x). A: cos(x)."},
    {"text": "Q: Factor 4x^2 - 4x + 1. A: (2x - 1)^2."},
    {"text": "Q: What is the limit of (1/x) as x approaches infinity? A: 0."},
    {"text": "Q: Solve 2x + 5 = 13 for x. A: x = 4."},
    {"text": "Q: Convert 0.25 to a fraction. A: 1/4."},
    {"text": "Q: Solve for x: x/2 = 6. A: x = 12."},
    {"text": "Q: Simplify the fraction 8/24. A: 1/3."},
    {"text": "Q: What is the square root of 81? A: 9."},
    {"text": "Q: If g(x) = x + 2, what is g(5)? A: 7."},
    {"text": "Q: Evaluate (3 + 4)^2. A: 49."},
    {"text": "Q: Solve for y if y + 10 = 15. A: y = 5."},
    {"text": "Q: What is 10 mod 3? A: 1."}
]

dataset = Dataset.from_list(math_dataset)

# Load the GPT-2 model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenize
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=64)

tokenized_dataset = dataset.map(tokenize_function, batched=False)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # causal LM
)

# Training arguments
training_args = TrainingArguments(
    output_dir=out_dir,
    overwrite_output_dir=True,
    num_train_epochs=20,       # Decrease for increased speed but decreased memory
    per_device_train_batch_size=2,
    save_steps=50,
    logging_steps=10,
    eval_strategy="no"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)
trainer.train()

# Save
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)

# Inference function
def generate_math_question(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        model.cuda()
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Keep it short to discourage multiple enumerations
    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=1,
        num_beams=1,
        do_sample=False,
        no_repeat_ngram_size=1,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Test a question
test_prompt = "Q: What is 2 + 2? A:"
answer = generate_math_question(test_prompt)
print("Prompt:", test_prompt)
print("Model Output:", answer)