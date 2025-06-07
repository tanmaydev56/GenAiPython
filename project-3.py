from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt

# Load GPT-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# Input text
input_text = "You are going to"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs['input_ids']
print(" Input IDs:", input_ids)

# ------------------ 1. Top-K Token Prediction (with optional logit bias) ------------------

# Get model logits
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits[0, -1]

# Optional: Penalize informal tokens
forbidden_words = ["gonna", "ain't"]
bias_tokens = [tokenizer.encode(w, add_special_tokens=False)[0] for w in forbidden_words]
logits[bias_tokens] -= 5.0  # Apply penalty

# Get Top-10 predictions after bias
top_k = 10
top_logits, top_indices = torch.topk(logits, top_k)
probs = torch.nn.functional.softmax(top_logits, dim=0)

# Display predictions
print("\n Top-10 predicted tokens (after bias):")
for i in range(top_k):
    token_id = top_indices[i].item()
    token = tokenizer.decode([token_id])
    prob = probs[i].item()
    print(f"{i+1}. Token: '{token}' | Probability: {prob:.4f}")

# ------------------ 2. Step-by-Step Token Generation ------------------

generated_ids = input_ids.clone()
print("\n Step-by-step token generation:")
for i in range(10):  # Generate 10 tokens one at a time
    with torch.no_grad():
        outputs = model(input_ids=generated_ids)
    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
    generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
    current_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"[{i+1}] {current_text}")

# Print Final Output IDs
print("\n Output IDs:", generated_ids)

# ------------------ 3. Plot Top-K Probabilities ------------------

tokens = [tokenizer.decode([idx]) for idx in top_indices]
probs = probs.detach().numpy()

plt.figure(figsize=(10, 5))
bars = plt.bar(tokens, probs, color='skyblue')
plt.title("Top-10 Token Predictions (After Bias)")
plt.xlabel("Tokens")
plt.ylabel("Probability")
plt.xticks(rotation=45)
for bar, prob in zip(bars, probs):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{prob:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
