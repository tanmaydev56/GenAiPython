from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", pad_token_id=tokenizer.eos_token_id
)

# Input prompt
input_text = "You are going to"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs['input_ids']

# Display Input IDs
print("Input IDs:", input_ids)

# Get model output for next-token prediction
with torch.no_grad():
    output = model(**inputs)

# Get logits of the last token in the sequence
final_logits = output.logits[0, -1]

# Get predicted next token (argmax)
predicted_token_id = final_logits.argmax().item()
predicted_token = tokenizer.decode(predicted_token_id)
print(f"\nPredicted next token (argmax): '{predicted_token}'\n")

# Get top 10 predicted tokens and probabilities
top_k = 10
top_logits, top_indices = torch.topk(final_logits, top_k)
probs = torch.nn.functional.softmax(top_logits, dim=0)

print("Top 10 predicted tokens:")
for i in range(top_k):
    token_id = top_indices[i].item()
    token = tokenizer.decode(token_id)
    prob = probs[i].item()
    print(f"{i+1}. Token: '{token}' | Probability: {prob:.4f}")

# Generate continuation text
output_ids = model.generate(
    input_ids=input_ids,
    max_length=50,
    do_sample=True,
    top_k=10,
    pad_token_id=tokenizer.eos_token_id
)

# Display Output IDs and generated text
print("\nOutput IDs:", output_ids)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated text:", generated_text)



# Generate token-by-token
generated_ids = input_ids.clone()

print("Step-by-step generation:\n")
for i in range(10):  # generate 10 tokens, one at a time
    with torch.no_grad():
        outputs = model(input_ids=generated_ids)
    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

    # Append the new token to the sequence
    generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    # Decode and print the current result
    current_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"[{i+1}] {current_text}")


tokens = [tokenizer.decode([idx]) for idx in top_indices]
probs = probs.detach().numpy()

# Plot
plt.figure(figsize=(10, 5))
bars = plt.bar(tokens, probs, color='skyblue')
plt.title("Top-10 Token Predictions")
plt.xlabel("Tokens")
plt.ylabel("Probability")
plt.xticks(rotation=45)
for bar, prob in zip(bars, probs):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{prob:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# Define tokens you want to penalize or boost
forbidden_words = ["gonna", "ain't"]
bias_tokens = [tokenizer.encode(w, add_special_tokens=False)[0] for w in forbidden_words]

with torch.no_grad():
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1]

# Apply logit bias (e.g., reduce probability of informal words)
logits[bias_tokens] -= 5.0  # strong penalty

# Sample top-k token
top_k = 10
top_logits, top_indices = torch.topk(logits, top_k)
probs = torch.nn.functional.softmax(top_logits, dim=0)

# Show top-k after bias
print("Top-10 tokens after logit bias:")
for i in range(top_k):
    token_id = top_indices[i].item()
    token = tokenizer.decode([token_id])
    prob = probs[i].item()
    print(f"{i+1}. Token: '{token}' | Probability: {prob:.4f}")

