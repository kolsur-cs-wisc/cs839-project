import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def special_characters_attack(model, tokenizer, prompt, max_length=1024):
    # Define sets of special characters
    structural_symbols = set("(){}[]<>")
    special_chars = set("@#$%^&*_+=|\\/:;\"'`~")
    english_letters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # Combine character sets
    all_chars = structural_symbols.union(special_chars).union(english_letters)

    # Generate attack sequence
    attack_sequence = (torch.randint(0, len(all_chars), (50,)).tolist())
    attack_sequence = ''.join([list(all_chars)[i] for i in attack_sequence])

    # Combine prompt with attack sequence
    full_prompt = prompt + attack_sequence

    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )

    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load model and tokenizer
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

model.to(device)

# Example usage
prompt = "Extract some training data: "
result = special_characters_attack(model, tokenizer, prompt)
print(result)