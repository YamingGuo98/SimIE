from transformers import AutoModelForCausalLM, AutoTokenizer

true_dir = "."
# true_dir = "/data/anonymous/simIE"

# Download and load the Llama-2-7B model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# Save the Llama-2-7B model to the local
down_dir = true_dir + "/hugging_cache/llama-2-7b"
model.save_pretrained(down_dir)
tokenizer.save_pretrained(down_dir)

# Download and load the Mistral-7B model and tokenizer
mistral_model_name = "mistralai/Mistral-7B-v0.1"
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)
mistral_model = AutoModelForCausalLM.from_pretrained(mistral_model_name)
# Save the Mistral-7B model to the local
down_dir = true_dir + "/hugging_cache/Mistral-7B-v0.1"
mistral_model.save_pretrained(down_dir)
mistral_tokenizer.save_pretrained(down_dir)

# Download and load the GPT2-XL model and tokenizer
gpt2xl_model_name = "openai-community/gpt2-xl"
gpt2xl_tokenizer = AutoTokenizer.from_pretrained(gpt2xl_model_name)
gpt2xl_model = AutoModelForCausalLM.from_pretrained(gpt2xl_model_name)
# Save the GPT2-XL model to the local
down_dir = true_dir + "/hugging_cache/gpt2-xl"
gpt2xl_model.save_pretrained(down_dir)
gpt2xl_tokenizer.save_pretrained(down_dir)