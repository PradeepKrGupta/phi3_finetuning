from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True, # Set to True to use 8-bit precision
    torch_dtype=torch.float16 # Use float16 for model precision
)

model = AutoModelForCausalLM.from_pretrained(model_name, config=bnb_config)