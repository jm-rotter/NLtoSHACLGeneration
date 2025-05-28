from unsloth import FastLanguageModel
import torch
from safetensors.torch import load_file

model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
    offload_buffers=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

lora_weights = load_file("lora_weights/adapter_model.safetensors", device="cpu")
missing, unexpected = model.load_state_dict(lora_weights, strict=False)
#print("Missing keys:", missing)
#print("Unexpected keys:", unexpected)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

prompt = (
    "Translate the following English sentence to SHACL: "
    "'For instances of edifact-o:E-Invoice, the edifact-o:belongsToProcess property must "
    "have exactly one value, which must be \"ProcessExample\".'"
)

inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:", generated_text)
print("\n")

