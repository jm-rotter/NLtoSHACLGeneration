from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from rag import rag


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    max_seq_length = 2048,   
    load_in_4bit = True,    
    load_in_8bit = False,   
    full_finetuning = False,
    offload_buffers=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8,           
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 8, 
    lora_dropout = 0,
    bias = "none", 
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,   
    loftq_config = None, 
)

shacl_dataset = load_dataset("json", data_files="training_translations.jsonl", split="train")

def convert_to_conversation(examples):
    conversations = []
    for prompt, response in zip(examples["prompt"], examples["response"]):
        rag_prefixes = rag(prompt)
        conversations.append([
            {"role": "user", "content": """

             You are a SHACL expert. 

             Translate the following natural language to shacl using the provided prefixes as help. 

             Consider the following few shot examples to help you. 


NL Input
The SHACL shape :LaengeBelegnummer targets all instances of the class edifact-o:InvoiceDetails. It enforces a constraint on the property edifact-o:hasDocumentNumber, requiring its value to be a string (xsd:string) with a maximum length of 12 characters. If this constraint is violated, the validation will produce the message:
"The data element 1004 in the BGM segment is too long."


SHACL Output
:LaengeBelegnummer 
    a sh:NodeShape;
    sh:targetClass edifact-o:InvoiceDetails;
    sh:property [
        sh:path edifact-o:hasDocumentNumber;
        sh:datatype xsd:string;
        sh:maxLength 12;
        sh:message "The data element 1004 in the BGM segment is too long";
    ]
.


NL input
The SHACL shape :Dokumentfunktion applies to all instances of the class edifact-o:InvoiceDetails. It specifies a property constraint on edifact-o:hasDocumentFunction that requires exactly one value (minCount and maxCount set to 1). The value must be a string (xsd:string) and must be one of the predefined allowed values:
"Cancellation", "Replacement", "Duplicate", "Original", "Copy", "Additional transfer", "Stornierung", "Ersatz", "Duplikat", "Original", "Kopie", "Zusaetzliche Uebertragung."
If this property is missing or its value is outside this list, validation will fail with the message:
"Data element 1225 is missing in the BGM segment, i.e. the specification of the document function."

             SHACL output
:Dokumentfunktion 
    a sh:NodeShape; 
    sh:targetClass edifact-o:InvoiceDetails; 
    sh:property [
        sh:path edifact-o:hasDocumentFunction;
        sh:datatype xsd:string;
        sh:minCount 1;
        sh:maxCount 1;
        sh:in ("Cancellation" "Replacement" "Duplicate" "Original" "Copy" "Additional transfer" "Stornierung" "Ersatz" "Duplikat" "Original" "Kopie" "Zusaetzliche Uebertragung");
        sh:message "Data element 1225 is missing in the BGM segment, i.e. the specification of the document function";
    ]
.


             NL input:
The SHACL shape :UmsatzsteuernummerKaeufer applies to all instances of the class org:FormalOrganization. It specifies a validation rule with an sh:or condition. Either the organization does not have the RDF type "http://example.com/BuyerRole" or, if it does have the "http://example.com/BuyerRole" type, then the property p2p-o-org:VATIdentifier must be present exactly once (minimum count 1 and maximum count 1), and its value must be a string with a maximum length of 14 characters.
If neither of these conditions is met (i.e., the organization is a buyer but the VAT identifier is missing or invalid), the validation will fail with the message: "The segment RFF+VA is missing for NAD+BY."
        
            SHACL output:

:UmsatzsteuernummerKaeufer 
    a sh:NodeShape;
    sh:targetClass org:FormalOrganization;
    sh:message "The segment RFF+VA is missing for NAD+BY";
    sh:or (
        [ sh:not [
            a sh:PropertyShape;
            sh:path rdf:type;
            sh:hasValue "http://example.com/BuyerRole";
        ] ]
        [
            sh:path p2p-o-org:VATIdentifier;
            sh:minCount 1;
            sh:maxCount 1;
            sh:maxLength 14;
        ]
    ) 
.



NL input
The SHACL shape :edifact-oEntity50Shape applies to all instances of the class edifact-o:Entity50. It specifies a property constraint on edifact-o:invoiceDate that requires exactly one value (with minCount and maxCount set to 1). The value must be a date (xsd:date).
If this property is missing or appears more than once, validation will fail with the message:
"The value must appear exactly once."

SHACL output

:edifact-oEntity50Shape a NodeShape;
    targetClass edifact-o:Entity50;
    property [
        path edifact-o:invoiceDate;
        datatype xsd:date;
        minCount 1;
        maxCount 1;
        message "The value must appear exactly once.";
    ] .


             Translate the following NL to shacl using the provided prefixes and few shot examples as help. Do not do any reasoning, just output the translated text.\n""" + prompt +"\nPrefixes: " + rag_prefixes},

            {"role": "assistant", "content": response},
        ])
    return {"conversations": conversations}


conversations_dataset = shacl_dataset.map(convert_to_conversation, batched=True)

shacl_conversations = tokenizer.apply_chat_template(
    conversations_dataset["conversations"],
    tokenize=False,
)

shacl_data = pd.Series(shacl_conversations)
shacl_data.name = "text"

final_dataset = Dataset.from_pandas(pd.DataFrame(shacl_data))
final_dataset = final_dataset.shuffle(seed=3407)


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=final_dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        num_train_epochs=30,
        learning_rate=1e-5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    ),
)


print("starting training")

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()


used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


model.save_pretrained("lora_weights")
