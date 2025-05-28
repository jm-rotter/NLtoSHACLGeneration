# NLtoSHACLGeneration
Bachelor Practical in Data Engineering to translate NL to SHACL shapes


### Synthetic Dataset

Step one of our goal to translate NL -> SHACL. 

#### Running the 1 step. 
To run the program, create a `.env` file in the root project, add your GROQ API key. Then install the dependencies in the root folder and run the main.py file.

```bash
$ touch .env
$ echo "GROQ_API_KEY=your_api_key_here" >> .env 
$ pip install -r synthDatarequirements.txt
$ python3 syntheticDataSet/main.py 
```


The `main.py` file in the directory `syntheticDataSet` does all the API calls to the model llama3-70b-8192 as specified by our presentation `shaclGenerationProject.pdf`.
This file uses the prompts defined in `syntheticDataset/prompts.py` and the parsing logic from `syntheticDataset/shaclParser.py` to output LLM generated NL translations. 
Additionally, the generated shapes are output to files, `synthteticDataset/shacltranslations.txt` and `syntheticDataset/shacltranslations.jsonl`, using the helper functions defined in `syntheticDatset/utils.py`


The dataset used for the synthetic data generation is provided by `https://github.com/DE-TUM/EDIFACT-VAL/blob/main/example/ProcessExample.ttl` and can also be found in `syntheticDataSet/shaclDataset.ttl`.

        
### Fine-Tuning

In progress

trainer.py loads the model and does the fine-tuning using LoRA described by the presentation mentioned above and saves the trained adaption layers to the file `lora_weights`. 
`lora_weights` is not in the repo due to size limitations.
inference.py loads the weights previously created by trainer.py and then does a basic inference of one of the 5 examples that was withheld in the split. 


shacltranslations.jsonl is the first 84 (out of 89) translations from our synthetic dataset. 
Last 5 are our test split (for right now).


Currently the program computes 90.95 epochs? idk? and the output from that run is in `output.txt` along with the corresponding weights.

