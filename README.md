# NLtoSHACLGeneration: Using LLMs for Structured Constraint Generation

Bachelor Practical in Data Engineering to convert natural language (NL) to shapes constraint language (SHACL) shapes, or constraints.

This project aims to simplify the human task of generating complex structured constraints from real life documentation.
This paper seeks to explain the codebase, for understanding of our project, please see our paper for more information: https://github.com/jm-rotter/NLtoSHACLGeneration/blob/main/paper.pdf


We have logically partitioned our code into three different submodules: synthetic dataset, fine-tuning and evaluation. 

Note: The fine-tuned model weights were too large for github, so for access to them please email me at jaden.rotter@tum.de or use the datasets for your own training on LLMs. 

### Synthetic Dataset

Step one of our goal to translate NL -> SHACL. 

We constructed a pipeline to generate synthetic data, using both a SHACL shape generator and a NL translations from the newly generated SHACL shapes.
The purpose of this dataset was to fine tune the LLM. 
#### Running the 1 step. 

The model chosen uses the free LLaMA 70B model API provided by GROQ to translate the SHACL to NL, due to the free API.
OpenAI has larger models, but lacking a subscription, we resorted to a weaker, but free model.

To run the program, create a `.env` file in the root project and add your GROQ API key. Then install the dependencies in the root folder and run the main.py file.

```bash
$ touch .env
$ echo "GROQ_API_KEY=your_api_key_here" >> .env 
$ pip install -r synthDatarequirements.txt
$ python3 syntheticDataSet/main.py 
```


The `main.py` file in the directory `syntheticDataSet` does all the API calls to the model llama3-70b-8192 as specified by our presentation `shaclGenerationProject.pdf`.
This file uses the prompts defined in `syntheticDataset/prompts.py` and the parsing logic from `syntheticDataset/shaclParser.py` to output LLM generated NL translations. 
Additionally, the generated shapes are output to files, `syntheticDataset/shacltranslations.txt` and `syntheticDataset/shacltranslations.jsonl`, using the helper functions defined in `syntheticDatset/utils.py`


Our synthetic datasets are given as both .txt and .jsonl files for training and human readability in `syntheticDataset/training_translations.{txt,jsonl}` and `syntheticDataset/shacl_translations.{txt,jsonl}`.

The `training_translations` files contain our manually generated shapes and the `shacl_translations` contain the translations from the `https://github.com/DE-TUM/EDIFACT-VAL/blob/main/example/ProcessExample.ttl`. 
These shapes can also be found in `syntheticDataSet/shaclDataset.ttl`.

        
### Fine-Tuning

For fine-tuning run the programs with a similar process, we included a fine-tuning requirements.txt folder; however the installation of unsloth through pip didn't work due to dependency issues and we installed directly from the github repository.

Running the models depends on the model itself as well. I.e. mistral models require a certain input and output syntax, otherwise they don't generate anything.
For trying it out, we recommend the use of llama models as the fine-tuning does not require any chat-templates and the implemenation is the most simple and straightforward. 
Please contact jaden.rotter@tum.de for more details.

`fine-tuning\trainer.py` loads the model and does the fine-tuning using LoRA described in the paper and presentation and saves the trained adaption layers.
Unfortunately, due to size limitations in github the weights are not shared; however, on request please email jaden.rotter@tum.de.
`fine-tuning\inference.py` loads the weights previously created by trainer.py and then does the inference on the NL prompts generated by our initial synthetic data generation pipeline. 

We additionally added in the logs from our training, as well as our output both in the standardized form of `.txt` and `.jsonl`.

The {model}7b.log files are the actual training logs that unsloth output, which were captured by an unnamed pipe and the running the process with nohup, such that termination of the SSH session didn't result in the termination of the model training. 

Unfortunately, checking up on the models progress by reading the files generated many newlines in the files themselves. 

The {model}7b.{jsonl, txt} are the inferenced outputs in both human and machine parsable outputs. 


The `llama7B.log` was a trial run as we were still working on the llama model, which was only trained with around 800 shapes. 
The llama7B models outputs additionally are from this model with 800 shapes, which is why it was left out of the evaluation, instead seeking to standardize the number of different shapes. 


### Evaluation

