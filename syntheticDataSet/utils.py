import json

def printTranslationsToFile(file, shaclTranslation, nlTranslation, i):
            file.write(f"Translation #{i}\n")
            file.write(shaclTranslation)
            file.write("\n")
            file.write(nlTranslation)
            file.write("\n----\n\n")

def printTranslationsToJSONFile(file, shacl, nl, i):
            json_line = json.dumps({"prompt": nl, "response": shacl})
            file.write(json_line + "\n")

def load_translations_from_json(filename):
    prompts = []
    responses = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            prompts.append(item["prompt"])
            responses.append(item["response"])
    return prompts, responses
