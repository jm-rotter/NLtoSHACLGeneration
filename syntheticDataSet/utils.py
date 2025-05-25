def printTranslationsToFile(fileName, translations):

    file = open(fileName, "w", encoding="utf-8")


    for i, (shaclTranslation, nlTranslation) in enumerate(translations):
        file.write(f"Translation #{i}\n")
        file.write(shaclTranslation)
        file.write("\n")
        file.write(nlTranslation)
        file.write("\n----\n\n")


def printTranslationToJSONFile(filename, translations):
    file = open(fileName, "w"):
        for (shacl, nl) in translations:
             prompt = f"### Instruction:\nTranslate this natural language sentence to SHACL:\n\"{nl}\"\n### Response:"
             response = shacl
             json_line = json.dumps({"prompt": prompt, "response": response})
             f.write(json_line + "\n")
