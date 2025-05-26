import json
def printTranslationsToFile(fileName, translations):

    file = open(fileName, "w", encoding="utf-8")


    for i, (shaclTranslation, nlTranslation) in enumerate(translations):
        file.write(f"Translation #{i}\n")
        file.write(shaclTranslation)
        file.write("\n")
        file.write(nlTranslation)
        file.write("\n----\n\n")


def printTranslationsToJSONFile(filename, translations):
    file = open(filename, "w")
    for (shacl, nl) in translations:
            json_line = json.dumps({"prompt": nl, "response": shacl})
            file.write(json_line + "\n")
