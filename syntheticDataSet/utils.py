def printTranslationsToFile(fileName, translations):

    file = open(fileName, "w", encoding="utf-8")


    for i, (shaclTranslation, nlTranslation) in enumerate(translations):
        file.write(f"Translation #{i}\n")
        file.write(shaclTranslation)
        file.write("\n")
        file.write(nlTranslation)
        file.write("\n----\n\n")
