import json

from analyser import Analyser


class MyDataAnalyser(Analyser):
    def __init__(self):
        super(MyDataAnalyser, self).__init__()

    def analyse_annotations(self, annotations_file, corpus_file, output_file):
        analysis = {"intents": {}, "entities": {}}

        corpus = json.load(open(corpus_file))
        gold_standard = []
        for s in corpus["sentences"]:
            if not s["training"]:  # only use test data
                gold_standard.append(s)

        annotations = json.load(open(annotations_file))

        i = 0
        for a in annotations["results"]:
            print(a)
            if not a["queryText"] == gold_standard[i]["text"]:
                print("WARNING! Texts not equal")

            # intent
            try:
                aIntent = a["intent"]
            except:
                aIntent = "None"
            oIntent = gold_standard[i]["intent"]

            Analyser.check_key(analysis["intents"], aIntent)
            Analyser.check_key(analysis["intents"], oIntent)

            if aIntent == oIntent:
                # correct
                analysis["intents"][aIntent]["truePos"] += 1
            else:
                # incorrect
                analysis["intents"][aIntent]["falsePos"] += 1
                analysis["intents"][oIntent]["falseNeg"] += 1

            # entities
            try:
                aEntities = a["queryResult"]["parameters"]
            except:
                aEntities = {}
            oEntities = gold_standard[i]["entities"]

            for x in aEntities.keys():
                Analyser.check_key(analysis["entities"], x)

                if len(oEntities) < 1:  # false pos
                    analysis["entities"][x]["falsePos"] += 1
                else:
                    truePos = False

                    for y in oEntities:
                        if len(aEntities[x]) != 0 and aEntities[x][0].lower() == y["text"].lower():
                            if x == y["entity"]:  # truePos
                                truePos = True
                                oEntities.remove(y)
                                break
                            else:  # falsePos + falseNeg
                                analysis["entities"][x]["falsePos"] += 1
                                analysis["entities"][y["entity"]]["falseNeg"] += 1
                                oEntities.remove(y)
                                break
                    if truePos:
                        analysis["entities"][x]["truePos"] += 1
                    else:
                        analysis["entities"][x]["falsePos"] += 1

            for y in oEntities:
                Analyser.check_key(analysis["entities"], y["entity"])
                analysis["entities"][y["entity"]]["falseNeg"] += 1

            i += 1

        self.write_json(output_file, analysis)


if __name__ == '__main__':
    # luis_analyser = MyDataAnalyser()
    # luis_analyser.analyse_annotations("../../../sentence_similarity/output_json/test_annotation.json",
    # 								  "../../../sentence_similarity/output_json/test.json",
    # 								  "../CosineAnalysis_my.json")

    luis_analyser = MyDataAnalyser()
    luis_analyser.analyse_annotations("../../../sentence_similarity/output_json/test-log-regression_annotation.json",
                                      "../../../sentence_similarity/output_json/test-log-regression.json",
                                      "../LogRegressionAnalysis_my.json")
