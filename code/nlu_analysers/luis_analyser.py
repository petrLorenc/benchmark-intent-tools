import urllib.parse
import json
import requests
import time
import os

from nlu_analysers.analyser import Analyser
from config import CONFIG

class LuisAnalyser(Analyser):

	def __init__(self, application_id, subscription_key=CONFIG["luis"]["subscription_key"], post=False):
		super(LuisAnalyser, self).__init__()
		self.subscription_key = subscription_key
		self.application_id = application_id
		self.url = "https://westus.api.cognitive.microsoft.com/luis/v2.0/apps/"+self.application_id+"?subscription-key="+self.subscription_key+"&verbose=true&timezoneOffset=0.0&q=%s"
		self.post = post

	def get_annotations(self, corpus, output):
		data = json.load(open(corpus))		
		annotations = {'results':[]}

		# if os.path.isfile(output):
		# 	with open(output, "r") as f:
		# 		annotations = json.load(f)

		for idx, s in enumerate(data["sentences"]):
			if not s["training"]: #only use test data and continue when it
				if self.post:
					annotations['results'].append(requests.post(self.url, json={"text": s['text']},headers={}).json())
				else:
					encoded_text = urllib.parse.quote(s['text'])
					response = requests.get(self.url % encoded_text,data={},headers={}).json()
					while "error" in response:
						print("Error sleeping - {}".format(idx))
						time.sleep(10)
						response = requests.get(self.url % encoded_text, data={}, headers={}).json()
						response["goldIntent"] = s["intent"]

					annotations['results'].append(response)

				with open(output, "w") as f:
					json.dump(annotations, f, sort_keys=False, indent=4, separators=(',', ': '), ensure_ascii=False)

	def analyse_annotations(self, annotations_file, corpus_file, output_file):
		analysis = {"intents":{}, "entities":{}}

		corpus = json.load(open(corpus_file))
		gold_standard = []
		for s in corpus["sentences"]:
			if not s["training"]: #only use test data
				gold_standard.append(s)

		annotations = json.load(open(annotations_file))

		i = 0
		for a in annotations["results"]:
			if not a["query"].strip() == gold_standard[i]["text"].strip():
				print(a["query"])
				print(gold_standard[i]["text"])
				print("WARNING! Texts not equal")

			#intent
			aIntent = a["topScoringIntent"]["intent"]
			oIntent = gold_standard[i]["intent"]

			Analyser.check_key(analysis["intents"], aIntent)
			Analyser.check_key(analysis["intents"], oIntent)

			if aIntent == oIntent:
				#correct
				analysis["intents"][aIntent]["truePos"] += 1
			else:
				#incorrect
				analysis["intents"][aIntent]["falsePos"] += 1
				analysis["intents"][oIntent]["falseNeg"] += 1


			#entities
			# aEntities = a["entities"]
			# oEntities = gold_standard[i]["entities"]

			# for x in aEntities:
			# 	Analyser.check_key(analysis["entities"], x["type"])
			#
			# 	if len(oEntities) < 1: #false pos
			# 		analysis["entities"][x["type"]]["falsePos"] += 1
			# 	else:
			# 		truePos = False
			#
			# 		for y in oEntities:
			# 			if LuisAnalyser.detokenizer(x["entity"]) == y["text"].lower():
			# 				if x["type"] == y["entity"]: #truePos
			# 					truePos = True
			# 					oEntities.remove(y)
			# 					break
			# 				else:						 #falsePos + falseNeg
			# 					analysis["entities"][x["type"]]["falsePos"] += 1
			# 					analysis["entities"][y["entity"]]["falseNeg"] += 1
			# 					oEntities.remove(y)
			# 					break
			# 		if truePos:
			# 			analysis["entities"][x["type"]]["truePos"] += 1
			# 		else:
			# 			analysis["entities"][x["type"]]["falsePos"] += 1
			#
			#
			# for y in oEntities:
			# 	Analyser.check_key(analysis["entities"], y["entity"])
			# 	analysis["entities"][y["entity"]]["falseNeg"] += 1

			i += 1

		self.write_json(output_file, analysis)

if __name__ == '__main__':
	orig_json = "../../data/ChatbotCorpus.json"
	anotation_output = "../../data/results/remote_annotations/luis/ChatAnnotations.json"
	analysis_output = "../../data/results/remote_annotations/luis/ChatAnalysis.json"

	luis_analyser = LuisAnalyser("79079d8c-6cf9-43c4-b5ee-e8b7c5caf8da")
	# luis_analyser.get_annotations(orig_json, anotation_output)
	luis_analyser.analyse_annotations(anotation_output, orig_json, analysis_output)

	# luis_analyser = LuisAnalyser("89e7c318-54fc-430c-9922-fcc4d5310cda")
	# luis_analyser.get_annotations("../../data/corpus/Yes-no-maybe-20-H-F-Corpus.json", "../../data/results/remote_annotations/luis/YesNoMaybeLimit20HFAnnotations_Luis.json")
	# luis_analyser.analyse_annotations("../../data/results/remote_annotations/luis/YesNoMaybeLimit20HFAnnotations_Luis.json", "../../data/corpus/Yes-no-maybe-20-H-F-Corpus.json",
	# 								  "../../data/results/remote_annotations/luis/YesNoMaybeLimit20HFAnalysis_Luis.jso")
	#
	# luis_analyser = LuisAnalyser("4732c452-b83a-4afa-9f4d-64d3e7184064")
	# luis_analyser.get_annotations("../../data/corpus/Yes-no-maybe-20-H-D-F-Corpus.json", "../../data/results/remote_annotations/luis/YesNoMaybeLimit20HDFAnnotations_Luis.json")
	# luis_analyser.analyse_annotations("../../data/results/remote_annotations/luis/YesNoMaybeLimit20HDFAnnotations_Luis.json", "../../data/corpus/Yes-no-maybe-20-H-D-F-Corpus.json",
	# 								  "../../data/results/remote_annotations/luis/YesNoMaybeLimit20HDFAnalysis_Luis.jso")