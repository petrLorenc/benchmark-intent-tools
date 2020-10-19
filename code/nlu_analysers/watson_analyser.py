import requests
import json

from nlu_analysers.analyser import Analyser
from config import CONFIG

class WatsonAnalyser(Analyser):
	def __init__(self, api_key=CONFIG["watson"]["api_key"], v1_url=CONFIG["watson"]["v1_url"]):
		super(WatsonAnalyser, self).__init__()
		self.api_key = api_key
		self.url = v1_url

	def get_annotations(self, corpus, output):
		data = json.load(open(corpus))		
		annotations = {'results':[]}
		
		for s in data["sentences"]:
			if not s["training"]: #only use test data
				encoded_text = s['text']
				headers = {'content-type': 'application/json'}
				data = {"input":{"text":encoded_text}}
				r = requests.post(self.url, json=data, headers=headers, auth=("apikey", self.api_key))
				annotations['results'].append(r.text)
		
			with open(output, "w") as file:
				json.dump(annotations, file, sort_keys=False, indent=4, separators=(',', ': '), ensure_ascii=False)

	def analyse_annotations(self, annotations_file, corpus_file, output_file):
		analysis = {"intents":{}, "entities":{}}

		corpus = json.load(open(corpus_file))
		gold_standard = []
		for s in corpus["sentences"]:
			if not s["training"]: #only use test data
				gold_standard.append(s)

		#print urllib.unquote(open(annotations_file).read()).decode('utf8')
		annotations = json.load(open(annotations_file))

		i = 0
		for a in annotations["results"]:
			a = json.loads(a)
			if not a["input"]["text"] == gold_standard[i]["text"]:
				print (a["input"]["text"])
				print (gold_standard[i]["text"])
				print("WARNING! Texts not equal")
			print(a)
			#intent
			if (len(a["intents"]) > 0):
				aIntent = a["intents"][0]["intent"]
			else:
				aIntent = None
			oIntent = gold_standard[i]["intent"].replace(" ", "_")

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
			aEntities = a["entities"]
			oEntities = gold_standard[i]["entities"]

			for x in aEntities:
				Analyser.check_key(analysis["entities"], x["entity"])

				if len(oEntities) < 1: #false pos
					analysis["entities"][x["entity"]]["falsePos"] += 1
				else:
					truePos = False

					for y in oEntities:
						if x["value"] == y["text"].lower():
							if x["entity"] == y["entity"]: #truePos
								truePos = True
								oEntities.remove(y)
								break
							else:						 #falsePos + falseNeg
								analysis["entities"][x["entity"]]["falsePos"] += 1
								analysis["entities"][y["entity"]]["falseNeg"] += 1
								oEntities.remove(y)
								break
					if truePos:
						analysis["entities"][x["entity"]]["truePos"] += 1
					else:
						analysis["entities"][x["entity"]]["falsePos"] += 1


			for y in oEntities:
				Analyser.check_key(analysis["entities"], y["entity"])
				analysis["entities"][y["entity"]]["falseNeg"] += 1

			i += 1

		self.write_json(output_file, analysis)

if __name__ == '__main__':
	orig_json = "../../data/corpus/from_editor/balanced/paper-data-limit-H-F-D.json"
	anotation_output = "../../data/results/remote_annotations/watson/paper-data-limit-H-F-DAnnotations.json"
	analysis_output = "../../data/results/remote_annotations/watson/paper-data-limit-H-F-DAnalysis.json"

	watson_analyser = WatsonAnalyser()
	watson_analyser.get_annotations(orig_json, anotation_output)
	watson_analyser.analyse_annotations(anotation_output, orig_json,analysis_output)