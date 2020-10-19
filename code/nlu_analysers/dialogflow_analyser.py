import json
import time
import random

import dialogflow_v2 as dialogflow

from nlu_analysers.analyser import Analyser
from config import CONFIG

class DialogflowAnalyser(Analyser):
	def __init__(self, project_id=CONFIG["dialogflow"]["project_id"]):
		super(DialogflowAnalyser, self).__init__()
		self.session_client = None
		self.project_id = project_id

	def get_annotations(self, corpus, output):
		data = json.load(open(corpus))		
		annotations = {'results':[]}
		# LAZY INIT because of local evaluation
		self.session_client = dialogflow.SessionsClient()

		for idx, s in enumerate(data["sentences"]):
			if not s["training"]: #only use test data
				text_input = dialogflow.types.TextInput(text=s["text"][:256], language_code="en-US")
				query_input = dialogflow.types.QueryInput(text=text_input)

				is_returned = False
				while is_returned is False:
					try:
						# create new session for not being influenced by history
						session = self.session_client.session_path(self.project_id, str(random.randint(0, 100000)))
						response = self.session_client.detect_intent(session=session, query_input=query_input)
						is_returned = True
					except:
						print("Saving progress at index {} and waiting 60s because of quota per minute.".format(idx))
						with open(output, "w") as file:
							json.dump(annotations, file, sort_keys=False, indent=4, separators=(',', ': '), ensure_ascii=False)
						time.sleep(60)


				annotations['results'].append({"queryText": response.query_result.query_text, "intent": response.query_result.intent.display_name})
		
		with open(output, "w") as file:
			json.dump(annotations, file, sort_keys=False, indent=4, separators=(',', ': '), ensure_ascii=False)

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
			print(a)
			if not a["queryText"].strip() == gold_standard[i]["text"].strip():
				print("{} !=".format(a["queryText"]))
				print("{}".format(gold_standard[i]["text"]))
				print("WARNING! Texts not equal")

			#intent
			try:
				aIntent = a["intent"]
			except:
				aIntent = "None"
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
			try:
				aEntities = a["queryResult"]["parameters"]
			except:
				aEntities = {}
			oEntities = gold_standard[i]["entities"]

			for x in aEntities.keys():
				Analyser.check_key(analysis["entities"], x)

				if len(oEntities) < 1: #false pos
					analysis["entities"][x]["falsePos"] += 1
				else:
					truePos = False

					for y in oEntities:
						if len(aEntities[x]) != 0 and aEntities[x][0].lower() == y["text"].lower():
							if x == y["entity"]: #truePos
								truePos = True
								oEntities.remove(y)
								break
							else:						 #falsePos + falseNeg
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
	# luis_analyser = MyDataAnalyser("testing-qdcuog")
	# luis_analyser.get_annotations("../WebApplicationsCorpus.json", "../WebApplicationsAnnotations_Dialogflow.json")
	# luis_analyser.analyse_annotations("../WebApplicationsAnnotations_Dialogflow.json", "../WebApplicationsCorpus.json",
	# 								  "../WebApplicationsAnalysis_Dialogflow.json")
	#
	# luis_analyser = MyDataAnalyser("testing-qdcuog")
	# luis_analyser.get_annotations("../AskUbuntuCorpus.json", "../AskUbuntuCorpusAnnotations_Dialogflow.json")
	# luis_analyser.analyse_annotations("../AskUbuntuCorpusAnnotations_Dialogflow.json", "../AskUbuntuCorpus.json",
	# 								  "../AskUbuntuCorpusAnalysis_Dialogflow.json")

	# luis_analyser = MyDataAnalyser("testing-qdcuog")
	# luis_analyser.get_annotations("../ChatbotCorpus.json", "../ChatbotCorpusAnnotations_Dialogflow.json")
	# luis_analyser.analyse_annotations("../ChatbotCorpusAnnotations_Dialogflow.json", "../ChatbotCorpus.json",
	# 								  "../ChatbotCorpusAnalysis_Dialogflow.json")

	# luis_analyser = MyDataAnalyser("testing-qdcuog")
	# luis_analyser.get_annotations("../editor-preview-dialogflow.json", "../EditorPreviewAnnotations_Dialogflow.json")
	# luis_analyser.analyse_annotations("../EditorPreviewAnnotations_Dialogflow.json", "../editor-preview-dialogflow.json",
	# 								  "../EditorPreviewAnalysis_Dialogflow.json")

	orig_json = "../../data/WebApplicationsCorpus.json"
	anotation_output = "../../data/results/remote_annotations/dialogflow/WebAnnotations.json"
	analysis_output = "../../data/results/remote_annotations/dialogflow/WebCorpusAnalysis.json"

	luis_analyser = DialogflowAnalyser("testing-qdcuog")
	luis_analyser.get_annotations(orig_json, anotation_output)
	luis_analyser.analyse_annotations(anotation_output, orig_json, analysis_output)


