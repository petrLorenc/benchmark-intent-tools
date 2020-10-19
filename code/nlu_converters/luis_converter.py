import json

from nlu_converters.converter import Converter
from nlu_converters.annotated_sentence import AnnotatedSentence


class LuisConverter(Converter):
	LUIS_SCHEMA_VERSION = "2.2.0"

	def __init__(self):
		super(LuisConverter, self).__init__()
		self.bing_entities = set()
		self.test_utterances = []

	def __add_intent(self, intent):
		self.intents.add(intent)

	def __add_entity(self, entity):
		self.entities.add(entity)

	def __add_bing_entity(self, entity):
		self.bing_entities.add(entity)

	def __add_utterance(self, sentence):
		entities = []
		start_index = None
		for e in sentence.entities:
			#Calculate the position based on character count.
			words = (sentence.text).split(' ')
			index = 0
			for i in range(len(words)):
				if i == e["start"]:
					start_index = index
					end_index = index + len(words[i])
				elif i == e["stop"]:
					end_index = index + len(words[i])
					break
				index = index + len(words[i]) + 1
			# if start_index:
			# 	entities.append({"entity": e["entity"], "startPos": start_index, "endPos": end_index-1})
		self.utterances.append({"text": sentence.text, "intent": sentence.intent, "entities": entities})

	def import_corpus(self, file):
		data = json.load(open(file))
		#meta data
		self.name = data["name"]
		self.desc = data["desc"]
		#dirty quickfix
		if(data["lang"] == "en"):
			self.lang = "en-us"
		else:
			self.lang = data["lang"] + "-" + data["lang"]

		#training data
		for s in data["sentences"]:
			if s["training"]: #only import training data
				#intents
				self.__add_intent(s["intent"])			
				#entities
				for e in s["entities"]:				
					self.__add_entity(e["entity"])        	
				#utterances
				self.__add_utterance(AnnotatedSentence(s["text"], s["intent"], s["entities"]))
			else:
				self.test_utterances.append({"text": s["text"], "intent": s["intent"], "entities": []})

	def export(self, file):
		luis_json = {}
		luis_json["luis_schema_version"] = self.LUIS_SCHEMA_VERSION 
		luis_json["versionId"] = "0.1.0"
		luis_json["name"] = self.name
		luis_json["desc"] = self.desc
		luis_json["culture"] = self.lang  
		luis_json["intents"] = self.array_to_json(self.intents)
		luis_json["entities"] = self.array_to_json(self.entities)
		luis_json["bing_entities"] = self.array_to_json(self.bing_entities)
		luis_json["model_features"] = []
		luis_json["regex_features"] = []
		luis_json["regex_entities"] = []
		luis_json["composites"] = []
		luis_json["closedLists"] = []
		luis_json["utterances"] = self.utterances
		self.write_json(file, luis_json)

	def export_test(self, file):
		self.write_json(file, self.utterances)


if __name__ == '__main__':
	# luis_converter = LuisConverter()
	# luis_converter.import_corpus("../../data/AskUbuntuCorpus.json")
	# luis_converter.export("../../data/generated/luis/AskUbuntuCorpusTraining_Luis.json")
	#
	# luis_converter = LuisConverter()
	# luis_converter.import_corpus("../../data/ChatbotCorpus.json")
	# luis_converter.export("../../data/generated/luis/ChatbotCorpusTraining_Luis.json")
	#
	# luis_converter = LuisConverter()
	# luis_converter.import_corpus("../../data/WebApplicationsCorpus.json")
	# luis_converter.export("../../data/generated/luis/WebApplicationsTraining_Luis.json")
	#


	# watson_converter = LuisConverter()
	# watson_converter.import_corpus("../../data/corpus/from_datasets/AskUbuntuCorpusEnrich.json")
	# watson_converter.export("../../data/generated/luis/AskUbuntuCorpusEnrichTraining_Luis.json")
	#
	# watson_converter = LuisConverter()
	# watson_converter.import_corpus("../../data/corpus/from_datasets/ChatbotCorpusEnrich.json")
	# watson_converter.export("../../data/generated/luis/ChatbotCorpusEnrichTraining_Luis.json")
	#
	# watson_converter = LuisConverter()
	# watson_converter.import_corpus("../../data/corpus/from_datasets/WebApplicationsCorpusEnrich.json")
	# watson_converter.export("../../data/generated/luis/WebApplicationsEnrichTraining_Luis.json")


	luis_converter = LuisConverter()
	luis_converter.import_corpus("../../data/corpus/from_editor/Yes-no-maybe-20-Corpus.json")
	luis_converter.export("../../data/generated/luis/YesNoMaybe20Training_Luis.json")
	#
	# luis_converter = LuisConverter()
	# luis_converter.import_corpus("../../data/corpus/Yes-no-maybe-20-H-F-Corpus.json")
	# luis_converter.export("../../data/generated/luis/YesNoMaybeLimit20HFTraining_Luis.json")
	#
	# luis_converter = LuisConverter()
	# luis_converter.import_corpus("../../data/corpus/Yes-no-maybe-20-H-D-F-Corpus.json")
	# luis_converter.export("../../data/generated/luis/YesNoMaybeLimit20HDFTraining_Luis.json")

	#
	watson_converter = LuisConverter()
	watson_converter.import_corpus("../../data/corpus/from_editor/balanced/paper-data-limit-H-F-D.json")
	watson_converter.export("../../data/generated/luis/paper-data-limitTraining_Luis.json")
	watson_converter.export_test("../../data/generated/luis/paper-data-limitTesting_Luis.json")


