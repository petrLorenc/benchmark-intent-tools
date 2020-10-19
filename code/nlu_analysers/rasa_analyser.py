from nlu_analysers.luis_analyser import *
from config import CONFIG

class RasaAnalyser(LuisAnalyser):
	def __init__(self, rasa_url=CONFIG["rasa"]["url"]):
		super(LuisAnalyser, self).__init__()
		self.url = rasa_url
		self.post = True

# rasa run --enable-api -m models/nlu-20190515-144445.tar.gz

if __name__ == '__main__':
	# orig_json = "../../data/corpus/from_datasets/WebApplicationsCorpusEnrich.json"
	# anotation_output = "../../data/results/remote_annotations/rasa/WebApplicationsCorpusEnrichAnnotations.json"
	# analysis_output = "../../data/results/remote_annotations/rasa/WebApplicationsCorpusEnrichAnalysis.json"
	#
	# watson_analyser = RasaAnalyser("http://localhost:5005/model/parse?emulation_mode=luis")
	# watson_analyser.get_annotations(orig_json, anotation_output)
	# watson_analyser.analyse_annotations(anotation_output, orig_json, analysis_output)

	# orig_json = "../../data/WebApplicationsCorpus.json"
	# anotation_output = "../../data/results/remote_annotations/rasa/WebAnnotations.json"
	# analysis_output = "../../data/results/remote_annotations/rasa/WebAnalysis.json"

	# orig_json = "../../data/ChatbotCorpus.json"
	# anotation_output = "../../data/results/remote_annotations/rasa/ChatAnnotations.json"
	# analysis_output = "../../data/results/remote_annotations/rasa/ChatAnalysis.json"

	# orig_json = "../../data/AskUbuntuCorpus.json"
	# anotation_output = "../../data/results/remote_annotations/rasa/AskAnnotations.json"
	# analysis_output = "../../data/results/remote_annotations/rasa/AskAnalysis.json"


	orig_json = "../../data/corpus/from_editor/balanced/paper-data-limit-H-F-D.json"
	anotation_output = "../../data/results/remote_annotations/rasa/CIIRCAnnotations.json"
	analysis_output = "../../data/results/remote_annotations/rasa/CIIRCAnalysis.json"


	watson_analyser = RasaAnalyser("http://localhost:5005/model/parse?emulation_mode=luis")
	watson_analyser.get_annotations(orig_json, anotation_output)
	watson_analyser.analyse_annotations(anotation_output, orig_json, analysis_output)
