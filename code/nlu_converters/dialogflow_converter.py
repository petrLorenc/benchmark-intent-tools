from nlu_converters.converter import Converter
from nlu_converters.annotated_sentence import AnnotatedSentence


import shutil
import os
import hashlib
import zipfile
import json

class DialogflowConverter(Converter):
    @staticmethod
    def zipdir(path, ziph):
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))

    @staticmethod
    def add_to_list(list, str_to_add):
        if str_to_add not in list:
            list.append(str_to_add)
        return list

    def __init__(self):
        super(DialogflowConverter, self).__init__()
        self.intents = {}
        self.entities = []

    def __add_entity(self, sentence):
        obj = None

        for e in self.entities:
            if e["name"] == sentence.entities["entity"]:
                obj = e
                break

        if obj == None:
            obj = {"name": sentence.entities["entity"], "values": []}
        self.entities.append(obj)

        s = Converter.tokenizer(sentence.text)
        value = ""
        e = sentence.entities

        for i in range(0, len(s)):
            if i in range(int(e["start"]), int(e["stop"]) + 1):
                value = value + " " + s[i]

        value = value.lstrip()
        value = " ".join(Converter.tokenizer(value)).replace("	", " ")
        self.add_to_list(obj["values"], value)

    def __add_utterance(self, sentence):
        entities = []

        for e in sentence.entities:
            entities.append({"entity": e["entity"], "startPos": e["start"], "endPos": e["stop"]})

        u = {"text": sentence.text, "intent": sentence.intent, "entities": entities}
        self.utterances.append(u)

        try:
            self.intents[u["intent"]]
        except KeyError as e:
            self.intents[u["intent"]] = []

        self.intents[u["intent"]].append(u)

    def import_corpus(self, file):
        data = json.load(open(file))

        # meta data
        self.name = data["name"]
        self.desc = data["desc"]
        self.lang = data["lang"]

        # training data
        for s in data["sentences"]:
            if s["training"]:  # only import training data
                # entities
                # for e in s["entities"]:
                #     self.__add_entity(AnnotatedSentence(s["text"], s["intent"], e))
                # utterances + intents
                self.__add_utterance(AnnotatedSentence(s["text"], s["intent"], s["entities"]))

    def __get_word(self, sentence, entity):
        s = Converter.tokenizer(sentence)
        value = ""
        e = entity
        for i in range(0, len(s)):
            if i in range(int(e["startPos"]), int(e["endPos"]) + 1):
                value = value + " " + s[i]
        value = value.lstrip()
        value = " ".join(Converter.tokenizer(value)).replace("	", " ")
        return value

    def __agent_to_json(self):
        my_json = {}
        my_json["language"] = self.lang
        my_json["enabledDomainFeatures"] = ['smalltalk-domain-on', 'smalltalk-fulfillment-on'];
        my_json["defaultTimezone"] = ""
        my_json["customClassifierMode"] = "use.after"
        my_json["mlMinConfidence"] = 0.2
        return my_json

    def __intent_to_json(self, intent):
        name = intent[0]['intent']

        my_json = {}
        my_json["name"] = name
        my_json["auto"] = True
        my_json["contexts"] = []
        my_json["priority"] = 500000
        my_json["fallbackIntent"] = False
        my_json["webhookUsed"] = False
        my_json["events"] = []

        response = {}
        response["affectedContexts"] = []
        response["resetContexts"] = False
        response["dataType"] = "@" + name
        response["name"] = name
        response["value"] = "$" + name
        parameter = {"isListe": False}
        message = {"type": 0, "speech": []}
        response["parameters"] = [parameter]
        response["messages"] = [message]
        user_says = []

        counter = 0

        for s in intent:
            counter += 1

            j = {}
            words = []
            entities = s["entities"]
            sen = s["text"]
            entities = sorted(entities, key=lambda k: k['startPos'])

            for e in entities:
                word = Converter.detokenizer(self.__get_word(s["text"], e))
                split = ' '
                if word != '':
                    split = sen.split(word)

                if len(split[0]) > 0:
                    words.append({'text': split[0]})

                words.append({'text': word, 'userDefined': True, 'alias': e["entity"], 'meta': "@" + e["entity"]})

                try:
                    sen = split[1]
                except:
                    pass

            try:
                if len(entities) < 1:
                    words.append({'text': sen})
                elif len(split[1]) > 0:
                    words.append({'text': split[1]})
            except:
                pass

            j["data"] = words
            j["isTemplate"] = False
            j["count"] = 0
            j["id"] = hashlib.md5(str(counter).encode("utf8")).hexdigest()
            user_says.append(j)

        my_json["userSays"] = user_says
        my_json["responses"] = [response]

        return my_json

    def __entity_to_json(self, entity):
        my_json = {}
        my_json["name"] = entity["name"]
        my_json["isOverridable"] = True
        my_json["isEnum"] = True
        my_json["automatedExpansion"] = False

        entries = []
        for value in entity["values"]:
            e = {};
            e["value"] = Converter.detokenizer(value)
            e["synonyms"] = []
            entries.append(e)

        my_json["entries"] = entries
        return my_json

    def export(self, file):
        # create temp folder
        shutil.rmtree(self.name, ignore_errors=True)
        newpath = self.name
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        os.makedirs(self.name + '/entities')
        os.makedirs(self.name + '/intents')

        # agent.json
        self.write_json(self.name + "/agent.json", self.__agent_to_json())
        # intents
        for i in self.intents:
            self.write_json(self.name + "/intents/" + i + ".json", self.__intent_to_json(self.intents[i]))
        # entities
        for e in self.entities:
            self.write_json(self.name + "/entities/" + e["name"] + ".json", self.__entity_to_json(e))

        # zip files
        zipf = zipfile.ZipFile(file, 'w', zipfile.ZIP_DEFLATED)
        self.zipdir(self.name, zipf)
        zipf.close()

        # remove temp files
        shutil.rmtree(self.name, ignore_errors=True)


if __name__ == '__main__':
    # watson_converter = DialogflowConverter()
    # watson_converter.import_corpus("../ChatbotCorpus.json")
    # watson_converter.export("../ChatbotCorpusTraining_Dialogflow.zip")
    #
    # watson_converter = DialogflowConverter()
    # watson_converter.import_corpus("../AskUbuntuCorpus.json")
    # watson_converter.export("../AskUbuntuCorpusTraining_Dialogflow.zip")
    #
    # watson_converter = DialogflowConverter()
    # watson_converter.import_corpus("../WebApplicationsCorpus.json")
    # watson_converter.export("../WebApplicationsTraining_Dialogflow.zip")


    # watson_converter = DialogflowConverter()
    # watson_converter.import_corpus("../../data/corpus/from_datasets/AskUbuntuCorpusEnrich.json")
    # watson_converter.export("../../data/generated/dialogflow/AskUbuntuCorpusEnrichTraining_Dialogflow.zip")
    #
    # watson_converter = DialogflowConverter()
    # watson_converter.import_corpus("../../data/corpus/from_datasets/ChatbotCorpusEnrich.json")
    # watson_converter.export("../../data/generated/dialogflow/ChatbotCorpusEnrichTraining_Dialogflow.zip")
    #
    # watson_converter = DialogflowConverter()
    # watson_converter.import_corpus("../../data/corpus/from_datasets/WebApplicationsCorpusEnrich.json")
    # watson_converter.export("../../data/generated/dialogflow/WebApplicationsEnrichTraining_Dialogflow.zip")
    #
    #
    # watson_converter = DialogflowConverter()
    # watson_converter.import_corpus("../../data/corpus/from_editor/Yes-no-maybe-20-Corpus.json")
    # watson_converter.export("../../data/generated/dialogflow/YesNoMaybeLimit20Training_Watson.zip")
    #
    # watson_converter = DialogflowConverter()
    # watson_converter.import_corpus("../../data/corpus/from_editor/Yes-no-maybe-20-H-F-Corpus.json")
    # watson_converter.export("../../data/generated/dialogflow/YesNoMaybeLimit20HFTraining_Watson.zip")
    #
    # watson_converter = DialogflowConverter()
    # watson_converter.import_corpus("../../data/corpus/from_editor/Yes-no-maybe-20-H-D-F-Corpus.json")
    # watson_converter.export("../../data/generated/dialogflow/YesNoMaybeLimit20HDFTraining_Watson.zip")

    watson_converter = DialogflowConverter()
    watson_converter.import_corpus("../../data/corpus/from_editor/balanced/paper-data-limit-H-F-D.json")
    watson_converter.export("../../data/generated/dialogflow/paper-data-limit-H-F-D_training_dialogflow.zip")

    # luis_converter = DialogflowConverter()
    # luis_converter.import_corpus("../editor-preview-dialogflow.json")
    # luis_converter.export("../Editor_Dialogflow.zip")
