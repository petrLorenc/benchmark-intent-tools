import os
import json
import pandas as pd

from config import CONFIG
from nlu_analysers.dialogflow_analyser import DialogflowAnalyser
from nlu_local.approaches.utils import remove_contraction

HTML_TEMPLATE_HEAD = """<!DOCTYPE html>
                        <html>

                        <head>
                          <meta charset="utf-8">
                          <title></title>
                          <meta name="author" content="">
                          <meta name="description" content="">
                          <meta name="viewport" content="width=device-width, initial-scale=1">

                        </head>

                        <body>
                        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">
                        """

HTML_TEMPLATE = """ 
        <facets-dive id="fd" height="600"></facets-dive>
        <script>
          var data = {jsonStr};
          var fd = document.querySelector("#fd");
          fd.data = data;
          fd['verticalFacet'] = '{goldenIntent}';
          fd['verticalBuckets'] = {numClass};
          fd['horizontalFacet'] = '{intent}';
          fd['horizontalBuckets'] = {numClass};
          fd['colorBy'] = '{goldenIntent}';
        </script>
        <h1>{modelParams}</h1>
        <h3>OOV: {oov}s</h3>
        """

HTML_TEMPLATE_END = """
    </body>

    </html>"""


class SentenceEmbeddingApproach:

    def __init__(self,
                 name,
                 corpus_path=None,
                 available_embeddings=None,
                 can_use_tfidf=False):

        if available_embeddings is None:
            available_embeddings = ["FastTextSW", "FastText", "Sent2Vec"]

        if can_use_tfidf is True:
            self.use_tfidf_options = [True, False]
        elif type(can_use_tfidf) == list:
            self.use_tfidf_options = can_use_tfidf
        else:
            self.use_tfidf_options = [False]

        self.name = name
        self.X_train, self.X_test, self.y_train, self.y_test = [], [], [], []

        if corpus_path is not None:
            self.corpus_path = corpus_path
            data = json.load(open(corpus_path))

            for s in data["sentences"]:
                if s["training"]:
                    self.X_train.append(s['text'])
                    self.y_train.append(s["intent"])
                else:
                    self.X_test.append(s['text'])
                    self.y_test.append(s["intent"])

        self.available_embeddings = available_embeddings

        self.path = CONFIG["output"]["base_path"]
        for alg_name in self.available_embeddings:
            self.visualisation_path = os.path.join(self.path, "visualisation/{}/{}/".format(self.name, alg_name))
            os.makedirs(self.visualisation_path, exist_ok=True)

    def evaluate(self, evaluation_fn, lang="en"):
        output = {}

        model_params = {
            'name': self.name,
            'algorithm': 'Sent2Vec',
            'lang': lang,
            'use_tfidf': True
        }

        for algorithm in self.available_embeddings:
            model_params["algorithm"] = algorithm
            for use_tfidf in self.use_tfidf_options:


                output[algorithm + "_" + str(use_tfidf)] = {"data": [], "confidences": []}
                model_params["use_tfidf"] = use_tfidf

                payload = {"model": model_params, 'qa': {}}

                payload['qa'] = {}

                for unique_label in list(set(self.y_train)):
                    payload['qa'][unique_label] = {"answer": unique_label, "questions": []}
                for unique_label in list(set(self.y_test)):
                    payload['qa'][unique_label] = {"answer": unique_label, "questions": []}

                for sentence, label in zip(self.X_train, self.y_train):
                    payload['qa'][label]["questions"].append(remove_contraction(sentence.strip()))

                similarity_data, oov = evaluation_fn(payload, self.X_train, self.y_train, self.X_test, self.y_test)

                annotation_json = {"results": []}
                for example in similarity_data:
                    annotation_json["results"].append(example)

                annotation_path = os.path.join(self.path, "local_annotations", self.corpus_path.split("/")[-1].split(".")[0])
                os.makedirs(annotation_path, exist_ok=True)
                with open(annotation_path + "/" + self.name + "_annotation.json", "w") as f:
                    json.dump(annotation_json, f, sort_keys=False, indent=4, separators=(',', ': '), ensure_ascii=False)

                panda_df = pd.DataFrame([(item["confidence"],
                                          item["goldenIntent"],
                                          item["intent"],
                                          item["queryText"],
                                          item["queryHit"]) for item in similarity_data],
                                        columns= ["confidence", "goldenIntent", "intent", "queryText", "queryHit"])

                analyzer = DialogflowAnalyser()
                analyzer.analyse_annotations(annotations_file=annotation_path + "/" + self.name + "_annotation.json",
                                             corpus_file=self.corpus_path,
                                             output_file=annotation_path + "/" + self.name + "_analysis.json")

                self.visualise_to_file(algorithm, model_params, oov, panda_df, len(payload['qa']))


    def visualise_to_file(self, algorithm, model_params, oov_lists, panda_df, num_labels):
        final_in_html = ""
        html = HTML_TEMPLATE.format(
            jsonStr=panda_df.to_json(orient='records'),
            numClass=num_labels,
            goldenIntent="goldenIntent",
            intent="intent",
            modelParams=json.dumps(model_params),
            oov=oov_lists
        )

        final_in_html += html

        with open(self.visualisation_path + "/output_{}.html".format(self.name), "w") as f:
            f.write(HTML_TEMPLATE_HEAD)
            f.write(final_in_html)
            f.write(HTML_TEMPLATE_END)
