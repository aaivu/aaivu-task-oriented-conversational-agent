import csv, numpy
import glob, os
import json
import pandas as pd

from rasa.test import get_evaluation_metrics
from rasa.shared.utils.cli import print_info, print_success
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.nlu.model import Interpreter, Metadata
from rasa import model
from rasa.shared.utils.io import json_to_string
import rasa.utils.common

class StackModels:
    __instance = None

    def __init__(self, model_folder_path):
        if StackModels.__instance is None:
            StackModels.__instance = object.__init__(self)
        self.model_folder_path = model_folder_path

    @staticmethod
    def get_loaded_model_path(model_path):
        temp_path = model.get_model(model_path)
        loaded_model_path = temp_path + "\\nlu"
        return loaded_model_path

    def load_models(self):
        models = glob.glob(self.model_folder_path + "\\*")
        loaded_models = {}
        for model in models:
            config = model.split("\\")[-1]
            # print("config", config)
            model = max(glob.glob(model + "\\*"), key=os.path.getctime)
            model_path = self.get_loaded_model_path(model)
            interpreter = Interpreter.load(model_path, component_builder=None)
            loaded_models["{}".format(config)] = interpreter
        return loaded_models


class ModelScores:
    __instance = None

    def __init__(self, model_score_path, model_folder_path=None, evaluation_dataset_path = None):
        if ModelScores.__instance is None:
            ModelScores.__instance = object.__init__(self)
        self.model_score_path = model_score_path
        self.evaluation_dataset_path = evaluation_dataset_path
        self.model_folder_path = model_folder_path

    def get_scores(self, avg="micro avg", score="f1-score", trim_count=None):
        models = glob.glob(self.model_score_path + "\\*")
        model_scores = {}

        for model in models:
            config = model.split("\\")[-1]
            try:
                with open("{}\intent_report.json".format(model, config)) as f:
                    intent_report = json.load(f)
            except (OSError, IOError) as e:
                print("error load in reports \n {} \n config {}".format(e, config))
            finally:
                model_score = intent_report[avg][score]
                model_scores[config] = model_score
                # print("score ", model_score)

        if trim_count:
            sorted_model_scores = {
                k: v
                for k, v in sorted(
                    model_scores.items(), key=lambda item: item[1], reverse=True
                )
            }
            try:
                dropped_model_scores = [
                    sorted_model_scores.popitem() for item in range(trim_count)
                ]
            except (KeyError or TypeError):
                print("cannot trim model scores")
            finally:
                return sorted_model_scores

        return model_scores

    def get_value_added_scores(self, avg="micro avg", score="f1-score"):
        model_scores = self.get_scores(avg, score)
        sorted_model_scores = {
            k: v for k, v in sorted(model_scores.items(), key=lambda item: item[1])
        }
        value_added_scores = {}

        for idx, key in enumerate(sorted_model_scores):
            value_added_scores[key] = idx + 1
        # print("value_added_scores :", value_added_scores)
        return value_added_scores

    @staticmethod
    def get_loaded_model_path(model_path):
        temp_path = model.get_model(model_path)
        loaded_model_path = temp_path + "\\nlu"
        return loaded_model_path

    @staticmethod
    def test_pipelines(self,test_data, model_path):
        model_path = self.get_loaded_model_path(model_path)
        interpreter = Interpreter.load(model_path, None)
        test_result = []
        test_entity_result = []

        for sentence in test_data:
            result = interpreter.parse(sentence)
            test_result.append(result["intent"]["name"])
            filtered_entities = {}
            for entity in result["entities"]:
                filtered_entities[entity["entity"]] = entity["value"]
            test_entity_result.append(filtered_entities)
        # print("test result  :", test_result)
        return tuple(test_result), test_entity_result

    @staticmethod
    def get_target_entities(dataset):
        target_entities = []

        for index, row in dataset.iterrows():
            extracted_entities = {}
            try:
                if type(row.entity_value_1) is not float:
                    extracted_entities[row.entity_name_1] = row.entity_value_1
                if type(row.entity_value_2) is not float:
                    extracted_entities[row.entity_name_2] = row.entity_value_2
                if type(row.entity_value_3) is not float:
                    extracted_entities[row.entity_name_3] = row.entity_value_3
            except ValueError:
                pass
            finally:
                target_entities.append(extracted_entities)

        return target_entities

    def get_model_evaluation_scores(self):
        evaluation_dataset = pd.read_csv(self.evaluation_dataset_path)
        models = glob.glob(self.model_folder_path + "\\*")
        # test_score_intents = {}
        test_score_entities={}

        sentences = evaluation_dataset.Sentence.to_list()
        # target_intents = evaluation_dataset.intent.to_list()
        target_entities = self.get_target_entities(evaluation_dataset)

        for model in models:
            config = model.split("\\")[-1]
            test_pipeline_predicted_intents, test_pipeline_predicted_entities= self.test_pipelines(self,sentences, model)
            dict_comparison = DictComparison(
                target_entities,
                test_pipeline_predicted_entities,
            )
            test_score_entities[config]= dict_comparison.f1score()
        print("test_score_entities: ",test_score_entities)
        return test_score_entities


class DictComparison:
    def __init__(self, targets, predictions, out_file_path=None):
        self.targets = targets
        self.predictions = predictions
        self.out_file_path = out_file_path

    @staticmethod
    def partial_confusion_matrix(self):
        TP, FP, FN = 0, 0, 0
        file_exist = 0
        if self.out_file_path:
            try:
                out_file = open(self.out_file_path, "a+")
                file_exist = 1
            except FileNotFoundError:
                print("cannot open output file to write partial_confusion_matrix")
                file_exist = 0

        for i in range(len(self.targets)):
            target = self.targets[i]
            prediction = self.predictions[i]
            target = dict((k.lower(), v.lower()) for k, v in target.items())
            prediction = dict((k.lower(), v.lower()) for k, v in prediction.items())

            TP += len(set(target.items()) & set(prediction.items()))
            FP += len(set(prediction.items()) - set(target.items()))
            FN += len(set(target.items()) - set(prediction.items()))

            if file_exist:
                try:
                    out_file.write("target :" + str(target) + "\n")
                    out_file.write("prediction :" + str(prediction) + "\n")
                    out_file.write(
                        "TP : {} \n".format(
                            set(target.items()) & set(prediction.items())
                        )
                    )
                    out_file.write(
                        "FP : {} \n".format(
                            set(prediction.items()) - set(target.items())
                        )
                    )
                    out_file.write(
                        "FN : {} \n\n".format(
                            set(target.items()) - set(prediction.items())
                        )
                    )
                except FileNotFoundError:
                    print("cannot write output file partial_confusion_matrix")

        if file_exist:
            try:
                out_file.close()
            except FileNotFoundError:
                print("no file to close in partial_confusion_matrix")

        print("TP, FP, FN ", TP, FP, FN)
        return (TP, FP, FN)

    def recall(self):
        TP, FP, FN = self.partial_confusion_matrix(self)
        return TP / (TP + FP)

    def precision(self):
        TP, FP, FN = self.partial_confusion_matrix(self)
        return TP / (TP + FN)

    def f1score(self):
        recall = self.recall()
        precision = self.precision()
        return 2 * precision * recall / (precision + recall)

