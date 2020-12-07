import logging
import typing
import glob, os
import json
from typing import Optional, Text

from rasa.shared.utils.cli import print_info, print_success
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.nlu.model import Interpreter, Metadata
from rasa import model
from rasa.shared.utils.io import json_to_string
import rasa.utils.common

if typing.TYPE_CHECKING:
    from rasa.nlu.components import ComponentBuilder

logger = logging.getLogger(__name__)


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
            print("config", config)
            model = max(glob.glob(model + "\\*"), key=os.path.getctime)
            model_path = get_loaded_model_path(model)
            interpreter = Interpreter.load(model_path, component_builder=None)
            loaded_models["{}".format(config)] = interpreter
        return loaded_models


class ModelScores:
    __instance = None

    def __init__(self, model_score_path):
        if ModelScores.__instance is None:
            ModelScores.__instance = object.__init__(self)
        self.model_score_path = model_score_path

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
                print("score ", model_score)

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
        print("value_added_scores :", value_added_scores)
        return value_added_scores


def run_cmdline(component_builder: Optional["ComponentBuilder"] = None) -> None:
    # interpreter = Interpreter.load(model_path, component_builder)
    regex_interpreter = RegexInterpreter()

    print_success("NLU model loaded. Type a message and press enter to parse it.")
    while True:
        print_success("Next message:")
        try:
            message = input().strip()
        except (EOFError, KeyboardInterrupt):
            print_info("Wrapping up command line chat...")
            break

        if message.startswith(INTENT_MESSAGE_PREFIX):
            result = rasa.utils.common.run_in_loop(regex_interpreter.parse(message))
        else:
            # result = interpreter.parse(message)
            model_folder_path = "stack_models\pipelines"
            model_stack_predict(model_folder_path, message)

        # print(json_to_string(result))


def test_pipelines(test_data, model_path):
    model_path = get_loaded_model_path(model_path)
    interpreter = Interpreter.load(model_path, None)
    test_result = []
    for sentence in test_data:
        result = interpreter.parse(sentence)
        test_result.append(result["intent"]["name"])
    print("test result  :", test_result)
    return tuple(test_result)


def test_stack_pipelines(test_data, output_file_path,trim_count=None):
    model_folder_path = "stack_models\pipelines"
    model_score_path = "results\pipelines"

    stackmodels = StackModels(model_folder_path)
    stackmodels = stackmodels.load_models()
    modelscores = ModelScores(model_score_path)
    model_scores = modelscores.get_scores(trim_count=trim_count)
    value_added_scores = modelscores.get_value_added_scores()
    test_result = []

    for sentence in test_data:
        result = model_stack_predict(
            sentence, stackmodels, model_scores, output_file_path
        )
        test_result.append(result["intent"]["name"])
    print("test result  :", test_result)
    return tuple(test_result)


def get_loaded_model_path(model_path):
    temp_path = model.get_model(model_path)
    loaded_model_path = temp_path + "\\nlu"
    return loaded_model_path


def model_stack_predict(
    message,
    stackmodels,
    modelscores,
    output_file_path,
    model_folder_path="stack_models\pipelines",
    model_score_path="results\pipelines",
):

    nlu_test_result_tofile = {"text": message}
    nlu_result = {"text": message}
    intent_ranking = []
    entities = []
    print("stacmodels , ", stackmodels)

    for config in modelscores:
        print("config", config)
        print(stackmodels[config])
        score = modelscores[config]
        
        try:
            result = stackmodels[config].parse(message)
        except KeyError:
            print("not found model scores")
            continue

        nlu_test_result_tofile[config] = result["intent"]

        for i in range(3):
            pred_intent = result["intent_ranking"][i]
            if not (pred_intent["name"] == "nlu_fallback"):
                if not intent_ranking:
                    pred_intent["count"] = score
                    pred_intent["confidence"] = pred_intent["confidence"] * score
                    intent_ranking.append(pred_intent)
                else:
                    for j in intent_ranking:
                        if j["name"] == pred_intent["name"]:
                            # print("######################")
                            j["confidence"] = (
                                j["count"] * j["confidence"]
                                + pred_intent["confidence"] * score
                            ) / (j["count"] + score)
                            # print("#####################jconfidence ",j['confidence'])
                            j["count"] += score
                            break
                    else:
                        pred_intent["count"] = 1
                        intent_ranking.append(pred_intent)
            # print("intent_ranking ", intent_ranking)

        for entity in result["entities"]:
            if not entities:
                entities.append(entity)
            elif (entity["entity"], entity["value"]) not in [
                (en["entity"], en["value"]) for en in entities
            ]:
                entities.append(entity)
            ## todo : check for same entities with diff values and same value with diff entities

    # print("final_intent_ranking ", intent_ranking)
    # print("final_intent ", [intent for intent in intent_ranking if intent['confidence']== max([en['confidence'] for en in intent_ranking])])
    # print("final_entities ", entities)
    nlu_result["intent"] = [
        intent
        for intent in intent_ranking
        if intent["confidence"] == max([en["confidence"] for en in intent_ranking])
    ][0]
    nlu_result["entities"] = entities
    nlu_result["intent_ranking"] = intent_ranking
    nlu_test_result_tofile["final intent"] = nlu_result["intent"]
    try:
        with open(output_file_path, "a") as outfile:
            for key in nlu_test_result_tofile.keys():
                outfile.write("%s,%s\n" % (key, nlu_test_result_tofile[key]))
            outfile.write("\n")
    except (OSError, IOError) as e:
        print("cannot open output file")
    finally:
        print("nlu_test_result_tofile :", nlu_test_result_tofile)
        print("final   ###### :", nlu_result)
    return nlu_result


# def model_stack_predict(model_folder_path, message, output_file_path):
#     models = glob.glob(model_folder_path + "\\*")
#     nlu_test_result_tofile = {"text": message}
#     nlu_result = {"text": message}
#     intent_ranking = []
#     entities = []

#     for idx,model in enumerate(models):
#         config = model.split("\\")[-1]
#         print("config", config)
#         model = max(glob.glob(model + "\\*"), key=os.path.getctime)

#         with open("results\pipelines\{}\intent_report.json".format(config)) as f:
#             intent_report = json.load(f)
#         f1_score = intent_report["micro avg"]["f1-score"]
#         print("f1_score ", f1_score)
#         # f1_score = 1
#         model_path = get_loaded_model_path(model)
#         interpreter = Interpreter.load(model_path, component_builder=None)
#         result = interpreter.parse(message)
#         # print("result ", result)

#         nlu_test_result_tofile[str(idx)] = result['intent']

#         for i in range(3):
#             pred_intent = result["intent_ranking"][i]
#             if not (pred_intent["name"]=='nlu_fallback'):
#                 if not intent_ranking:
#                     pred_intent["count"] = f1_score
#                     pred_intent["confidence"] = pred_intent["confidence"] * f1_score
#                     intent_ranking.append(pred_intent)
#                 else:
#                     for j in intent_ranking:
#                         if j["name"] == pred_intent["name"]:
#                             # print("######################")
#                             j["confidence"] = (
#                                 j["count"] * j["confidence"]
#                                 + pred_intent["confidence"] * f1_score
#                             ) / (j["count"] + f1_score)
#                             # print("#####################jconfidence ",j['confidence'])
#                             j["count"] += f1_score
#                             break
#                     else:
#                         pred_intent["count"] = 1
#                         intent_ranking.append(pred_intent)
#             # print("intent_ranking ", intent_ranking)

#         for entity in result["entities"]:
#             if not entities:
#                 entities.append(entity)
#             elif (entity["entity"], entity["value"]) not in [
#                 (en["entity"], en["value"]) for en in entities
#             ]:
#                 entities.append(entity)
#             ## todo : check for same entities with diff values and same value with diff entities

#     # print("final_intent_ranking ", intent_ranking)
#     # print("final_intent ", [intent for intent in intent_ranking if intent['confidence']== max([en['confidence'] for en in intent_ranking])])
#     # print("final_entities ", entities)
#     nlu_result["intent"] = [
#         intent
#         for intent in intent_ranking
#         if intent["confidence"] == max([en["confidence"] for en in intent_ranking])
#     ][0]
#     nlu_result["entities"] = entities
#     nlu_result["intent_ranking"] = intent_ranking
#     nlu_test_result_tofile['final intent'] = nlu_result["intent"]
#     try:
#         with open(output_file_path, 'a') as outfile:
#             for key in nlu_test_result_tofile.keys():
#                 outfile.write("%s,%s\n"%(key,nlu_test_result_tofile[key]))
#             outfile.write("\n")
#     except (OSError, IOError) as e:
#         print("cannot open output file")
#     finally:
#         print("nlu_test_result_tofile :", nlu_test_result_tofile)
#         print("final   ###### :", nlu_result)
#     return (nlu_result)


# path1 = "C:\\Users\\Durashi\\AppData\\Local\\Temp\\tmpp4vfq2da\\nlu"
# path2 = "C:\\Users\\Durashi\\testbot\\stack_models\\nlu-20201031-221332.tar.gz"
# path0 = "stack_models\\nlu-20201111-025247.tar.gz"
# path3 = model.get_model(path0)
# print("path3: ",path3)
# run_cmdline()
