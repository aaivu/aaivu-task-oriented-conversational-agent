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
            # print("config", config)
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


def test_stack_pipelines(
    test_data, output_file_path, trim_count=None, fallback_confidence=0.5
):
    model_folder_path = "stack_models\pipelines"
    model_score_path = "results\pipelines"
    entity_out_file_path = "results\\stack_reports\\full_entity_result14.csv"

    target_entities = []

    stackmodels = StackModels(model_folder_path)
    stackmodels = stackmodels.load_models()
    modelscores = ModelScores(model_score_path)
    model_scores = modelscores.get_scores(trim_count=trim_count)
    value_added_scores = modelscores.get_value_added_scores()
    test_result = []
    test_entity_result = []

    for sentence_id, sentence in enumerate(test_data):
        if target_entities:
            try:
                with open(entity_out_file_path, "a+") as f:
                    f.write(
                        sentence
                        + "\n"
                        + "target :"
                        + str(target_entities[sentence_id])
                        + "\n"
                    )
            except FileNotFoundError:
                print("not found file to write target entities")

        result = model_stack_predict(
            sentence,
            stackmodels,
            model_scores,
            output_file_path,
            fallback_confidence,
            value_added_scores=None,
            entity_stacking_type=3,
            entity_out_file_path=entity_out_file_path,
        )

        test_result.append(result["intent"]["name"])
        filtered_entities = {}
        for entity in result["entities"]:
            filtered_entities[entity["entity"]] = entity["value"]
        test_entity_result.append(filtered_entities)

    # print("test_entity_result  :", test_entity_result)
    return tuple(test_result), test_entity_result


def get_loaded_model_path(model_path):
    temp_path = model.get_model(model_path)
    loaded_model_path = temp_path + "\\nlu"
    return loaded_model_path


def model_stack_predict(
    message,
    stackmodels,
    modelscores,
    output_file_path=None,
    fallback_confidence=0.5,
    entity_stacking_type=3,
    value_added_scores=None,
    entity_out_file_path=None,
    model_folder_path="stack_models\pipelines",
    model_score_path="results\pipelines",
):

    nlu_test_result_tofile = {"text": message}
    nlu_result = {"text": message}
    intent_ranking = []
    stack_entities = []
    entity_file_exist = 0

    if entity_out_file_path:
        try:
            entity_out_file = open(entity_out_file_path, "a+")
            entity_file_exist = 1
        except FileNotFoundError:
            print("cannot open output file to write entity results")
            entity_file_exist = 0

    for config_id, config in enumerate(modelscores):

        score = modelscores[config]

        try:
            result = stackmodels[config].parse(message)
        except KeyError:
            print("not found model {}".format(config_id))
            continue

        if entity_file_exist:
            entity_out_file.write(
                "pipeline {}".format(config_id) + str(result["entities"]) + "\n"
            )

        nlu_test_result_tofile[config] = result["intent"]

        for i in range(3):
            """consider the highest 3 intents predicted from the pipleline while stacking"""

            pred_intent = result["intent_ranking"][i]
            if pred_intent["name"] == "nlu_fallback":
                pred_intent["confidence"] = fallback_confidence * score

            if not intent_ranking:
                pred_intent["count"] = score
                pred_intent["confidence"] = pred_intent["confidence"] * score
                intent_ranking.append(pred_intent)
            else:
                for j in intent_ranking:
                    if j["name"] == pred_intent["name"]:
                        j["confidence"] = (
                            j["count"] * j["confidence"]
                            + pred_intent["confidence"] * score
                        ) / (j["count"] + score)
                        j["count"] += score
                        break
                else:
                    pred_intent["count"] = score
                    pred_intent["confidence"] = pred_intent["confidence"] * score
                    intent_ranking.append(pred_intent)

        if entity_stacking_type == 1:
            for entity in result["entities"]:
                if not stack_entities:
                    stack_entities.append(entity)
                elif (entity["entity"], entity["value"]) not in [
                    (en["entity"], en["value"]) for en in stack_entities
                ]:
                    stack_entities.append(entity)
            ## todo : check for same entities with diff values and same value with diff entities

        elif entity_stacking_type == 2:
            threshold = 0.9
            for entity in result["entities"]:
                try:
                    confidence_entity = entity["confidence_entity"]
                except KeyError:
                    confidence_entity = 0.9
                    print(
                        "not found confidence of the entity. set confidence entity to {}".format(
                            confidence_entity
                        )
                    )
                # else:
                #     confidence_entity = 0.7
                #     print("set confidence entity to 0.7")
                finally:
                    if not stack_entities and confidence_entity >= threshold:
                        stack_entities.append(entity)
                    if entity["entity"] in [en["entity"] for en in stack_entities]:
                        matched_entity = next(
                            item
                            for item in stack_entities
                            if item["entity"] == entity["entity"]
                        )
                        if matched_entity["value"] != entity["value"]:
                            if confidence_entity > matched_entity["confidence_entity"]:
                                matched_entity["value"] = entity["value"]
                    elif entity["value"] in [en["value"] for en in stack_entities]:
                        matched_entity = next(
                            item
                            for item in stack_entities
                            if item["value"] == entity["value"]
                        )
                        if matched_entity:
                            if matched_entity["entity"] != entity["entity"]:
                                if (
                                    confidence_entity
                                    > matched_entity["confidence_entity"]
                                ):
                                    matched_entity["entity"] = entity["entity"]
                    elif (entity["entity"], entity["value"]) not in [
                        (en["entity"], en["value"]) for en in stack_entities
                    ]:
                        stack_entities.append(entity)

        elif entity_stacking_type == 3:
            for entity in result["entities"]:
                try:
                    if value_added_scores:
                        confidence_entity = (
                            value_added_scores[config] * entity["confidence_entity"]
                        )
                    else:
                        confidence_entity = entity["confidence_entity"]
                except KeyError:
                    remove_from_regex = ["greeting", "thanks", "deny", "confirm_answer"]
                    if result["intent"]["name"] in remove_from_regex:
                        confidence_entity = 0
                    else:
                        confidence_entity = 1

                if not stack_entities:
                    entity["count"] = confidence_entity
                    stack_entities.append(entity)
                elif (entity["entity"], entity["value"]) in [
                    (en["entity"], en["value"]) for en in stack_entities
                ]:
                    matched_entity = next(
                        item
                        for item in stack_entities
                        if item["entity"] == entity["entity"]
                    )
                    matched_entity["count"] += confidence_entity
                else:
                    entity["count"] = confidence_entity
                    stack_entities.append(entity)

        nlu_result["intent"] = [
            intent
            for intent in intent_ranking
            if intent["confidence"] == max([en["confidence"] for en in intent_ranking])
        ][0]

    if entity_stacking_type == 3:
        initial_stack_entities = stack_entities
        if entity_file_exist:
            entity_out_file.write(
                "stack_entities before trim :" + str(initial_stack_entities) + "\n"
            )

        for num, i in enumerate(stack_entities):
            if i:
                if value_added_scores:
                    if i["count"] < 25:
                        stack_entities.remove(i)
                        continue
                if i["count"] < (len(modelscores) / 2):
                    stack_entities[num] = None
                    if entity_file_exist:
                        entity_out_file.write(
                            "remove {} with count {}".format(i["entity"], i["count"])
                            + "\n"
                        )
                    continue
                elif num > (len(stack_entities) - 1):
                    break
                else:
                    for j in stack_entities[num + 1 :]:
                        if j:
                            if i["entity"] == j["entity"]:
                                low_entity = i if i["count"] < j["count"] else j
                                stack_entities[stack_entities.index(low_entity)] = None
                            if i["value"] == j["value"] and i["start"] == j["start"]:
                                low_entity = i if i["count"] < j["count"] else j
                                stack_entities[stack_entities.index(low_entity)] = None

        stack_entities = [i for i in stack_entities if i]

        if entity_file_exist:
            entity_out_file.write(
                "stack_entities after trim :" + str(stack_entities) + "\n"
            )

    if entity_file_exist:
        entity_out_file.write(
            "stack_type {}".format(entity_stacking_type) + str(stack_entities) + "\n\n"
        )
        entity_out_file.close()

    nlu_result["entities"] = stack_entities
    nlu_result["intent_ranking"] = intent_ranking
    nlu_test_result_tofile["final intent"] = nlu_result["intent"]
    if output_file_path:
        try:
            with open(output_file_path, "a") as outfile:
                for key in nlu_test_result_tofile.keys():
                    outfile.write("%s,%s\n" % (key, nlu_test_result_tofile[key]))
                outfile.write("\n")
        except (OSError, IOError) as e:
            print("cannot open output file")
        finally:
            # print("nlu_test_result_tofile :", nlu_test_result_tofile)
            print("done")
    return nlu_result
