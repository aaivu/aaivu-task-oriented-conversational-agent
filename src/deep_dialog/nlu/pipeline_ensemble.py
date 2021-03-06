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
from deep_dialog.nlu.pipeline_ensemble_preprocessing import ModelScores, StackModels
import logging
import typing
import glob, os
import json
from typing import Optional, Text

logger = logging.getLogger(__name__)


def run_cmdline(component_builder: Optional["ComponentBuilder"] = None) -> None:
    # interpreter = Interpreter.load(model_path, component_builder)
    regex_interpreter = RegexInterpreter()

    while True:
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
    model_folder_path = "nlu/stack_models/pipelines"
    model_score_path = "nlu/results/pipelines"
    entity_out_file_path = "nlu/results/stack_reports/full_entity_result14.csv"
    evaluation_dataset_path = "nlu/data/evaluation_data.csv"

    target_entities = []

    stackmodels = StackModels(model_folder_path)
    stackmodels = stackmodels.load_models()
    modelscores = ModelScores(
        model_score_path,
        model_folder_path=model_folder_path,
        evaluation_dataset_path=evaluation_dataset_path,
    )
    model_scores = modelscores.get_scores(trim_count=trim_count)
    entity_scores = modelscores.get_model_evaluation_scores()
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
            entity_scores,
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
    loaded_model_path = temp_path + "/nlu"
    return loaded_model_path


def model_stack_predict(
    message,
    stackmodels,
    modelscores,
    modelentityscores,
    output_file_path=None,
    fallback_confidence=0.5,
    entity_stacking_type=3,
    value_added_scores=None,
    entity_out_file_path=None,
    model_folder_path="nlu/stack_models/pipelines",
    model_score_path="nlu/results/pipelines",
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
            if not modelentityscores[config]:
                modelentityscores[config] = 0.5
            for entity in result["entities"]:
                try:
                    if value_added_scores:
                        confidence_entity = value_added_scores[config]
                        count = value_added_scores[config]
                    else:
                        confidence_entity = (
                            entity["confidence_entity"] * modelentityscores[config]
                        )
                        count = modelentityscores[config]
                except KeyError:
                    remove_from_regex = ["greeting", "thanks", "deny", "confirm_answer"]
                    if result["intent"]["name"] in remove_from_regex:
                        confidence_entity = 0
                    else:
                        confidence_entity = modelentityscores[config]
                    count = modelentityscores[config]

                if not stack_entities:
                    entity["count"] = count
                    entity["confidence_entity"] = confidence_entity
                    stack_entities.append(entity)
                elif (entity["entity"], entity["value"]) in [
                    (en["entity"], en["value"]) for en in stack_entities
                ]:
                    matched_entity = next(
                        item
                        for item in stack_entities
                        if item["entity"] == entity["entity"]
                    )
                    matched_entity["confidence_entity"] += confidence_entity
                    matched_entity["count"] += count
                else:
                    entity["count"] = count
                    entity["confidence_entity"] = confidence_entity
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

        pipeline_sum = sum(modelentityscores.values())

        for num, i in enumerate(stack_entities):
            if i:
                # if i["entity"] in ["PERSON", "LOC", "ORG", "PRODUCT","DATE","TIME"]:
                #     stack_entities.remove(i)
                #     continue
                if value_added_scores:
                    if i["confidence_entity"] < 25 or i["confidence"] < 25:
                        stack_entities.remove(i)
                        continue
                if (i["confidence_entity"] or i["confidence"]) < (pipeline_sum / 2):
                    stack_entities[stack_entities.index(i)] = None
                    if entity_file_exist:
                        entity_out_file.write(
                            "remove {} with count {}".format(
                                i["entity"], i["confidence_entity"]
                            )
                            + "\n"
                        )
                elif num > (len(stack_entities) - 1):
                    break
                else:
                    for j in stack_entities[num + 1 :]:
                        if j:
                            if i["entity"] == j["entity"]:
                                low_entity = (
                                    i
                                    if i["confidence_entity"] < j["confidence_entity"]
                                    else j
                                )
                                if low_entity in stack_entities:
                                    stack_entities[
                                        stack_entities.index(low_entity)
                                    ] = None
                            if i["value"] == j["value"] and i["start"] == j["start"]:
                                low_entity = (
                                    i
                                    if i["confidence_entity"] < j["confidence_entity"]
                                    else j
                                )
                                if low_entity in stack_entities:
                                    stack_entities[
                                        stack_entities.index(low_entity)
                                    ] = None


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
    return nlu_result
