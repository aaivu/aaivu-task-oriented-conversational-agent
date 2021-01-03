import csv, numpy
import pandas as pd
from math import isnan

from rasa.test import get_evaluation_metrics
from pipeline_ensemble import test_stack_pipelines, test_pipelines
from pipeline_info import DictComparison

# itargets = ('greeting', 'greeting', 'greeting', 'confirm_answer', 'deny', 'deny', 'thanks', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'ticket', 'ticket', 'moviename', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'date', 'date', 'date', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'date+starttime', 'date+starttime', 'date+starttime', 'greeting', 'greeting', 'greeting', 'confirm_answer', 'deny', 'deny', 'thanks', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'ticket', 'ticket', 'moviename', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'date', 'date', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'date+starttime', 'date+starttime', 'date+starttime', 'greeting', 'greeting', 'greeting', 'confirm_answer', 'deny', 'thanks', 'thanks', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'ticket', 'ticket', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'date', 'date', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'date+starttime', 'date+starttime', 'date+starttime', 'greeting', 'greeting', 'confirm_answer', 'confirm_answer', 'deny', 'thanks', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'ticket', 'ticket', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'date', 'date', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'date+starttime', 'date+starttime', 'date+starttime', 'greeting', 'greeting', 'confirm_answer', 'deny', 'deny', 'thanks', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'ticket', 'ticket', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'date', 'date', 'date', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'date+starttime', 'date+starttime', 'date+starttime')
# ipredictions =  ('greeting', 'nlu_fallback', 'greeting', 'deny', 'deny', 'deny', 'greeting', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'nlu_fallback', 'inform', 'nlu_fallback', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'nlu_fallback', 'starttime', 'nlu_fallback', 'nlu_fallback', 'theater', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'nlu_fallback', 'date+starttime', 'date+starttime', 'greeting', 'greeting', 'inform', 'deny', 'nlu_fallback', 'deny', 'nlu_fallback', 'inform', 'theater', 'inform', 'inform', 'nlu_fallback', 'inform', 'inform', 'theater', 'theater', 'ticket', 'theater', 'theater', 'ticket', 'nlu_fallback', 'nlu_fallback', 'nlu_fallback', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'date', 'date', 'theater+starttime', 'theater+starttime', 'nlu_fallback', 'theater+starttime', 'date+starttime', 'date+starttime', 'date+starttime', 'greeting', 'greeting', 'greeting', 'greeting', 'nlu_fallback', 'nlu_fallback', 'greeting', 'nlu_fallback', 'ticket', 'inform', 'nlu_fallback', 'inform', 'nlu_fallback', 'inform', 'nlu_fallback', 'nlu_fallback', 'theater', 'theater', 'theater', 'ticket', 'inform', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'date', 'starttime', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'date+starttime', 'theater+starttime', 'starttime', 'greeting', 'greeting', 'deny', 'nlu_fallback', 'deny', 'thanks', 'inform', 'greeting', 'nlu_fallback', 'inform', 'inform', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'nlu_fallback', 'nlu_fallback', 'moviename', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'nlu_fallback', 'nlu_fallback', 'date', 'date', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'theater+starttime', 'date+starttime', 'date+starttime', 'date+starttime', 'greeting', 'greeting', 'nlu_fallback', 'inform', 'deny', 'nlu_fallback', 'inform', 'inform', 'inform', 'inform', 'nlu_fallback', 'inform', 'inform', 'inform', 'theater', 'theater', 'theater', 'theater', 'theater', 'ticket', 'nlu_fallback', 'theater', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'starttime', 'nlu_fallback', 'starttime', 'theater+starttime', 'theater+starttime', 'theater', 'nlu_fallback', 'date+starttime', 'date+starttime')



df = pd.read_csv("nlu\\data\\test_data.csv")
sentences = df.Sentence.to_list()
itargets = df.intent.to_list()


def get_target_entities():
    target_entities = []

    for index, row in df.iterrows():
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


target_entities = get_target_entities()
print("target_entities: ", target_entities)

single_models = [
    "stack_models\\pipelines\\config_1.yml\\nlu-20201118-182926.tar.gz",
    "stack_models\\pipelines\\config_2.yml\\nlu-20201118-183729.tar.gz",
    "stack_models\\pipelines\\config_3.yml\\nlu-20201119-020428.tar.gz",
    "stack_models\\pipelines\\config_4.yml\\nlu-20201119-023646.tar.gz",
    "stack_models\\pipelines\\config_5.yml\\nlu-20201119-030659.tar.gz",
    "stack_models\\pipelines\\config_7.yml\\nlu-20201119-152018.tar.gz",
    "stack_models\\pipelines\\config_8.yml\\nlu-20201119-124504.tar.gz",
    "stack_models\\pipelines\\config_9.yml\\nlu-20201119-130525.tar.gz",
]

# ipredictions_intent,ipredictions_entities = test_pipelines(sentences, single_models[2])
ipredictions_intent, ipredictions_entities = test_stack_pipelines(
    sentences,
    "results\\stack_reports\\with_entity5.csv",
    trim_count=1,
)
# print(ipredictions_entities)
# print(len(ipredictions_intent), len(itargets))
# get_evaluation_metrics(itargets, ipredictions_intent)

# targets = [{'theater': 'liberty'}, {'numberofpeople': '4', 'date': 'tomorrow'}, {'moviename': 'infinity war', 'starttime': '4.30 p.m.'}, {'numberofpeople': '3', 'moviename': 'Zootopia'}, {'genre': 'comedy', 'city': 'nyc'}, {'moviename': 'Deadpool ', 'city': 'hamilton'}, {'numberofpeople': '3', 'moviename': 'batman'}, {'moviename': 'zootopis'}, {'moviename': 'batman vs superman'}, {}, {'theater': 'amc pacific place 11'}, {'moviename': "brother's grimsby", 'theater': 'emagine ', 'city': 'Portland'}, {'numberofpeople': '4', 'city': 'chicago'}, {'moviename': 'Foxtrot', 'city': 'seattle'}, {'city': 'nyc'}, {'city': 'san francisco'}, {'moviename': 'batman', 'city': 'miami'}, {'city': 'dallas'}, {}, {'starttime': '10'}, {'genre': 'action', 'date': 'tomorrow'}, {'starttime': 'before 12pm'}, {}, {}, {}, {}, {'video_format': '3d', 'moviename': 'Avengers', 'city': 'atlanta'}, {'city': 'nyc'}, {'moviename': 'DeadPool'}, {'starttime': 'DONT CARE'}, {'starttime': '4.00 pm', 'moviename': 'Antman', 'numberofpeople': '2'}, {}, {'numberofpeople': '3', 'moviename': 'batman'}, {'numberofpeople': '7', 'moviename': 'magnificent', 'starttime': '8.00 pm'}, {}, {}, {}, {}, {'moviename': 'Black swan'}, {}, {}, {'moviename': 'bird box', 'city': 'portland'}, {'moviename': 'quiet place', 'starttime': '12.00 pm', 'theater': 'los angeles'}, {}, {'theater': 'liberty'}, {}, {'starttime': '8.00 pm', 'date': 'tomorrow'}, {'starttime': 'DONT CARE'}, {'moviename': 'Dealpool', 'theater': 'liberty'}, {}, {}, {}, {}, {}, {}, {}, {}, {'genre': 'horror', 'city': 'san francisco'}, {}, {}, {'video_format': '3d'}, {}, {'numberofpeople': '2', 'theater': 'emagine'}, {'date': 'tomorrow'}, {}, {'starttime': '4 p.m.'}, {}, {}, {}, {'moviename': 'avengers'}]
# predictions =[{'state': 'liberty', 'theater': 'liberty', 'city': 'liberty', 'distanceconstraints': 'liberty'}, {'numberofpeople': '4 people', 'date': 'tomorrow', 'moviename': '4 people'}, {'moviename': 'infinity war'}, {'numberofpeople': '3', 'moviename': 'zootopia'}, {'genre': 'comedy', 'city': 'nyc'}, {'moviename': 'Deadpool', 'city': 'hamilton'}, {'numberofpeople': '3', 'moviename': 'batman', 'city': 'los angeles', 'state': 'angeles'}, {'moviename': 'zootopis'}, {'numberofpeople': '8', 'moviename': 'superman'}, {}, {'theater_chain': 'amc', 'theater': 'place 11', 'moviename': 'amc pacific', 'city': 'amc pacific', 'date': '11'}, {'moviename': "the brother's grimsby", 'theater': 'emagine', 'city': 'Portland'}, {'numberofpeople': '4 people', 'city': 'chicago', 'moviename': 'people'}, {'moviename': 'Foxtrot', 'city': 'seattle'}, {'city': 'nyc'}, {'city': 'san francisco'}, {'city': 'miami', 'moviename': 'batman', 'video_format': 'batman'}, {'city': 'dallas'}, {}, {'city': 'nyc', 'starttime': '10', 'zip': '10', 'date': '10'}, {'date': 'tomorrow', 'genre': 'action'}, {'starttime': 'before12pm'}, {}, {'critic_rating': 'Good'}, {}, {}, {'video_format': '3d Avengers', 'moviename': 'Avengers', 'city': 'atlanta'}, {'city': 'nyc'}, {'moviename': 'DeadPool'}, {'theater': 'any'}, {'numberofpeople': '2'}, {}, {'numberofpeople': '3', 'moviename': '3'}, {'numberofpeople': 'magnificent', 'moviename': 'magnificent', 'starttime': 'pm'}, {}, {'date': 'me', 'distanceconstraints': 'me'}, {}, {}, {'moviename': 'the'}, {}, {}, {'moviename': 'bird', 'theater': 'box moive', 'city': 'portland'}, {'starttime': '12:00 pm', 'city': 'los angeles', 'closing': 'see', 'theater': 'place', 'distanceconstraints': 'place', 'state': 'angeles'}, {'closing': 'Bye', 'critic_rating': 'nice', 'moviename': 'work'}, {'state': 'liberty', 'theater': 'liberty', 'city': 'liberty'}, {'critic_rating': 'good'}, {'date': 'tomorrow', 'theater': 'pm', 'starttime': 'pm'}, {}, {'moviename': 'liberty', 'theater': 'liberty'}, {}, {}, {}, {}, {}, {}, {'critic_rating': 'Good'}, {}, {'genre': 'horror', 'city': 'san francisco', 'moviename': 'san francisco'}, {'distanceconstraints': 'time'}, {}, {'video_format': '3d'}, {}, {'numberofpeople': '2', 'theater': 'emagine', 'numberofkids': '2'}, {'date': 'tomorrow'}, {}, {'numberofpeople': '4', 'starttime': '4'}, {}, {}, {'genre': 'kids'}, {'moviename': 'avengers'}]
# print("len",len(target_entities),len(ipredictions_entities))
dict_comparison = DictComparison(
    target_entities,
    ipredictions_entities,
    out_file_path="results\\stack_entity_reports\\entity_result.txt",
)
print(dict_comparison.recall(), dict_comparison.precision(), dict_comparison.f1score())

