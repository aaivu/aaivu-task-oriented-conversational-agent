import os
from os import listdir
from os.path import isfile, join
import glob
import argparse

from rasa.cli.test import run_nlu_test
from rasa.test import perform_nlu_cross_validation
from rasa.shared.data import get_nlu_directory

args = argparse.Namespace
# onlyfiles = [f for f in listdir("pipelines/") if isfile(join("pipelines/", f))]
# print (onlyfiles)
pipelines = glob.glob("pipelines\\*")
nlu_data = get_nlu_directory("data")
additional_args = {
    "loglevel": None,
    "model": "models",
    "stories": "tests",
    "max_stories": None,
    "endpoints": None,
    "fail_on_prediction_errors": False,
    "url": None,
    "evaluate_model_directory": False,
    "nlu": "data",
    "config": None,
    "cross_validation": True,
    "folds": 5,
    "runs": 3,
    "percentages": [0, 25, 50, 75],
    "disable_plotting": False,
    "successes": False,
    "no_errors": False,
    "out": "results",
    "errors": True,
}


for i in range(len(pipelines)):
    try:
        os.system(
            "rasa train nlu --config {} --out stack_models\\{}".format(
                pipelines[i], pipelines[i]
            )
        )
        perform_nlu_cross_validation(
            "{}".format(pipelines[i]), nlu_data, "results\\{}".format(pipelines[i]), {}
        )
    except OSError:
        print("{} cannot find the model".format(pipelines[i]))
        continue
