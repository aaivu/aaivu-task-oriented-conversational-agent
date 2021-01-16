import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from pipeline_ensemble_preprocessing import StackModels
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV, Lasso
from sklearn.metrics import f1_score
import logging

logging.basicConfig(filename="meta_model_test.log", level=logging.DEBUG)


class IntentEnsembleModel:
    def __init__(self, models, train_data_file, test_data_file):
        self.models = models
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        df_train = pd.read_csv(self.train_data_file)
        self.train_sentences = df_train.Sentence.to_list()
        self.train_targets = df_train.intent.to_list()
        df_test = pd.read_csv(self.test_data_file)
        self.test_sentences = df_test.Sentence.to_list()
        self.test_targets = df_test.intent.to_list()

    @staticmethod
    def create_dataset(self, models, sentences, target_list, dataset_out_file=None):
        data = {}
        intent_list = []
        for model in models:
            result_list = []
            for sentence in sentences:
                parsed_result = models[model].parse(sentence)
                intent = parsed_result["intent"]["name"]
                result_list.append(intent)
                if intent not in intent_list:
                    intent_list.append(intent)
            data[model] = result_list
        for intent in target_list:
            if intent not in intent_list:
                intent_list.append(intent)
        data["targets"] = target_list
        df = pd.DataFrame(data)
        logging.debug("data : {}".format(df))
        for column in df:
            df[column].replace(
                intent_list, value=[i for i in range(len(intent_list))], inplace=True
            )
        # preprocessing.LabelEncoder().fit_transform(df)
        # df.apply(preprocessing.LabelEncoder().fit_transform)
        logging.debug("data after encoding : {}".format(df))
        if dataset_out_file:
            df.to_csv(dataset_out_file, sep=",", index=False)
        return df.targets, df.drop("targets", axis=1)

    def test_ensemble_model(self, classifiers):
        y_train, x_train = self.create_dataset(
            self, self.models, self.train_sentences, self.train_targets
        )
        y_test, x_test = self.create_dataset(
            self, self.models, self.test_sentences, self.test_targets
        )
        for classifier in classifiers:
            print("classifier :", classifier)
            logging.debug("classifier : {}".format(classifier))
            ensemble_model = classifier.fit(x_train, y_train)
            predictions = ensemble_model.predict(x_test)
            print([(y_test[i], predictions[i]) for i in range(len(y_test))])
            logging.debug(
                "predictions : {}".format(
                    [(y_test[i], predictions[i]) for i in range(len(y_test))]
                )
            )
            cm = confusion_matrix(y_test, predictions)
            f1 = f1_score(predictions, y_test, average="macro")
            print("cm :", cm)
            print("f1 :", f1)
            logging.debug("cm : {}, f1: {}".format(cm, f1))
            try:
                accuracy = ensemble_model.score(x_test, y_test)
                print(accuracy)
            except ValueError:
                print("cannot calculate accuracy")


model_folder_path = "nlu\\stack_models\\pipelines"
test_data_file = "nlu\\data\\test_data.csv"
train_data_file = "nlu\\data\\evaluation_data.csv"

stackmodels = StackModels(model_folder_path)
stackmodels = stackmodels.load_models()

intentensemble = IntentEnsembleModel(stackmodels, train_data_file, test_data_file)

classifiers = [
    SVC(kernel="linear", C=1),
    KNeighborsClassifier(n_neighbors=7),
    DecisionTreeClassifier(max_depth=2),
    GaussianNB(),
    LogisticRegression(multi_class="ovr", solver="liblinear"),
    RidgeCV(alphas=np.logspace(-6, 6, 13))
    # Lasso(alpha=0.1),
    # LinearRegression(multi_class='ovr', solver='liblinear')
]

intentensemble.test_ensemble_model(classifiers)
