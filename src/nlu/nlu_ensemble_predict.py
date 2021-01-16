import pipeline_ensemble_preprocessing as pe

model_folder_path = "nlu\\stack_models\\pipelines"
model_score_path = "nlu\\results\\pipelines"
evaluation_dataset_path = "nlu\\data\\test_data.csv"


class NluEnsemblePredict:
    __instance = None

    def __init__(
        self,
        model_folder_path=model_folder_path,
        model_score_path=model_score_path,
        trim_count=1,
    ):
        if NluEnsemblePredict.__instance is None:
            NluEnsemblePredict.__instance = object.__init__(self)
        stackmodels = pe.StackModels(model_folder_path)
        modelscores = pe.ModelScores(
            model_score_path,
            model_folder_path=model_folder_path,
            evaluation_dataset_path=evaluation_dataset_path,
        )

        self.stack_models = stackmodels.load_models()
        self.model_scores = modelscores.get_scores(trim_count=trim_count)
        self.model_entity_scores = modelscores.get_model_evaluation_scores()
        print("self.model_entity_scores :", self.model_entity_scores)
        self.value_added_scores = modelscores.get_value_added_scores()

    def run_stack_pipelines(self, sentence):
        return pe.model_stack_predict(
            sentence, self.stack_models, self.model_scores, self.model_entity_scores
        )

