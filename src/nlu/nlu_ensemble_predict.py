import pipeline_ensemble as pe

model_folder_path = "stack_models\pipelines"
model_score_path = "results\pipelines"


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
        modelscores = pe.ModelScores(model_score_path)
        self.stack_models = stackmodels.load_models()
        self.model_scores = modelscores.get_scores(trim_count=trim_count)
        self.value_added_scores = modelscores.get_value_added_scores()

    def run_stack_pipelines(self, sentence):
        return pe.model_stack_predict(sentence, self.stack_models, self.model_scores)

