from ray import tune, train
from hacklib.trainable import HackTrainable, OneIterStopper


def main():
    search_space = {
        "model": tune.grid_search(
            [
                "LogisticRegression",
                "RandomForest",
                "GradientBoosting",
                "AdaBoost",
                "ExtraTrees",
                "GaussianNB",
                "KNeighbors",
                "SVC",
                "DecisionTree",
                "LDA",
                "LGBM",
            ]
        ),
        "explainer": tune.grid_search(["shap", "lime"]),
    }

    run_config = train.RunConfig(
        "grid_search",
        stop=OneIterStopper(),
        storage_path="/home/decaro/xai-hack/experiments/kernel",
    )

    tune_config = tune.TuneConfig(
        num_samples=1,
        trial_name_creator=lambda trial: f"{trial.config['model']}_{trial.config['explainer']}",
    )
    tuner = tune.Tuner(
        tune.with_resources(HackTrainable, {"cpu": 1}),
        param_space=search_space,
        run_config=run_config,
        tune_config=tune_config,
    )

    tuner.fit()


if __name__ == "__main__":
    main()
