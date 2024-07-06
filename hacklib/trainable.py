import json
import os

import pickle
from ray import tune, train

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
from .data import load_data
from .models import select_model
from .explainers import select_explainer, meta_explain


class HackTrainable(tune.Trainable):
    def setup(self, _):
        self.stats_dir = os.path.join(
            "/home/decaro/xai-hack/experiments/kernel/results", self.trial_name
        )
        os.makedirs(self.stats_dir, exist_ok=True)
        print(self.stats_dir, "created")

        print("Loading the data...")
        self.X_train, self.X_test, self.y_train, self.y_test = load_data()
        print("Data loaded")

    def step(self):
        config = self.config

        print(f"Training the model {config['model']}...")
        model = select_model(config["model"])
        model = model.fit(self.X_train, self.y_train)
        y_test_pred = model.predict(self.X_test)
        print(f"Model {config['model']} trained")

        # Concatenate predictions with the test data
        cat_df = self.X_test.copy()
        cat_df["y_true"] = self.y_test
        cat_df["y_pred"] = y_test_pred

        # Explain the model
        print(f"Explaining the model with {config['explainer']}...")
        explain_fn = select_explainer(config["explainer"])
        explained_df = explain_fn(
            model, self.X_train, self.y_train, self.X_test, self.y_test
        )
        cat_df[[f"exp_{col}" for col in explained_df.columns]] = explained_df
        print(f"Model explained with {config['explainer']}")

        # Train "meta"-explainer
        print(
            f"Applying meta-explanation on {config['model']}-{config['explainer']}..."
        )
        meta_results = meta_explain(explained_df)
        print(f"Meta-explanation applied")

        meta_results.update(
            {
                "accuracy": accuracy_score(self.y_test, y_test_pred),
                "precision": precision_score(self.y_test, y_test_pred),
                "recall": recall_score(self.y_test, y_test_pred),
                "f1": f1_score(self.y_test, y_test_pred),
                "roc_auc": roc_auc_score(self.y_test, y_test_pred),
            }
        )

        print(f"Saving results in {self.stats_dir}")
        with open(os.path.join(self.stats_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        cat_df.to_csv(os.path.join(self.stats_dir, "run_df.csv"), index=False)
        with open(os.path.join(self.stats_dir, "meta_results.json"), "w") as f:
            json.dump(meta_results, f)
        print("Results saved")

        return meta_results

    def save_checkpoint(self, tmp_checkpoint_dir):
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        pass


from ray.tune import Stopper


class OneIterStopper(Stopper):
    def __call__(self, trial_id, result):
        return True

    def stop_all(self):
        return False
