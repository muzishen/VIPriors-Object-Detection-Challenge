# -*- coding: utf-8 -*-
from __future__ import print_function  # do not delete this line if you want to save your log file.

def load_data():
    x_train = None
    y_train = None
    x_validation = None
    y_validation = None
    """
    ====================== YOUR CODE HERE ======================
    Using softcomai data reference api to read data set, read sdk docs for more details.
    e.g.
    from naie.datasets import get_data_reference
    data_reference = get_data_reference(dataset="any_dataset", dataset_entity="entity_of_dataset")
    df = data_reference.to_pandas_dataframe()

    or

    file_paths = data_reference.get_files_paths() # to get data files full path list
    ============================================================
    Parameters
    ----------
    dataset : name of dataset
    dataset_entity : name of dataset entity
    """
    return x_train, y_train, x_validation, y_validation


def model_fn():
    model = None
    """
    ====================== YOUR CODE HERE ======================
    you can write your model function here.
    Example:
    model = RFC(
        n_estimators=int(n_estimators),
        min_samples_split=int(min_samples_split),
        max_features=max(min(max_features, 0.999), 1e-3),
        random_state=2
        )
    ============================================================
    """
    return model


def train(x_train, y_train, model):
    """
    ====================== YOUR CODE HERE ======================
    you can write the main process here.
    there are several api you can use here.
    Example:
    model.fit(x_train, y_train)
    ============================================================
    """
    pass


def save_model(model):
    """
    ====================== YOUR CODE HERE ======================
    write model to the specific model path of train job

    e.g.
    from naie.context import Context
    with open(os.path.join(Context.get_output_path(), 'model.pkl'), 'w') as ff:
        pickle.dump(clf, ff)
    or
    tf.estimator.Estimator(model_dir=Context.get_output_path())  # using tensorflow Estimator
    ============================================================
    """
    pass


def score_model(x_validation, y_validation, model):
    score = None
    """
    ====================== YOUR CODE HERE ======================
    there are several api you can use here.
    Example:
    from naie.metrics import report
    with report(True) as log_report:
        log_report.log_property("score", accuracy_score(y_validation, model.predict(x_validation)))
    ============================================================
    """
    return score


def main():
    """
    ====================== YOUR CODE HERE ======================
    you can write the main process here.
    ============================================================
    """

    x_train, y_train, x_validation, y_validation = load_data()
    model = model_fn()
    train(x_train, y_train, model)
    score = score_model(x_validation, y_validation, model)
    save_model(model)

    # return the score for hyperparameter tuning
    return score

if __name__ == "__main__":
    main()
