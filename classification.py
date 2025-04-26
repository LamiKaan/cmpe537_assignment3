import os
import numpy as np
import pickle
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
from itertools import product


# Custom svc class that combines LinearSVC and SVC with Chi-Squared kernel, so we can switch between them during grid search
class MySVC(BaseEstimator, ClassifierMixin):
    def __init__(self, svc_type='LinearSVC', C=1.0):
        self.svc_type = svc_type
        self.C = C
        if svc_type == 'LinearSVC':
            self.model = LinearSVC(C=self.C, random_state=537, verbose=True)
        elif svc_type == 'ChiSquaredSVC':
            self.kernel = chi2_kernel
            self.model = SVC(kernel=self.kernel, C=self.C, random_state=537, cache_size=500, verbose=True)
        else:
            raise ValueError("Unsupported svc_type. Use 'LinearSVC' or 'ChiSquaredSVC'")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self.model)
        return self.model.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.model.score(X=X, y=y, sample_weight=sample_weight)

    @property
    def classes_(self):
        # Return the classes_ attribute of the underlying model
        return self.model.classes_


def mean_f1_scorer(clf, X, y):
    y_pred = clf.predict(X)
    # Calculate mean f1 score
    mean_f1 = f1_score(y, y_pred, average='macro')

    return mean_f1


def get_results(param_grid, cv, model, X, y, X_test, y_test, prefix):
    # Create an iterable of dictionaries with all combinations of parameters
    param_combinations = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]

    # Create variables to hold results
    results = []
    best_result = None

    # For each parameter combination (of C and svc_type)
    for params in param_combinations:
        # Variable to store the results for the current parameters
        params_result = {"params": params}
        C = params['C']
        svc_type = params['svc_type']

        # Initialize model instance with current parameters
        model_instance = model(svc_type=svc_type, C=C)
        # Get cross validation predictions (overall predictions for parameter combination, not per fold) for train set
        y_pred = cross_val_predict(model_instance, X, y, cv=cv)
        model_instance.fit(X, y)

        # Get class labels from the model
        params_result["classes"] = model_instance.classes_

        # Calculate metrics and add to current results
        mean_f1 = f1_score(y, y_pred, average='macro')
        params_result["mean_f1"] = mean_f1

        per_class_f1 = f1_score(y, y_pred, average=None, labels=params_result["classes"])
        params_result["per_class_f1"] = per_class_f1

        cm = confusion_matrix(y, y_pred, labels=params_result["classes"])
        params_result["cm"] = cm

        mean_balanced_accuracy = balanced_accuracy_score(y, y_pred)
        params_result["mean_balanced_accuracy"] = mean_balanced_accuracy

        mean_imbalanced_accuracy = accuracy_score(y, y_pred)
        params_result["mean_imbalanced_accuracy"] = mean_imbalanced_accuracy

        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        params_result["per_class_accuracy"] = per_class_accuracy

        # Check if current parameters give the best result
        if best_result is None:
            best = True
        else:
            if mean_f1 > best_result["mean_f1"]:
                best = True
            else:
                best = False

        # If current parameters belong to the best model
        if best:
            # Make predictions and evaluate metrics also for the test set
            y_test_pred = model_instance.predict(X_test)
            params_result["y_test_pred"] = y_test_pred
            params_result["test_mean_f1"] = f1_score(y_test, y_test_pred, average='macro')
            params_result["test_per_class_f1"] = f1_score(y_test, y_test_pred, average=None, labels=params_result["classes"])
            test_cm = confusion_matrix(y_test, y_test_pred, labels=params_result["classes"])
            params_result["test_cm"] = test_cm
            params_result["test_mean_balanced_accuracy"] = balanced_accuracy_score(y_test, y_test_pred)
            params_result["test_mean_imbalanced_accuracy"] = accuracy_score(y_test, y_test_pred)
            params_result["test_per_class_accuracy"] = test_cm.diagonal() / test_cm.sum(axis=1)

        # Save results for the current parameters
        if best:
            if best_result is not None:
                results.append(best_result)
            best_result = params_result
        else:
            results.append(params_result)

    return results, best_result


def export_to_files(prefix, types, featureExtractor, K, resultsPath, results, best_result, y_test, test_paths):
    results_file_path = os.path.join(resultsPath, f"{prefix}_{featureExtractor}_histograms_{K}_results.txt")

    # Open results file in write mode
    with open(results_file_path, "w") as f:
        # First, write results for non-best parameters (whose metrics are evaluated on train set during grid search)
        f.write("RESULTS OF NON-BEST MODELS ON TRAINING SET:\n\n")

        # For the result of each parameter combination
        for result in results:
            # Get parameters
            params = result["params"]
            C = params["C"]
            svc_type = params["svc_type"]

            f.write(f"Model parameters:\tC={C}\tSVC Type={svc_type}\n")
            # Write mean scores to the file
            f.write(
                f"Mean F1 Score: {result['mean_f1']}\nMean Balanced Accuracy: {result['mean_balanced_accuracy']}\nMean Imbalanced Accuracy: {result['mean_imbalanced_accuracy']}\n\n")

            # Export confusion matrix and per class scores to html files
            # (Those tables can be very large, for example, confusion matrix for turcoins dataset is of size 138x138. It wouldn't fit on page when I tried to include it as a table in the report. I also tried to convert the table into an image, but even in that case, the file resolution and size were very large. So the only method I found to visualize them properly was through exporting them to html, where both the format is readable and file size is manageable.)

            # PER CLASS F1
            # Create dataframe from per class f1 scores
            per_class_f1_df = pd.DataFrame(result["per_class_f1"], index=result["classes"], columns=["Per Class F1 Score"])
            if prefix == "turcoins":
                # Convert indices (class labels) to int from string and sort in increasing order
                per_class_f1_df.index = per_class_f1_df.index.astype(int)
            per_class_f1_df = per_class_f1_df.sort_index()
            # Set html file path
            per_class_f1_html_path = os.path.join(resultsPath,
                                                  f"{prefix}_{types[0]}_{featureExtractor}_histograms_{K}_{svc_type}_{C}_per_class_f1.html")
            # Style the table and export to html
            per_class_f1_df_styled = per_class_f1_df.style.set_table_styles(
                [
                    {'selector': 'td', 'props': [('padding', '10px'), ('border', '1px solid lightgray')]},
                    # Cell padding and borders
                    {'selector': 'th',
                     'props': [('padding', '10px'), ('border', '1px solid lightgray'), ('text-align', 'center')]},
                    # Header styles
                    {'selector': 'table', 'props': [('border-collapse', 'collapse')]}  # Ensure no double borders
                ]
            ).set_properties(**{
                'font-size': '12pt',  # Increase font size (optional)
                'width': '100px',  # Set column width
                'height': '50px'  # Set row height
            }).background_gradient(cmap="BuPu")  # Set colormap for cell backgrounds based on score values
            per_class_f1_df_styled.to_html(per_class_f1_html_path)

            # APPLY SIMILAR PROCESS FOR CONFUSION MATRIX
            cm_df = pd.DataFrame(data=result["cm"], index=result["classes"], columns=result["classes"])
            if prefix == "turcoins":
                cm_df.index = cm_df.index.astype(int)
                cm_df.columns = cm_df.columns.astype(int)
            cm_df = cm_df.sort_index()
            cm_df = cm_df.sort_index(axis=1)
            cm_html_path = os.path.join(resultsPath,
                                        f"{prefix}_{types[0]}_{featureExtractor}_histograms_{K}_{svc_type}_{C}_confusion_matrix.html")
            cm_df_styled = cm_df.style.set_table_styles(
                [
                    {'selector': 'td', 'props': [('padding', '10px'), ('border', '1px solid lightgray')]},
                    {'selector': 'th',
                     'props': [('padding', '10px'), ('border', '1px solid lightgray'), ('text-align', 'center')]},
                    {'selector': 'table', 'props': [('border-collapse', 'collapse')]}
                ]
            ).set_properties(**{
                'font-size': '12pt',
                'width': '100px',
                'height': '50px'
            }).background_gradient(cmap="BuPu")
            cm_df_styled.to_html(cm_html_path)

            # ALSO APPLY FOR PER CLASS ACCURACY SCORES
            per_class_accuracy_df = pd.DataFrame(data=result["per_class_accuracy"], index=result["classes"],
                                                 columns=["Per Class Accuracy"])
            if prefix == "turcoins":
                per_class_accuracy_df.index = per_class_accuracy_df.index.astype(int)
            per_class_accuracy_df = per_class_accuracy_df.sort_index()
            per_class_accuracy_html_path = os.path.join(resultsPath,
                                                        f"{prefix}_{types[0]}_{featureExtractor}_histograms_{K}_{svc_type}_{C}_per_class_accuracy.html")
            # Style the table and export to html
            per_class_accuracy_df_styled = per_class_accuracy_df.style.set_table_styles(
                [
                    {'selector': 'td', 'props': [('padding', '10px'), ('border', '1px solid lightgray')]},
                    {'selector': 'th',
                     'props': [('padding', '10px'), ('border', '1px solid lightgray'), ('text-align', 'center')]},
                    {'selector': 'table', 'props': [('border-collapse', 'collapse')]}
                ]
            ).set_properties(**{
                'font-size': '12pt',
                'width': '100px',
                'height': '50px'
            }).background_gradient(cmap="BuPu")
            per_class_accuracy_df_styled.to_html(per_class_accuracy_html_path)

        # Then, write the results for the best model
        f.write("\nRESULTS OF BEST MODEL:\n\n")
        # Get parameters
        params = best_result["params"]
        C = params["C"]
        svc_type = params["svc_type"]
        f.write(f"Model parameters:\tC={C}\tSVC Type={svc_type}\n\n")

        # First, write results of the best model on the training set
        f.write("Results on the training set:\n")
        f.write(
            f"Mean F1 Score: {best_result['mean_f1']}\nMean Balanced Accuracy: {best_result['mean_balanced_accuracy']}\nMean Imbalanced Accuracy: {best_result['mean_imbalanced_accuracy']}\n\n")

        # Again, export confusion matrix and per class scores to html files
        # PER CLASS F1
        per_class_f1_df = pd.DataFrame(best_result["per_class_f1"], index=best_result["classes"], columns=["Per Class F1 Score"])
        if prefix == "turcoins":
            per_class_f1_df.index = per_class_f1_df.index.astype(int)
        per_class_f1_df = per_class_f1_df.sort_index()
        per_class_f1_html_path = os.path.join(resultsPath,
                                              f"{prefix}_{types[0]}_{featureExtractor}_histograms_{K}_{svc_type}_{C}_per_class_f1_BEST.html")
        per_class_f1_df_styled = per_class_f1_df.style.set_table_styles(
            [
                {'selector': 'td', 'props': [('padding', '10px'), ('border', '1px solid lightgray')]},
                {'selector': 'th',
                 'props': [('padding', '10px'), ('border', '1px solid lightgray'), ('text-align', 'center')]},
                {'selector': 'table', 'props': [('border-collapse', 'collapse')]}
            ]
        ).set_properties(**{
            'font-size': '12pt',
            'width': '100px',
            'height': '50px'
        }).background_gradient(cmap="BuPu")
        per_class_f1_df_styled.to_html(per_class_f1_html_path)
        # CONFUSION MATRIX
        cm_df = pd.DataFrame(data=best_result["cm"], index=best_result["classes"], columns=best_result["classes"])
        if prefix == "turcoins":
            cm_df.index = cm_df.index.astype(int)
            cm_df.columns = cm_df.columns.astype(int)
        cm_df = cm_df.sort_index()
        cm_df = cm_df.sort_index(axis=1)
        cm_html_path = os.path.join(resultsPath,
                                    f"{prefix}_{types[0]}_{featureExtractor}_histograms_{K}_{svc_type}_{C}_confusion_matrix_BEST.html")
        cm_df_styled = cm_df.style.set_table_styles(
            [
                {'selector': 'td', 'props': [('padding', '10px'), ('border', '1px solid lightgray')]},
                {'selector': 'th',
                 'props': [('padding', '10px'), ('border', '1px solid lightgray'), ('text-align', 'center')]},
                {'selector': 'table', 'props': [('border-collapse', 'collapse')]}
            ]
        ).set_properties(**{
            'font-size': '12pt',
            'width': '100px',
            'height': '50px'
        }).background_gradient(cmap="BuPu")
        cm_df_styled.to_html(cm_html_path)
        # PER CLASS ACCURACY SCORES
        per_class_accuracy_df = pd.DataFrame(data=best_result["per_class_accuracy"], index=best_result["classes"],
                                             columns=["Per Class Accuracy"])
        if prefix == "turcoins":
            per_class_accuracy_df.index = per_class_accuracy_df.index.astype(int)
        per_class_accuracy_df = per_class_accuracy_df.sort_index()
        per_class_accuracy_html_path = os.path.join(resultsPath,
                                                    f"{prefix}_{types[0]}_{featureExtractor}_histograms_{K}_{svc_type}_{C}_per_class_accuracy_BEST.html")
        per_class_accuracy_df_styled = per_class_accuracy_df.style.set_table_styles(
            [
                {'selector': 'td', 'props': [('padding', '10px'), ('border', '1px solid lightgray')]},
                {'selector': 'th',
                 'props': [('padding', '10px'), ('border', '1px solid lightgray'), ('text-align', 'center')]},
                {'selector': 'table', 'props': [('border-collapse', 'collapse')]}
            ]
        ).set_properties(**{
            'font-size': '12pt',
            'width': '100px',
            'height': '50px'
        }).background_gradient(cmap="BuPu")
        per_class_accuracy_df_styled.to_html(per_class_accuracy_html_path)

        # Secondly, write results of the best model on the test set too
        f.write("Results on the test set:\n")
        f.write(
            f"Mean F1 Score: {best_result['test_mean_f1']}\nMean Balanced Accuracy: {best_result['test_mean_balanced_accuracy']}\nMean Imbalanced Accuracy: {best_result['test_mean_imbalanced_accuracy']}\n\n")

        # Export the confusion matrix and per class scores for the test set to html
        # PER CLASS F1
        per_class_f1_df = pd.DataFrame(best_result["test_per_class_f1"], index=best_result["classes"],
                                       columns=["Per Class F1 Score"])
        if prefix == "turcoins":
            per_class_f1_df.index = per_class_f1_df.index.astype(int)
        per_class_f1_df = per_class_f1_df.sort_index()
        per_class_f1_html_path = os.path.join(resultsPath,
                                              f"{prefix}_{types[1]}_{featureExtractor}_histograms_{K}_{svc_type}_{C}_per_class_f1_BEST.html")
        per_class_f1_df_styled = per_class_f1_df.style.set_table_styles(
            [
                {'selector': 'td', 'props': [('padding', '10px'), ('border', '1px solid lightgray')]},
                {'selector': 'th',
                 'props': [('padding', '10px'), ('border', '1px solid lightgray'), ('text-align', 'center')]},
                {'selector': 'table', 'props': [('border-collapse', 'collapse')]}
            ]
        ).set_properties(**{
            'font-size': '12pt',
            'width': '100px',
            'height': '50px'
        }).background_gradient(cmap="BuPu")
        per_class_f1_df_styled.to_html(per_class_f1_html_path)
        # CONFUSION MATRIX
        cm_df = pd.DataFrame(data=best_result["test_cm"], index=best_result["classes"], columns=best_result["classes"])
        if prefix == "turcoins":
            cm_df.index = cm_df.index.astype(int)
            cm_df.columns = cm_df.columns.astype(int)
        cm_df = cm_df.sort_index()
        cm_df = cm_df.sort_index(axis=1)
        cm_html_path = os.path.join(resultsPath,
                                    f"{prefix}_{types[1]}_{featureExtractor}_histograms_{K}_{svc_type}_{C}_confusion_matrix_BEST.html")
        cm_df_styled = cm_df.style.set_table_styles(
            [
                {'selector': 'td', 'props': [('padding', '10px'), ('border', '1px solid lightgray')]},
                {'selector': 'th',
                 'props': [('padding', '10px'), ('border', '1px solid lightgray'), ('text-align', 'center')]},
                {'selector': 'table', 'props': [('border-collapse', 'collapse')]}
            ]
        ).set_properties(**{
            'font-size': '12pt',
            'width': '100px',
            'height': '50px'
        }).background_gradient(cmap="BuPu")
        cm_df_styled.to_html(cm_html_path)
        # PER CLASS ACCURACY SCORES
        per_class_accuracy_df = pd.DataFrame(data=best_result["test_per_class_accuracy"], index=best_result["classes"],
                                             columns=["Per Class Accuracy"])
        if prefix == "turcoins":
            per_class_accuracy_df.index = per_class_accuracy_df.index.astype(int)
        per_class_accuracy_df = per_class_accuracy_df.sort_index()
        per_class_accuracy_html_path = os.path.join(resultsPath,
                                                    f"{prefix}_{types[1]}_{featureExtractor}_histograms_{K}_{svc_type}_{C}_per_class_accuracy_BEST.html")
        per_class_accuracy_df_styled = per_class_accuracy_df.style.set_table_styles(
            [
                {'selector': 'td', 'props': [('padding', '10px'), ('border', '1px solid lightgray')]},
                {'selector': 'th',
                 'props': [('padding', '10px'), ('border', '1px solid lightgray'), ('text-align', 'center')]},
                {'selector': 'table', 'props': [('border-collapse', 'collapse')]}
            ]
        ).set_properties(**{
            'font-size': '12pt',
            'width': '100px',
            'height': '50px'
        }).background_gradient(cmap="BuPu")
        per_class_accuracy_df_styled.to_html(per_class_accuracy_html_path)

        # Get indices of test samples for which the predicted classes are incorrect
        y_test_pred = best_result["y_test_pred"]
        example_indices = np.where((y_test == y_test_pred) == False)[0]
        # Select a maximum of 10 misclassified example images randomly
        if len(example_indices) > 10:
            example_indices = np.random.choice(example_indices, size=10, replace=False)
        # Write misclassified example images info to the results file
        f.write("Example misclassified test images:\n")
        for index in example_indices:
            f.write(
                f"True class:{y_test[index]} - Predicted Class:{y_test_pred[index]} - Image Path:{test_paths[index]}\n")
        f.write("\n")

        # Lastly, report 5 classes for which the model shows the best and worst performance
        per_class_f1_df_class_sorted = per_class_f1_df.sort_values(by="Per Class F1 Score", ascending=False)
        f.write("Best 5 classes:\n")
        f.write("Class - Per Class F1 Score\n")
        for index, row in per_class_f1_df_class_sorted.iloc[0:5].iterrows():
            f.write(f"{index}\t{row['Per Class F1 Score']}\n")
        f.write("\n")
        f.write("Worst 5 classes:\n")
        f.write("Class - Per Class F1 Score\n")
        for index, row in per_class_f1_df_class_sorted.iloc[-5:].iterrows():
            f.write(f"{index}\t{row['Per Class F1 Score']}\n")


def main(prefix, types, featureExtractors, Ks, producedDataPath, resultsPath):
    y_train_path = os.path.join(producedDataPath, f"{prefix}_{types[0]}_class_labels.pkl")
    y_test_path = os.path.join(producedDataPath, f"{prefix}_{types[1]}_class_labels.pkl")

    test_paths_path = os.path.join(producedDataPath, f"{prefix}_{types[1]}_image_paths.pkl")

    with open(y_train_path, "rb") as f:
        y_train = pickle.load(f)

    with open(y_test_path, "rb") as f:
        y_test = pickle.load(f)

    with open(test_paths_path, "rb") as f:
        test_image_paths = pickle.load(f)

    for K in Ks:
        for featureExtractor in featureExtractors:

            X_train_path = os.path.join(producedDataPath, f"{prefix}_{types[0]}_{featureExtractor}_histograms_{K}.pkl")
            X_test_path = os.path.join(producedDataPath, f"{prefix}_{types[1]}_{featureExtractor}_histograms_{K}.pkl")

            # Get data from pickle files
            with open(X_train_path, "rb") as f:
                X_train = pickle.load(f)

            with open(X_test_path, "rb") as f:
                X_test = pickle.load(f)

            # Define parameter grid
            param_grid = {"svc_type": ["LinearSVC", "ChiSquaredSVC"],
                          "C": [10, 1, 0.1, 0.01, 0.001]}
            # Evaluate performance of cross validated model based on the mean f1 score
            scoring = {"mean_f1": mean_f1_scorer}
            # Stratified K fold (create folds considering consistent class distribution in each fold)
            skf = StratifiedKFold(n_splits=5)
            # Create grid search object
            # gscv = GridSearchCV(estimator=MySVC(), param_grid=param_grid, scoring=scoring, refit="mean_f1", cv=skf,
            #                     verbose=True,
            #                     return_train_score=False) // GridSearchCV function produced exactly the same scores for both LinearSVC and ChiSquaredSVC and I couldn't
            #                                               //figure out why. So, instead, I decided to use cross_val_predict function instead, for which the result were more meaningful.

            # Stack all train image histograms into a single matrix for model input
            X = np.vstack(X_train)
            y = y_train
            # Perform cross validation
            # gscv.fit(X, y)

            # Get cross validation results
            results, best_result = get_results(param_grid, skf, MySVC, X, y, X_test, y_test, prefix)
            # Write results to files
            export_to_files(prefix, types, featureExtractor, K, resultsPath, results, best_result, y_test,
                            test_image_paths)



if __name__ == '__main__':
    # Initial variables to set the context of model
    prefix = "turcoins"
    types = ["train", "test"]
    featureExtractors = ["sift", "hynet"]
    Ks = [100, 50]

    producedDataPath = "/Users/lkk/Documents/BOUN CMPE/CMPE 537-Computer Vision/Assignment3/ProducedData/100_64_sklearn"
    resultsPath = "/Users/lkk/Documents/BOUN CMPE/CMPE 537-Computer Vision/Assignment3/ProducedData/100_64_sklearn_results"

    # Train classifiers and save classification results
    main(prefix, types, featureExtractors, Ks, producedDataPath, resultsPath)


