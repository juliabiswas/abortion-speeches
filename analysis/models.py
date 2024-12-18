'''training & evaluating logistic regression, svm, naive bayes, random forest, ada boost, neural networks'''

import os
import pickle
from optparse import OptionParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from scikeras.wrappers import KerasClassifier
import seaborn as sns

def main():
    indir = '../data'

    models = [
        ('logistic_regression', LogisticRegression(max_iter=200)),
        ('svm', LinearSVC()),
        ('naive_bayes', MultinomialNB()),
        ('random_forest', RandomForestClassifier(random_state=42)),
        ('ada_boost', AdaBoostClassifier(algorithm='SAMME', random_state=42)),
        ('nn_one_relu', None),
        ('nn_one_relu_dropout', None),
        ('nn_two_relu', None),
        ('nn_two_relu_dropout', None)
    ]
    
    infiles = {
        'bag_of_words1': models[:2],
        'tf_idf1': models,
        'custom_train': models[:2],
        **{f'tf_idf{i}': models[1:2] for i in range(2, 6)}
    }

    for infile in infiles.keys():
        print(f"for {infile}...")

        data = pd.read_csv(os.path.join(indir, f"{infile}.csv"))
        data = data.drop(columns=['id'])

        curr_models = infiles[infile]
        outfiles = [f"../models/{infile}-{m[0]}.pkl" for m in curr_models]
    
        if 'custom' in infile:
            data['vector'] = data['vector'].apply(lambda x: list(map(float, x.strip('[]').split())))
            x = MinMaxScaler().fit_transform(np.vstack(data['vector'].values))
        else:
            x = MinMaxScaler().fit_transform(data.drop(columns=['labeled_class']))
        y = data['labeled_class']
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        
        y_pred_best_baseline = None
        y_pred_best = None
        
        for (name, model), outfile in zip(curr_models, outfiles):
            print(f"evaluating {name}...")

            if name == 'random_forest' or name == 'ada_boost' or 'nn' in name:
                if 'nn' in name:
                    param_grid = {
                        'batch_size': [8, 16, 32, 64],
                        'epochs': [3, 5, 7, 9, 11, 13, 15],
                        'optimizer': ['adam', 'sgd']
                    }
                    model = KerasClassifier(model=build_neural_network, input_dim=x_train.shape[1], num_layers=1 if 'one' in name else 2, dropout_rate=0.2 if 'dropout' in name else 0.0, verbose=0)
                            
                    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1)
                    grid_search.fit(x_train, y_train)

                    best_params = grid_search.best_params_
                    print(f"best hyperparameters for {name}: {best_params}")

                    best_model = grid_search.best_estimator_
                
                else:
                    param_grid = {
                        'n_estimators': [50, 100, 150, 200, 250],
                    }
                    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1)
                    grid_search.fit(x_train, y_train)

                    best_params = grid_search.best_params_
                    print(f"best hyperparameters for {name}: {best_params}")

                    best_model = grid_search.best_estimator_

            else:  # no hyperparam tuning
                cv_accuracy = cross_val_score(model, x_train, y_train, cv=10, scoring=make_scorer(accuracy_score)).mean()
                print(f"{name} - validation accuracy averaged across 10 folds: {cv_accuracy:.4f}")
                
                cv_mse = cross_val_score(model, x_train, y_train, cv=10, scoring=make_scorer(mean_squared_error)).mean()
                print(f"{name} - validation mse averaged across 10 folds: {cv_mse:.4f}")
                
                model.fit(x_train, y_train)
                best_model = model
  
            y_train_pred = best_model.predict(x_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_mse_val = mean_squared_error(y_train, y_train_pred)
            y_test_pred = best_model.predict(x_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_mse_val = mean_squared_error(y_test, y_test_pred)
            
            print(f"train accuracy for {name}: {train_accuracy:.4f}")
            print(f"train mse for {name}: {train_mse_val:.4f}")
            print(f"test accuracy for {name}: {test_accuracy:.4f}")
            print(f"test mse for {name}: {test_mse_val:.4f}")
            
            best_model.fit(x, y) #fitting on all data now

            with open(outfile, "wb") as f:
                pickle.dump(best_model, f)
            
            if infile == 'tf_idf1':
                if name == 'svm':
                    y_pred_best_baseline = y_test_pred
                elif name == 'nn_two_relu':
                    y_pred_best = y_test_pred
                    plot_accuracy_vs_epoch(best_model.fit(x_train, y_train, validation_split=0.1, epochs=15), infile, name)

        if infile == 'tf_idf1':
            plot_confusion_matrix(y_test, y_pred_best_baseline, 'svm', 'tf_idf1')
            plot_confusion_matrix(y_test, y_pred_best, 'nn_two_relu', 'tf_idf1')

def build_neural_network(input_dim, num_layers=1, optimizer='adam', dropout_rate=0.0, units=128):
    model = Sequential()
    model.add(Dense(units, input_dim=input_dim, activation='relu'))
    
    for _ in range(num_layers - 1):
        model.add(Dense(units, activation='relu'))
    
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def plot_confusion_matrix(y_true, y_pred, model_name, infile_name):
    cm = confusion_matrix(y_true, y_pred)

    true_labels = ['True Pro-Choice', 'True Pro-Life']
    predicted_labels = ['Predicted Pro-Choice', 'Predicted Pro-Life']

    plt.figure(figsize=(6,6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False,
                     xticklabels=predicted_labels, yticklabels=true_labels,
                     annot_kws={"size": 64},
                     cbar_kws={'label': 'Colorbar'})

    ax.set_title(f"Confusion Matrix ({model_name})", fontsize=18)
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_ylabel('True Label', fontsize=16)

    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()

    plt.savefig(f"../plots/{infile_name}-{model_name}-confusion_matrix.png")
    plt.close()
    
def plot_accuracy_vs_epoch(history, infile_name, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history_['accuracy'], label='Training Accuracy')
    plt.plot(history.history_['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy vs Epoch ({model_name} with {infile_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.savefig(f"../plots/{infile_name}-{model_name}-accuracy_vs_epoch.png")
    plt.close()

if __name__ == '__main__':
    main()
