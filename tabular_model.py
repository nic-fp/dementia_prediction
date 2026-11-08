import xgboost as xgb
import pandas as pd  
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def plot_training_curve(model, name):

    #this exctracts the results from each epoch
    results = model.evals_result()
    train_loss = results['validation_0']['logloss']
    val_loss = results['validation_1']['logloss']
    # (Assuming 'train_loss' and 'val_loss' are already extracted from your model)
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Boosting rounds')
    plt.ylabel('Log loss')
    plt.title(f'Loss by boosting round - {name}')
    plt.legend()

    # Save to PNG instead of displaying
    plt.savefig(f'results/tabular/{name}/learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close() 

def plot_importance(model, name):
    plt.figure(figsize=(8, 5))

    xgb.plot_importance(
        model,
        max_num_features=20,
        importance_type='gain',
        height=0.8
    )

    plt.title(f"Feature Importance - {name}")
    plt.savefig(f"results/tabular/{name}/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_roc_auc_curve(model, y_probs, y_true, name):
    # cacl ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    # calc AUC
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve - {name}')
    plt.legend(loc='lower right')
    plt.savefig(f"results/tabular/{name}/roc_auc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def evaluate(model, x_eval, y_eval, name ):
    """
    This function takes in a model and evaluates performance on the eval set.
    Plots perf
    """
    y_pred = model.predict(x_eval)
    y_prob = model.predict_proba(x_eval)[:, 1]
    plot_importance(model, name)
    plot_roc_auc_curve(model, y_prob, y_eval, name)
    print(f'Accuracy: {accuracy_score(y_eval, y_pred):.4f}')
    print(f'ROC-AUC:  {roc_auc_score(y_eval, y_prob):.4f}')
    print('\nClassification Report:\n', classification_report(y_eval, y_pred))


def train(train_data, columns_to_use, name):
    X = train_data[columns_to_use]
    y = train_data['final_label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,stratify=y
    )
    #from the training data split into train and test for hyper param tuning. Val data will ONLY be used for final results so not leaking

    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=10,
        min_child_weight=2, 
        random_state=42,
        eval_metric='logloss',   
        early_stopping_rounds = 20, 
    )
    #note, I pass both sets in to eval set so I can plot train and val loss. Early stopp only looks at last in list
    model.fit(X_train, y_train, eval_set = [(X_train,y_train), (X_test, y_test)],verbose=False)
    plot_training_curve(model, name)
    evaluate(model, X_test, y_test, name)
    return model 

def validate(model, val_data, columns_to_use,  name):
    X = val_data[columns_to_use]
    y = val_data['final_label']
    evaluate(model, X,y, name)

def main():
    train_data = pd.read_csv('data/tabular/train_data.csv')
    val_data = pd.read_csv('data/tabular/val_data.csv')
    train_data['final_label'] = (train_data['label'] == 'dementia').astype(int)   
    val_data['final_label'] = (val_data['label'] == 'dementia').astype(int)

    columns = train_data.columns.unique().to_list()
    audio_columns = [col for col in columns if col.startswith('audio_')]
    text_columns = [col for col in columns if col.startswith('text_')]
    print('Audio only')
    audio_model = train(train_data, audio_columns, 'Audio only')
    print('Audio only validation')
    validate(audio_model, val_data, audio_columns, 'Audio only validation')
    print('='*200)
    print('Text only')
    text_model = train(train_data, text_columns, 'Text only')
    print('Text only validation')
    validate(text_model, val_data, text_columns, 'Text only validation')
    print('='*200)
    print('All data')
    all_data_model = train(train_data, audio_columns+text_columns, 'All data')
    print('All data validation')
    validate(all_data_model, val_data, audio_columns+text_columns, 'All data validation')
    print('='*200)

if __name__ == "__main__":
    main()
