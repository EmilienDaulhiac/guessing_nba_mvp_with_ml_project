import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regression_model(model, X_test, y_test, df_result = pd.DataFrame(), name: str = None, print_results: bool = True):
    """
    Evaluate a pre-fitted regression model on the test set and return a DataFrame with MSE, MAE, and R2 Score.

    Parameters:
    - model (object): A pre-fitted regression model.
    - X_test (array-like): Features of the test set.
    - y_test (array-like): True labels of the test set.
    - print_results (bool): Print MSE, MAE and R2 score on console if TRUE

    Returns:
    - pd.DataFrame: A DataFrame containing MSE, MAE, and R2 Score for the test set.
    """

    # Predictions on the test set
    y_predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_predictions)
    mae = mean_absolute_error(y_test, y_predictions)
    r2 = r2_score(y_test, y_predictions)


    if print_results:
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Squarred Error: {mse}")
        print(f"R2 score: {r2}")
        
    if name is None:
        name = 'Model '+str(df_result.shape[0])
    # Create a DataFrame to store the results
    evaluation_results = pd.DataFrame.from_dict( {name : [mse, mae, r2]}, orient='index', columns = ['MSE', 'MAE', 'R2 Score'])

    if df_result is not None:
        evaluation_results = pd.concat([evaluation_results,df_result])

    return evaluation_results

def evaluate_MVP_classification_from_regression(
        model, X_test, df_test, name_model : str = None,
        report_file_name : str = None):
    
    # Predictions on the test set
    y_predictions = model.predict(X_test)

    if name_model is None:
        name_model = "model"

    i=0
    while name_model in df_test.columns:
        name_model = name_model+"_"+str(i)
        i = i+1
    
    df_test["award_share_"+name_model] = y_predictions

    if report_file_name is not None:
        file_report = open(report_file_name, 'w')

        file_report.write(f'--------------------------------------------------------- \n')
        file_report.write(f'--------- Results of tests run on the CNN model --------- \n')
        file_report.write(f'--------------------------------------------------------- \n \n')

        file_report.write(f'This report list for the season selected to test our models \n')
        file_report.write(f'The actual MVP elected that specific year, the MVP selected by the model \n ')
        file_report.write(f'The amount of votes computed by the model, and the actual amount of votes the player got\n \n')

    succes_at_classifying_MVP = []
    for num,year in enumerate(df_test['season'].unique()):
        preticted_mvp = df_test[df_test["season"] == year].sort_values(by="award_share_"+name_model, ascending=False).head(1)["player"].values[0]
        odds_for_preticted_mvp = df_test[df_test["season"] == year].sort_values(by="award_share_"+name_model, ascending=False).head(1)[["award_share_"+name_model,"award_share"]].values[0]
        real_mvp = df_test[df_test["season"] == year].sort_values(by="award_share", ascending=False).head(1)["player"].values[0]
        odds_for_real_mvp = df_test[df_test["season"] == year].sort_values(by="award_share", ascending=False).head(1)[["award_share_"+name_model,"award_share"]].values[0]
        succes_at_classifying_MVP.append(preticted_mvp == real_mvp)
        
        if report_file_name is not None:
            file_report.write(f'--------- \n Guess n°{num+1}  - season {year}\n')
            file_report.write(f'Success: {preticted_mvp == real_mvp}\n')
            file_report.write(f'Guessed MVP: {preticted_mvp}\n')
            file_report.write(f'Odds computed: {odds_for_preticted_mvp[0]}  Real odds: {odds_for_preticted_mvp[1]}\n')
            file_report.write(f'Real MVP: {real_mvp}\n')
            file_report.write(f'Odds computed: {odds_for_real_mvp[0]}  Real odds: {odds_for_real_mvp[1]}\n')
        
    print('MVP Classifaction measured Accuracy : ', round(100*sum(succes_at_classifying_MVP)/len(succes_at_classifying_MVP), 3),'%')

    if report_file_name is not None:
        file_report.write(f' ---- Overall result -----\n')
        file_report.write(f'Accuracy: {round(100*sum(succes_at_classifying_MVP)/len(succes_at_classifying_MVP), 3)}%\n')

    return df_test

    

def evaluate_cnn_model(model, X_test, y_test, player_dict, df_result = pd.DataFrame(), report_file_name = None,  name: str = None):
    """
    Evaluate a pre-fitted cnn model on the test set and return a DataFrame with MSE, MAE, and R2 Score.

    Parameters:
    - model (object): A pre-fitted regression model.
    - X_test (array-like): Features of the test set.
    - y_test (array-like): True labels of the test set.

    Returns:
    - pd.DataFrame: A DataFrame containing MSE, MAE, and R2 Score for the test set.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if name is None:
        name = 'Model '+str(df_result.shape[0])
    # Create a DataFrame to store the results
    evaluation_results = pd.DataFrame.from_dict( {name : [mse, mae, r2]}, orient='index', columns = ['MSE', 'MAE', 'R2 Score'])

    if df_result is not None:
        evaluation_results = pd.concat([evaluation_results,df_result])

    list_of_result = []

    if report_file_name is not None:

        # Assuming you have already opened a file for writing
        with open(report_file_name, 'w') as file:
            file.write(f'--------------------------------------------------------- \n')
            file.write(f'--------- Results of tests run on the CNN model --------- \n')
            file.write(f'--------------------------------------------------------- \n \n')

            file.write(f"This report list for each partition of test which season actually it was extracted from \n")
            file.write(f"The actual MVP elected that specific year, the MVP selected by the model on the given partition of eligible players \n") 
            file.write(f"The odds computed by the model, and the actual amount of votes the player got\n")
            file.write(f' /!\ Important to notes that the odds computed by the CNN model is not an attempt at predicting the number of votes \n \n')
            for num, pred in enumerate(y_pred):
                try:
                    index_of_predicted = np.where(pred == np.max(pred))[0][0]
                    index_of_real = np.where(y_test[num] == 1)[0][0]
                    if index_of_predicted == index_of_real:
                        list_of_result.append(True)
                    else:
                        list_of_result.append(False)

                    file.write(f'--------- \n Guess n°{num+1}\n')
                    file.write(f'Success: {index_of_predicted == index_of_real}\n')
                    file.write(f'Guessed MVP: {player_dict[num][index_of_predicted]["player"]}\n')
                    file.write(f'Odds computed: {np.max(pred)}  Real odds: {player_dict[num][index_of_predicted]["award_share"]}\n')
                    file.write(f'Real MVP: {player_dict[num][index_of_real]["player"]}\n')
                    file.write(f'Odds computed: {pred[index_of_real]}  Real odds: {player_dict[num][index_of_real]["award_share"]}\n')

                except Exception as e:
                    file.write(f'Error: {str(e)}\n')

            file.write(f'Results: {list_of_result}\n')
            file.write(f'Accuracy: {round(100*sum(list_of_result)/len(list_of_result), 3)}%\n')

    print('MVP Classification Accuracy :')
    print(f'Accuracy: {round(100*sum(list_of_result)/len(list_of_result), 3)}%\n')

    return y_pred, evaluation_results

def read_and_sort_CNN_MVP_prediction(cnn_predicition, df_MVP_player):
    """Out of the a dataset of predicted MVPs for the current season, 
    count how much time one appears as MVP and store the result in a dict and returns it.
    
    Parameters:
    - cnn_prediction: Output predicition of the CNN model on a multiple set of 10 players 'eligible' for the MVP
    - df_MVP_player: list of set of 10 players 'eligible' for the MVP 

    Returns:
    - dict: dict of how much time one player was selected as the MVP out of his group.
    """
    dict_of_MVP_count = {}
    for num, pred in enumerate(cnn_predicition):
        index_of_predicted = np.argmax(pred)
        player = df_MVP_player[num][index_of_predicted]["player"]
        dict_of_MVP_count.setdefault(player,0)
        dict_of_MVP_count[player] = dict_of_MVP_count.get(player,0) + 1
        
    return dict(sorted(dict_of_MVP_count.items(), key=lambda item:item[1], reverse=True))

    

