from src.PreProcessing import PlayersData, create_clean_dataset_for_test, clean_input_data_for_regression, clean_scale_input_data_for_cnn, apply_eligibilty_criteria
from src.modeling import train_decision_tree, evaluate_regression_model, train_random_forest, train_cnn, evaluate_cnn_model,  evaluate_MVP_classification_from_regression,read_and_sort_CNN_MVP_prediction
import numpy as np
from src.get_data_online import BaskRefPlayerDataScraper
from src.PreProcessing import create_clean_dataset_for_test
import json
import pprint
import os


if __name__== '__main__':
    # Load CSV dataset
    data = PlayersData('Data/NBA_Dataset.csv')
    X_train, y_train, X_test, y_test, years_for_test, imputer,df_train_reg,df_test_reg  = data.get_splitted_data_for_regression(number_of_season_for_test=5, year_test_regression=np.array([2019,1985,2009,2011,2022]))
    print('---------- Training Regression Model --------- ')
    
    isExist = os.path.exists('output')
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs('output')

    #Train decision tree model
    print('----- Decision Tree model ------')
    print('* TRAIN SET * \n')
    model_decision_tree = train_decision_tree(X_train,y_train, max_depth=4, min_samples_leaf=4, min_samples_split=10)
    df_result = evaluate_regression_model(model_decision_tree, X_train, y_train, name='Decision Tree')

    print('* TEST SET * \n')
    df_result = evaluate_regression_model(model_decision_tree, X_test, y_test, name='Decision Tree')

    df_train_w_result = evaluate_MVP_classification_from_regression(
        model_decision_tree, 
        X_test, 
        df_test_reg,
        report_file_name='output/report_decision_tree.txt')
    
    print('----- Random Forest model ------')
    print('* TRAIN SET * \n')
    model_random_forest = train_random_forest(X_train,y_train)
    df_result = evaluate_regression_model(model_random_forest,  X_train,  y_train, df_result, name='Random Forest')
    
    print('* TEST SET * \n')
    df_result = evaluate_regression_model(model_random_forest, X_test, y_test, name='Decision Tree')

    df_train_w_result = evaluate_MVP_classification_from_regression(
        model_random_forest, 
        X_test, 
        df_test_reg,
        report_file_name='output/report_test_random_forest.txt')

    print(' \n----- CNN ------ \n')
    X_train, y_train, X_test, y_test, players_list_train, players_list_test,  years_for_test, nan_imputer, scaler_dl, onehotencoder = data.get_splitted_data_for_dl(years_for_test=years_for_test)

    model3 = train_cnn(X_train, y_train, num_epochs=50)

    _, y = evaluate_cnn_model(model3, X_test, y_test, players_list_test, df_result= df_result, name= "CNN", report_file_name='output/report_test_cnn.txt')

    print('\n ----- Scrape Players stats from basketball-reference.com -----\n ')
    player_data_scraper = BaskRefPlayerDataScraper()
    game_data = player_data_scraper.scrape_table_data(
        "https://www.basketball-reference.com/leagues/NBA_2024_per_game.html#per_game_stats", 
        'div_per_game_stats', 
        'full_table')
    
    advanced_data = player_data_scraper.scrape_table_data(
        'https://www.basketball-reference.com/leagues/NBA_2024_advanced.html', 
        'div_advanced_stats', 
        'full_table')
    
    team_data = player_data_scraper.scrape_table_data(
        'https://www.basketball-reference.com/leagues/NBA_2024_ratings.html', 
        'div_ratings')

    df_players_stats_2023 = create_clean_dataset_for_test(
        game_data, 
        advanced_data, 
        team_data, 
        path_to_aggragation_team_json = 'Data/external/teams_abbreviation.json',
        code_for_position = data.dict_pos_factorize
        )

    X_2023_regression = clean_input_data_for_regression(df_players_stats_2023, imputer)

    X_2023_cnn, player_D = clean_scale_input_data_for_cnn(
        df_players_stats_2023, 
        nan_imputer, 
        scaler_dl, 
        onehotencoder,
        multiplier_of_data=5000)

    test_2023 = model3.predict(X_2023_cnn)

    CNN_predicted_MVP_output = read_and_sort_CNN_MVP_prediction(test_2023, player_D)
    # Prints the nicely formatted dictionary
    print('\n MVP top 10 based on CNN :\n ')
    pprint.pprint({k: CNN_predicted_MVP_output[k] for k in list(CNN_predicted_MVP_output.keys())[:10]}, sort_dicts=False)

    y_pred_dt = model_decision_tree.predict(X_2023_regression)
    y_pred_rf = model_random_forest.predict(X_2023_regression)

    df_mvp_candidates_stats_2023 = apply_eligibilty_criteria(df_players_stats_2023)

    df_mvp_candidates_stats_2023['odds computed (decision tree)'] = y_pred_dt
    df_mvp_candidates_stats_2023['odds computed (random forest)'] = y_pred_rf

    print('\n MVP top 5 based on Decision tree :\n ', df_mvp_candidates_stats_2023.sort_values(by=['odds computed (decision tree)'], ascending=False).loc[:,['player','odds computed (decision tree)']].head(5))

    print('\n MVP top 5 based on Random Forest :\n ', df_mvp_candidates_stats_2023.sort_values(by=['odds computed (random forest)'], ascending=False).loc[:,['player','odds computed (random forest)']].head(5))
