"""End-to-end NBA MVP prediction pipeline.

Trains three models (Decision Tree, Random Forest, CNN) on historical
voting data, evaluates them on held-out seasons, then scrapes the current
season's stats from basketball-reference.com to predict that year's MVP.
"""
import os
import pprint

import numpy as np

from src.PreProcessing import (
    PlayersData,
    apply_eligibilty_criteria,
    clean_input_data_for_regression,
    clean_scale_input_data_for_cnn,
    create_clean_dataset_for_test,
)
from src.modeling import (
    evaluate_cnn_model,
    evaluate_MVP_classification_from_regression,
    evaluate_regression_model,
    read_and_sort_CNN_MVP_prediction,
    train_cnn,
    train_decision_tree,
    train_random_forest,
)
from src.get_data_online import BaskRefPlayerDataScraper


def main():
    np.random.seed(42)
    import tensorflow as tf
    tf.random.set_seed(42)

    os.makedirs("output", exist_ok=True)

    data = PlayersData("Data/NBA_Dataset.csv")
    (
        X_train,
        y_train,
        X_test,
        y_test,
        years_for_test,
        imputer,
        _,
        df_test_reg,
    ) = data.get_splitted_data_for_regression(
        number_of_season_for_test=5,
        year_test_regression=np.array([2019, 1985, 2009, 2011, 2022]),
    )

    print("----- Decision Tree -----")
    model_decision_tree = train_decision_tree(
        X_train, y_train, max_depth=4, min_samples_leaf=4, min_samples_split=10
    )
    df_result = evaluate_regression_model(
        model_decision_tree, X_test, y_test, name="Decision Tree"
    )
    evaluate_MVP_classification_from_regression(
        model_decision_tree,
        X_test,
        df_test_reg,
        name_model="Decision Tree",
        report_file_name="output/report_decision_tree.txt",
    )

    print("----- Random Forest -----")
    model_random_forest = train_random_forest(X_train, y_train)
    df_result = evaluate_regression_model(
        model_random_forest, X_test, y_test, df_result, name="Random Forest"
    )
    evaluate_MVP_classification_from_regression(
        model_random_forest,
        X_test,
        df_test_reg,
        name_model="Random Forest",
        report_file_name="output/report_random_forest.txt",
    )

    print("----- CNN -----")
    (
        X_train,
        y_train,
        X_test,
        y_test,
        _,
        players_list_test,
        years_for_test,
        nan_imputer,
        scaler_dl,
        onehotencoder,
    ) = data.get_splitted_data_for_dl(years_for_test=years_for_test)

    model_cnn = train_cnn(X_train, y_train, num_epochs=50)
    evaluate_cnn_model(
        model_cnn,
        X_test,
        y_test,
        players_list_test,
        df_result=df_result,
        name="CNN",
        report_file_name="output/report_cnn.txt",
    )

    print("----- Scrape current season stats from basketball-reference.com -----")
    scraper = BaskRefPlayerDataScraper()
    game_data = scraper.scrape_table_data(
        "https://www.basketball-reference.com/leagues/NBA_2024_per_game.html#per_game_stats",
        "div_per_game_stats",
        "full_table",
    )
    advanced_data = scraper.scrape_table_data(
        "https://www.basketball-reference.com/leagues/NBA_2024_advanced.html",
        "div_advanced_stats",
        "full_table",
    )
    team_data = scraper.scrape_table_data(
        "https://www.basketball-reference.com/leagues/NBA_2024_ratings.html",
        "div_ratings",
    )

    df_players_stats = create_clean_dataset_for_test(
        game_data,
        advanced_data,
        team_data,
        path_to_aggragation_team_json="Data/external/teams_abbreviation.json",
        code_for_position=data.dict_pos_factorize,
    )

    X_regression = clean_input_data_for_regression(df_players_stats, imputer)
    X_cnn, player_partitions = clean_scale_input_data_for_cnn(
        df_players_stats,
        nan_imputer,
        scaler_dl,
        onehotencoder,
        multiplier_of_data=5000,
    )

    cnn_predictions = model_cnn.predict(X_cnn)

    player_counts = {}
    for partition in player_partitions:
        for player_dict in partition:
            name = player_dict["player"]
            player_counts[name] = player_counts.get(name, 0) + 1

    cnn_top_picks = read_and_sort_CNN_MVP_prediction(cnn_predictions, player_partitions)
    print("\nMVP top 10 (CNN):\n")
    pprint.pprint(
        {
            k: {"Count": cnn_top_picks[k], "Ratio": round(cnn_top_picks[k] / player_counts[k], 4)}
            for k in list(cnn_top_picks.keys())[:10]
        },
        sort_dicts=False,
    )

    df_candidates = apply_eligibilty_criteria(df_players_stats)
    df_candidates["odds (decision tree)"] = model_decision_tree.predict(X_regression)
    df_candidates["odds (random forest)"] = model_random_forest.predict(X_regression)

    print("\nMVP top 5 (Decision Tree):\n")
    print(
        df_candidates.sort_values(by="odds (decision tree)", ascending=False)
        .loc[:, ["player", "odds (decision tree)"]]
        .head(5)
    )
    print("\nMVP top 5 (Random Forest):\n")
    print(
        df_candidates.sort_values(by="odds (random forest)", ascending=False)
        .loc[:, ["player", "odds (random forest)"]]
        .head(5)
    )


if __name__ == "__main__":
    main()
