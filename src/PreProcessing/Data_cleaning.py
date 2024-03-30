import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import KNNImputer
import json

#Dcitionnary of columns and values players have to 
#be above to be relevant for our analysis.
eligibilty_criterias = {
    'vorp': 0.1,
    'g': 40,
    'gs': 20,
    'mp_per_g': 27.89,
    "usg_pct": 17.5,
    "per": 17,
    "ws": 5
}

#path to regular dataset
path_to_csv = '../../Data/NBA_Dataset.csv'

def change_columns_name(df: pd.DataFrame, columns_name):
    return df.rename(columns={ old : real for old, real in list(zip(df.columns, columns_name))})

def apply_eligibilty_criteria(df: pd.DataFrame, dict_of_criterias: dict[str,int] = eligibilty_criterias):
    """
    From a dict of criteria on stats, select players with values over designed treshold.

    Parameters:
    - df (pd.DataFrame): Dataframe of players and their stats
    - dict_of_criterias (dict): dict of criterias where keys are stats and item are values to filter eligible player.

    Returns:
    - pd.DataFrame: A DataFrame containing eligible players.
    """
    for column, threshold in dict_of_criterias.items():
        df = df[df[column] >= threshold]
    df.reset_index(inplace = True)
    return df

def create_clean_dataset_for_test(df_per_game: pd.DataFrame,
                                  df_advanced: pd.DataFrame,
                                  df_teams: pd.DataFrame,
                                  path_to_aggragation_team_json: str,
                                  code_for_position: dict[str,int] = None,
                                  columns_name: list[str] = None,
                                  path_to_save_csv: str = None,
                                  ):
    
    """
    From datafarmes extracted on BasketballReference of stats average, advanced stats, and team stats, 
    merged them all together to create a similar dataset as the original training set.

    Parameters:
    - df_per_game, df_advanced, df_teams (pd.DataFrame)
    - path_to_aggragation_team_json (dict): dict where keys are full NBA team name and item are the abreviation
    - code_for_position (dict): dict to factorize the players position the same way as in the train set
    - columns_name (list): columns names of the original dataset
    - path_to_save_csv (str): path to save merged dataframe as a csv. No save if none.


    Returns:
    - pd.DataFrame: A DataFrame containing eligible players.
    """
    f = open(path_to_aggragation_team_json)
    abbreviation_dict = json.load(f)

    df_per_game['season'] = 2024
    df_advanced = df_advanced.drop(['pos','age','team_id','g'], axis=1)

    df_merged = df_per_game.merge(df_advanced, on='player', how='left')

    df_teams['team_id'] = df_teams['team_name'].apply(lambda x: abbreviation_dict.get(x, 'TOT'))
    df_teams['G'] = df_teams['wins'] + df_teams['losses']

    df_merged = df_merged.merge(df_teams.loc[:,["team_id","mov","mov_adj","win_loss_pct"]], on='team_id', how='left')

    if columns_name is not None:
        df_merged = change_columns_name(df_merged, columns_name)

    games_per_season = 82

    for ind, row in df_merged.iterrows():
        try : 
            number_of_games_team_played = df_teams[df_teams['team_id'] == row['team_id']]['G'].values[0]
        except:
            number_of_games_team_played = df_teams['G'].mean()
        df_merged.loc[ind,'g'] = min(int( row['g'] * games_per_season / number_of_games_team_played),82)
        df_merged.loc[ind,'gs'] = int( row['gs'] * games_per_season / number_of_games_team_played)
        df_merged.loc[ind,'mp'] = int( row['mp'] * games_per_season / number_of_games_team_played)
        df_merged.loc[ind,'ows'] =  row['ows'] * games_per_season / number_of_games_team_played
        df_merged.loc[ind,'dws'] = row['dws'] * games_per_season / number_of_games_team_played
        df_merged.loc[ind,'ws'] =  row['ws'] * games_per_season / number_of_games_team_played

    #keeping only primary position on the court
    df_merged["pos"] = df_merged["pos"].apply(lambda x: x.split("-")[0])
    if code_for_position is not None:
        df_merged["pos"] = df_merged["pos"].apply(lambda x: code_for_position[x])
        
    if path_to_save_csv is not None:
        df_merged.to_csv(path_to_save_csv)

    return df_merged

def clean_input_data_for_regression(data: pd.DataFrame,
                                      imputer: KNNImputer,
                                      dict_of_criterias : dict[str,int] = eligibilty_criterias):
    """Drop useless columns and fill NA values with previously trained imputer."""
    data = apply_eligibilty_criteria(data, dict_of_criterias=dict_of_criterias )
    X_2023 = data.drop(["season","player",'team_id'], axis=1)
    X_2023 = imputer.transform(X_2023)
    return X_2023

def clean_scale_input_data_for_cnn(data: pd.DataFrame,
                                nan_imputer: KNNImputer,
                                scaler_dl: MinMaxScaler,
                                onehotencoder: OneHotEncoder,
                                multiplier_of_data: int = 10,
                                dict_of_criterias : dict[str,int] = eligibilty_criterias):
    """Drop useless columns, keep eligible players, encode position, scale the data and fill NA values with previously trained transformer."""
    data = apply_eligibilty_criteria(data, dict_of_criterias=dict_of_criterias )
    encoded_data = onehotencoder.transform(data[['pos']]).toarray()
    feature_names = onehotencoder.get_feature_names_out(['pos'])

    # Replace original column with one-hot encoded columns for the first DataFrame
    X_2023 = pd.concat([data, pd.DataFrame(encoded_data, columns=feature_names)], axis=1)

    X_2023.drop(['pos'], axis=1, inplace=True)

    X_list_test = []
    players_list = []

    list_of_index = [ ]
    for i in range(multiplier_of_data):
        temp = np.random.choice( np.arange(0,X_2023.shape[0]), 10, replace= False)
        while 1 in temp and any(set(temp) == set(listindex) for listindex in list_of_index):
            temp = np.random.choice( np.arange(0,X_2023.shape[0]), 10, replace= False)
        list_of_index.append(temp)

    for partition in list_of_index:
        players_list.append(X_2023.iloc[partition,:][["player"]].to_dict('records'))
        X_matrix = scaler_dl.transform(
            nan_imputer.transform(
                X_2023.iloc[partition,:].drop(['season', 'player','team_id'], axis=1)
                )
                )
        X_list_test.append(X_matrix)
        
    X_test = np.array(X_list_test)
    X_test = X_test[:, :, :, np.newaxis]


    return X_test, players_list    

class PlayersData:
    def __init__(self, 
                 path: str = path_to_csv):
        self.raw_data = pd.read_csv(path)
        self.eligible_players_data = self.get_cleaned_data()

    def get_cleaned_data(self, 
                         dict_of_criterias: dict[str, float] = eligibilty_criterias):
        """
        From raw data of players:
         1. fatorize position on the court.
         2. Apply previously designated eligibilty criterias for MVP over the players

        Parameters:
        - dict_of_criterias (dict): dict of criterias where keys are stats and item are values to filter eligible player.

        Returns:
        - pd.DataFrame: A clan DataFrame containing eligible players.
        """
        temp_df = self.raw_data.copy()
        
        temp_df = apply_eligibilty_criteria(temp_df, dict_of_criterias)

        temp_df["pos"] = temp_df["pos"].apply(lambda x: x.split("-")[0])
        # Label Encoding position
        temp_df["pos"], unique_categories = pd.factorize(temp_df["pos"])

        #Save Label Encoding to apply it to other dataset
        self.dict_pos_factorize = {code: value for value, code in enumerate(unique_categories)}

        return temp_df
    
    def get_splitted_data_for_regression(self, number_of_season_for_test = 5, year_test_regression: list[int] = [2022] , imputation: bool = True):
        """
        From dataset of eligible players:
         1. Drop useless columns.
         2. Randomly select season to split the dataset to create a test set
         3. Split dataset into train and test set
         4. Use KNN to fill NA values. No impuation if imputation = False
        Return X,Y for train and test set

        Parameters:
        - number_of_season_for_test: number of season to select fo test.
        - year_test_regression: list of season that are pre selected to be in the test set.
        - imputation: boolean value to apply imputation or not over the test and train set.

        Returns:
        - X_train, y_train, X_test, y_test: X,Y for train and test.
        - year_test_regression (list[int]): list of years selected for test.
        - imputer_regression (KNNImputer): KNN Imputer train on train data.
        - df_train, df_test(pd.Dataframe): dataframe for train and test
        """
        temp_df = self.eligible_players_data.drop(["team_id"],axis=1)

        if  number_of_season_for_test - len(year_test_regression) > 0:
            year_test_regression = np.append(
                                np.random.choice( 
                                    np.setdiff1d(
                                        temp_df["season"].unique(), year_test_regression
                                        ), 
                                    number_of_season_for_test - len(year_test_regression), replace= False), 
                                2022)
        
        df_test = temp_df[temp_df['season'].isin(year_test_regression)]

        df_train = temp_df[
            temp_df['season'].isin(
                np.setdiff1d(
                    temp_df["season"].unique(), year_test_regression
                            )
                        )
                    ]
        
        X_train = df_train.drop(columns=["award_share","player","season"])
        y_train = df_train["award_share"]

        X_test = df_test.drop(columns=["award_share","player","season"])
        y_test = df_test["award_share"]
        
        if imputation == True:
            # define imputer
            imputer_regression = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
            # fit on the dataset
            imputer_regression.fit(X_train)
            # transform the dataset
            X_train = imputer_regression.transform(X_train)
            X_test = imputer_regression.transform(X_test)

        return X_train, y_train, X_test, y_test, year_test_regression, imputer_regression, df_train, df_test
    
    def get_splitted_data_for_dl(self, 
                                 years_for_test: list[int] = [], 
                                 multiplier_of_data: int = 10, 
                                 max_pos_for_eligibility: int = 21,
                                 nan_imputer: KNNImputer = None,
                                 scaler_dl: MinMaxScaler = None,
                                 onehotencoder: OneHotEncoder = None):
        """ Returns X and labels spliited ready for training with nan imputed,
          scaling and one hot encoding in categorical value.

          Input:
          - years_for_test: list of years selected to split models
          - multiplier_of_data: number of parttion to do for each year where we hide the MVP between 9 other players in the top 20 voting.
          - max_pos_eligibilty: lowest ranking position to be selected in the partition where we hide the MVP and his stats
          - nan_imputer: KNNImputer to impute Nan values
          - scaler_dl: MinMaxScaler to rescale the data
          - onehotencoder: OneHotEncoder to encode the position of the players

          Return:
          X_train, y_train, X_test, Y_test: 
                shape of X (number of seasons for (training/test)*multiplier of data, 10, number of features), 
                shape of Y (number of seasons for (training/test)*multiplier of data, 10)
          players_list_train, players_list_train : list of dict where we can find for each partition the name and the number of votes for each player of a parttion
          years_for_test : years selected to split the data
          nan_imputer, scaler_dl, onehotencoder: encoder used for building the dataset
          """

        temp_df = self.eligible_players_data.drop(["team_id"],axis=1)
        years = temp_df['season'].unique()

        if len(years_for_test)== 0:
            years_for_test = np.append(
                            np.random.choice( 
                                np.setdiff1d(
                                    years, [2022]
                                    ), 
                                4, replace= False), 
                            2022)
        
            
        df_train = temp_df[temp_df['season'].isin(np.setdiff1d(years, years_for_test))].reset_index(drop=True)
        df_test = temp_df[temp_df['season'].isin(years_for_test)].reset_index(drop=True)

        if onehotencoder is None:
            onehotencoder = OneHotEncoder()
            encoded_data = onehotencoder.fit_transform(df_train[['pos']]).toarray()
        else :
            encoded_data = onehotencoder.transform(df_train[['pos']]).toarray()

        feature_names = onehotencoder.get_feature_names_out(['pos'])
        # Replace original column with one-hot encoded columns for the first DataFrame
        df_train = pd.concat([df_train, pd.DataFrame(encoded_data, columns=feature_names)], axis=1)
        df_train.drop(['pos'], axis=1, inplace=True)
        # Apply the same encoding to the second DataFrame
        encoded_data2 = onehotencoder.transform(df_test[['pos']]).toarray()
        df_test = pd.concat([df_test, pd.DataFrame(encoded_data2, columns=feature_names)], axis=1)
        df_test.drop(['pos'], axis=1, inplace=True)

        if nan_imputer is None:
            nan_imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
            nan_imputer.fit(df_train.drop(['season', 'player', 'award_share'], axis=1))

        if scaler_dl is None:
            scaler_dl = MinMaxScaler()
            scaler_dl.fit_transform(
                    nan_imputer.transform(
                        df_train.drop(['season', 'player', 'award_share'], axis=1)))

        X_list_train, y_list_train, players_list_train  = [], [], []
        X_list_test, y_list_test, players_list_test = [], [], []

        weights = [1/i for i in range(1,max_pos_for_eligibility)]

        normalized_weights = [w / sum(weights) for w in weights]

        for dataset, x, y, player in [(df_train, X_list_train, y_list_train, players_list_train),
                              (df_test, X_list_test, y_list_test, players_list_test)]:
            for year in dataset['season'].unique():
                # Filter data for the current year
                year_data = dataset[dataset['season'] == year].sort_values(
                    by=["award_share","per"], ascending=False).reset_index(drop=True)
                
                # Select the top 10 players
                top_10_players = year_data.head(max_pos_for_eligibility+1)

                list_of_index = [ np.arange(0,10) ]

                np.random.shuffle(list_of_index[0])

                for i in range(multiplier_of_data):
                    temp = np.random.choice( np.arange(0,max_pos_for_eligibility-1), 10, p=normalized_weights, replace= False)

                    while 1 in temp and any(set(temp) == set(listindex) for listindex in list_of_index):

                        temp = np.random.choice( np.arange(0,max_pos_for_eligibility-1), 10, p=normalized_weights, replace= False)
                    list_of_index.append(temp)
                        
                for partition in list_of_index:
                    player.append(top_10_players.iloc[partition,:][["player","award_share"]].to_dict('records'))
                        
                    X_matrix = scaler_dl.transform(nan_imputer.transform(top_10_players.iloc[partition,:].drop(['season', 'player', 'award_share'], axis=1)))
                    y_vector = [ 0 if rank != 0 else 1 for rank in partition]

                    x.append(X_matrix)
                    y.append(y_vector)

        X_train = np.array(X_list_train)
        X_train = X_train[:, :, :, np.newaxis]
        y_train = np.array(y_list_train)

        X_test = np.array(X_list_test)
        X_test = X_test[:, :, :, np.newaxis]
        y_test = np.array(y_list_test)

        return X_train, y_train, X_test, y_test, players_list_train, players_list_test,  years_for_test, nan_imputer, scaler_dl, onehotencoder
        