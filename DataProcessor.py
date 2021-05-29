import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings

from statsmodels.stats.outliers_influence import variance_inflation_factor


warnings.filterwarnings('ignore')
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


class DataProcessor:

    def read_data_from_database(self, schema_path, loaded_path):
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123Saeed",
            database="even_financial"
        )
        my_cursor = mydb.cursor()

        my_cursor.execute('SELECT * FROM clicks')
        table_rows = my_cursor.fetchall()
        clicks_df = pd.DataFrame(table_rows)

        my_cursor.execute('SELECT * FROM leads')
        table_rows = my_cursor.fetchall()
        leads_df = pd.DataFrame(table_rows)

        my_cursor.execute('SELECT * FROM offers')
        table_rows = my_cursor.fetchall()
        offers_df = pd.DataFrame(table_rows)

        click_schema = pd.read_csv(schema_path + 'clicks_schema.csv', header=1)
        lead_schema = pd.read_csv(schema_path + 'leads_schema.csv', header=1)
        offer_schema = pd.read_csv(schema_path + 'offers_schema.csv', header=1)

        click_columns = click_schema.columns
        lead_columns = lead_schema.columns
        offer_columns = offer_schema.columns

        clicks_df.columns = click_columns
        leads_df.columns = lead_columns
        offers_df.columns = offer_columns

        clicks_df.to_csv(loaded_path + 'clicks.csv', index=False)
        leads_df.to_csv(loaded_path + 'leads.csv', index=False)
        offers_df.to_csv(loaded_path + 'offers.csv', index=False)

    def generate_data_frame(self, loaded_path):

        click_df = pd.read_csv(loaded_path + 'clicks.csv')
        lead_df = pd.read_csv(loaded_path + 'leads.csv')
        offer_df = pd.read_csv(loaded_path + 'offers.csv')

        data = offer_df.copy()
        data['clicked'] = data['offer_id'].isin(click_df['offer_id']).astype(int)
        data = data.merge(lead_df, how='left', left_on='lead_uuid', right_on='lead_uuid')
        return data

    def preprocess(self, data):
        print('Proportion of null values in data:\n{}'.format((data.isna().sum() / len(data)) * 100))
        data.dropna(inplace=True)
        print(data['clicked'].value_counts())

        fig = plt.figure(figsize=(5, 5))
        plt.bar(data['clicked'].value_counts().index, data['clicked'].value_counts(), width=0.2)
        plt.xlabel("Click label")
        plt.ylabel("frequency")
        plt.title("frequency of each class")
        # plt.show()
        data = data.drop('lender_id', axis=1)
        categorical_features = data.select_dtypes(include=['object'])
        categorical_features = categorical_features.drop('lead_uuid', axis=1)
        for column in categorical_features:
            print('********************')
            print('frequency of values for column {0} is:\n{1}'.format(column, data[column].value_counts()))
            print('********************')

        for column in categorical_features:
            fig, ax = plt.subplots(figsize=(20, 7.5))
            sns.countplot(data[column], ax=ax, order=(data[column].value_counts()).index)
            plt.xlabel(column, fontsize=15)
            plt.ylabel('Frequency', fontsize=15)
            plt.xticks(rotation=90, fontsize=8)
            plt.subplots_adjust(hspace=0.7)
            # plt.show()

        loan_purpose_others = ['unknown',
                               'green',
                               'emergency',
                               'life_event',
                               'car_repair',
                               'cosmetic',
                               'student_loan_refi',
                               'home_purchase',
                               'motorcycle']

        data.loc[data['loan_purpose'].isin(loan_purpose_others), 'loan_purpose'] = 'other'
        for column in categorical_features:
            fig, ax = plt.subplots(figsize=(20, 7.5))
            sns.countplot(data[column], ax=ax, order=(data[column].value_counts()).index)
            plt.xlabel(column, fontsize=15)
            plt.ylabel('Frequency', fontsize=15)
            plt.xticks(rotation=90, fontsize=8)
            plt.subplots_adjust(hspace=0.7)
            # plt.show()

        features = data.drop('clicked', axis=1)
        numeric_data = features.drop(['loan_purpose', 'credit', 'lead_uuid', 'offer_id'], axis=1)
        fig, axes = plt.subplots(1, 3, figsize=(10, 10))
        for i in range(len(numeric_data.columns)):
            data[numeric_data.columns[i]].hist(legend=numeric_data.columns[i], ax=axes[i], bins=50)
        # plt.show()

        for column in numeric_data.columns:
            data = self.outlier_detection(column, data)

        sns.pairplot(data)
        # plt.show()
        data = self.convert_ordinal_features(data, 'credit')
        categorical_features = data.select_dtypes(include=['object'])
        categorical_features = categorical_features.drop('lead_uuid', axis=1)
        features = data.drop('clicked', axis=1)
        numeric_data = features.drop(['loan_purpose', 'lead_uuid', 'offer_id'], axis=1)
        encoded_categorical_data = self.convert_categorical_features(categorical_features)
        print(data.isna().sum())
        df = pd.concat((data['lead_uuid'], data['offer_id'], encoded_categorical_data, numeric_data, data['clicked']), axis=1)

        print(df.isna().sum())
        df_features = df.drop(['lead_uuid', 'offer_id'], axis=1)
        # df_features = self.calculate_vif(df_features)
        df_features = self.linear_correlation(df_features)
        df = pd.concat((df['lead_uuid'], df['offer_id'], df_features), axis=1)
        df.to_csv(data_path + 'processed_data.csv', index=False)

    @staticmethod
    def outlier_detection(col, data):
        first_quartile = data[col].describe()['25%']
        third_quartile = data[col].describe()['75%']
        iqr = third_quartile - first_quartile
        data = data[
            (data[col] > first_quartile - 3 * iqr)
            & (data[col] < third_quartile + 3 * iqr)]
        return data

    @staticmethod
    def convert_categorical_features(categorical_subset):
        categorical_subset_df = pd.get_dummies(categorical_subset)
        return categorical_subset_df

    @staticmethod
    def convert_ordinal_features(df, column):
        credit_mapper = {'excellent': 5, 'good': 4, 'fair': 3, 'poor': 2, 'limited': 1, 'unknown': 1}
        df[column] = df[column].replace(credit_mapper)
        return df

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        variables = list(range(X.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
                   for ix in range(X.iloc[:, variables].shape[1])]

            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                      '\' at index: ' + str(maxloc))
                del variables[maxloc]
                dropped = True

        print('Remaining variables:')
        print(X.columns[variables])
        return X.iloc[:, variables]

    @staticmethod
    def linear_correlation(df):
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        print('Features with corr more than 0.95 {}'.format(to_drop))
        df.drop(to_drop, axis=1, inplace=True)
        return df


if __name__ == '__main__':
    data_path = os.path.join(ROOT_PATH, 'data/')
    schema_path = os.path.join(data_path, 'schemas/')
    loaded_path = os.path.join(data_path, 'loaded/')

    data_processor = DataProcessor()
    if not os.path.isdir(loaded_path):
        os.mkdir(loaded_path)
        data_processor.read_data_from_database(schema_path, loaded_path)
    data = data_processor.generate_data_frame(loaded_path)
    data_processor.preprocess(data)
