import mysql.connector
import os
import pandas as pd
import pyarrow.parquet as pq

from sqlalchemy import create_engine


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
CLICKS_FILE_NAME = 'ds_clicks.parquet.gzip'
LEADS_FILE_NAME = 'ds_leads.parquet.gzip'
OFFERS_FILE_NAME = 'ds_offers.parquet.gzip'


class SchemaManager:

    def define_data_schema(self, data_path, schema_path):
        clicks_path = os.path.join(data_path, CLICKS_FILE_NAME)
        leads_path = os.path.join(data_path, LEADS_FILE_NAME)
        offers_path = os.path.join(data_path, OFFERS_FILE_NAME)
        parquet_clicks = pq.read_table(clicks_path)
        print("Column names: {}".format(parquet_clicks.column_names))
        print("Schema -> data types: {}".format(parquet_clicks.schema.types))
        print('********************')
        click_cols_dict = {'col_name': parquet_clicks.column_names, 'data_types': parquet_clicks.schema.types}
        clicks_df = pd.DataFrame.from_dict(click_cols_dict, orient='index')
        clicks_schema_path = schema_path + 'clicks_schema.csv'
        clicks_df.to_csv(clicks_schema_path, index=False)

        parquet_leads = pq.read_table(leads_path)
        print("Column names: {}".format(parquet_leads.column_names))
        print("Schema -> data types: {}".format(parquet_leads.schema.types))
        print('********************')
        leads_cols_dict = {'col_name': parquet_leads.column_names, 'data_types': parquet_leads.schema.types}
        leads_df = pd.DataFrame.from_dict(leads_cols_dict, orient='index')
        leads_schema_path = schema_path + 'leads_schema.csv'
        leads_df.drop(leads_df.columns[5], axis=1, inplace=True)
        leads_df.to_csv(leads_schema_path, index=False)

        parquet_offers = pq.read_table(offers_path)
        print("Column names: {}".format(parquet_offers.column_names))
        print("Schema -> data types: {}".format(parquet_offers.schema.types))
        print('********************')
        offers_cols_dict = {'col_name': parquet_offers.column_names, 'data_types': parquet_offers.schema.types}
        offers_df = pd.DataFrame.from_dict(offers_cols_dict, orient='index')
        offers_schema_path = schema_path + 'offers_schema.csv'
        offers_df.drop(offers_df.columns[4], axis=1, inplace=True)
        offers_df.to_csv(offers_schema_path, index=False)

    def create_sql_tables(self):
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123Saeed",
            database="even_financial"
        )

        my_cursor = mydb.cursor()

        sql_1 = "DROP TABLE clicks"
        sql_2 = "DROP TABLE leads"
        sql_3 = "DROP TABLE offers"

        my_cursor.execute(sql_1)
        my_cursor.execute(sql_2)
        my_cursor.execute(sql_3)

        my_cursor.execute("CREATE TABLE clicks (offerId BIGINT, clickedAt TIMESTAMP)")
        my_cursor.execute("CREATE TABLE leads (leadUUId VARCHAR(255),requested  DOUBLE, loanPurpose VARCHAR(255), credit VARCHAR(255), anualIncome DOUBLE)")
        my_cursor.execute("CREATE TABLE offers (leadUUId VARCHAR(255), offerId BIGINT, apr DOUBLE, lenderId BIGINT)")

    def insert_to_databse(self, data_path):
        clicks_path = os.path.join(data_path, CLICKS_FILE_NAME)
        leads_path = os.path.join(data_path, LEADS_FILE_NAME)
        offers_path = os.path.join(data_path, OFFERS_FILE_NAME)
        df_click = pd.read_parquet(clicks_path, engine='pyarrow')
        df_click.columns = ['offerId', 'clickedAt']
        df_lead = pd.read_parquet(leads_path, engine='pyarrow')
        df_lead.columns = ['leadUUId', 'requested', 'loanPurpose', 'credit', 'anualIncome']
        df_offer = pd.read_parquet(offers_path, engine='pyarrow')
        df_offer.columns = ['leadUUId', 'offerId', 'apr', 'lenderId']

        engine = create_engine('mysql://root:123Saeed@localhost:3306/even_financial', echo=False)

        df_click.to_sql(name='clicks', con=engine, if_exists='append', index=False)
        df_lead.to_sql(name='leads', con=engine, if_exists='append', index=False)
        df_offer.to_sql(name='offers', con=engine, if_exists='append', index=False)


if __name__ == '__main__':

    data_path = os.path.join(ROOT_PATH, 'data/')
    schema_path = os.path.join(data_path, 'schemas/')

    schema_manager = SchemaManager()
    if not os.path.isdir(schema_path):
        os.mkdir(schema_path)
        schema_manager.define_data_schema(data_path, schema_path)
        schema_manager.create_sql_tables()
    schema_manager.insert_to_databse(data_path)

