from sqlalchemy import text as sql_text
import sqlalchemy as db
import pandas as pd
import pymysql

class MYSQL_DB_API:
    def __init__(self, host, port, user, password, database):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.engine = None
        self.connect()
    
    def connect(self):
        self.connection = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )
        self.engine = db.create_engine("mysql+pymysql://", creator=lambda: self.connection)
    
    def disconnect(self):
        self.connection.close()
    
    def load_data(self, query):
        data = pd.read_sql_query(query, self.engine)
        return data

    def count_data(self, query):
        with self.engine.connect() as conn:
            count_results = conn.execute(query).fetchall()[0][0]
            return count_results
    
    def insert_data(self, table_name: str, df: pd.DataFrame, dtype=None):
        with self.engine.connect() as conn:
            df.to_sql(table_name, conn, if_exists='append', index=False, dtype=dtype)