from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, Float
import sqlite3

# Global Variables
SQLITE                  = 'sqlite'
# https://medium.com/@mahmudahsan/how-to-use-python-sqlite3-using-sqlalchemy-158f9c54eb32
# Table Names

class MyDatabase:
    # http://docs.sqlalchemy.org/en/latest/core/engines.html
    DB_ENGINE = {
        SQLITE: 'sqlite:///{DB}'
    }

    # Main DB Connection Ref Obj
    db_engine = None
    def __init__(self, dbtype, username='', password='', dbname=''):
        dbtype = dbtype.lower()
        if dbtype in self.DB_ENGINE.keys():
            engine_url = self.DB_ENGINE[dbtype].format(DB=dbname)
            self.db_engine = create_engine(engine_url)
            print(self.db_engine)
        else:
            print("DBType is not found in DB_ENGINE")
        print("DBType is  found in DB_ENGINE YAYYYYYY.....")

    def execute_query(self, query=''):
        if query == '': return
        # print(query)
        with self.db_engine.connect() as connection:
            try:
                connection.execute(query)
            except Exception as e:
                print(e)

    def insertmany_sqlite3(self,table='',columns='',data=''):
        for values in data:
            # query = "INSERT INTO " + table + ' (' + columns + ') VALUES ' + row + ";"
            query = "INSERT INTO {} ({}) VALUES ({});".format(table,columns,values)
            # print(query)
            self.execute_query(query)

    def get_count_result(self, query):
        with self.db_engine.connect() as connection:
            try:
                result = connection.execute(query)
            except Exception as e:
                print(e)
            else:
                data = result.fetchall()[0][0]
                # print('--',data)
                # for row in result:
                #     # print(row)  # print(row[0], row[1], row[2])
                #     data.append(row)
                result.close()
                return data

    def get_face_attributes(self,query):
        with self.db_engine.connect() as connection:
            try:
                result = connection.execute(query)
            except Exception as e:
                print(e)
            else:
                # data = result.fetchall()[0][0]
                # print('--',data)
                data = []
                for row in result:
                    # print(row)  # print(row[0], row[1], row[2])
                    data.append(row)
                result.close()
                return data

    def update_face_attribute(self,query):
        self.execute_query(query)
