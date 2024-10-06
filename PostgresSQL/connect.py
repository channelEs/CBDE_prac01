import psycopg2

class postgres_conn:
    def __init__(self):
        self.conn = None
        self.cursor = None

    def post_connect(self, config):
        """ Connect to the PostgreSQL database server """
        try:
            # connecting to the PostgreSQL server
            with psycopg2.connect(**config) as conn:
                print('Connected to the PostgreSQL server.')
                self.conn = conn
                self.cursor = conn.cursor()
        except (psycopg2.DatabaseError, Exception) as error:
            print(error)

    def execute_query(self, postgres_query, params=None):
        try:
            if params:
                self.cursor.execute(postgres_query, params)
            else:
                self.cursor.execute(postgres_query)
            self.conn.commit()  # Commit changes only if successful
            # print("Query executed successfully.")
        except (psycopg2.DatabaseError, Exception) as error:
            print(f"Error while executing query: {error}")
            self.conn.rollback()  # Rollback the transaction in case of an error
        
    def cursor_fetch(self):
        return self.cursor.fetchall()
            

    def close_connection(self):
        self.cursor.close()
        self.conn.close()

    def commit_trans(self):
        self.conn.commit()
