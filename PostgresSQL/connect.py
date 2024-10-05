import psycopg2

class postgres_conn:

    def connect(self, config):
        """ Connect to the PostgreSQL database server """
        try:
            # connecting to the PostgreSQL server
            with psycopg2.connect(**config) as conn:
                print('Connected to the PostgreSQL server.')
                self.conn = conn
                self.cursor = conn.cursor
        except (psycopg2.DatabaseError, Exception) as error:
            print(error)

    def execute_query(self, postgres_query):
        try:
            self.cursor.execute(postgres_query) 
            self.conn.commit()
            print("Query successfully.")
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error query: {e}")

    def close_connection(self):
        self.cursor.close()
        self.conn.close()

    def commit_trans(self):
        self.conn.commit()
