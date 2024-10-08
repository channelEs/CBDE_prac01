# CBDE_prac01
private repo for the first CBDE delivery

## venv
### connection
```
source venv/bin/activate
```
### install requirements
(once in venv enviroment)
```
pip install -r requirements.txt
```
## PostgresSQL
### execute connection
Create inside ./PostgresSQL a new file called 'database.ini'. There you need to configure the postgres connection using this structure:
```
[postgresql]
host=
database=
user=
password=
```
Then you can connect to the databse
```
python3 connect.py
```

### connect via command line
enter database
```
psql -h *localhost* -d *databasename* -U *user*
```

list tables
```
\dt
\d <table-name>
```

export tables
```
\copy (*SELECT_QUERY*) to 'export.csv' with csv header
```

## Chroma
Only run Chroma scripts with an existing Postgre database, since Chroma fetches data from Postgres "bookcorpus" table!
Before the execution, make sure that the script has rights to write and read from the same folder, due to Chroma local storage requirements.
### Order of operations:
```
py chroma_load_data.py
```

```
py chroma_load_embeddings.py
```

```
py chroma_calcs.py
```

The waiting time might be longer then expected. It's not broken, it's just thinking!
