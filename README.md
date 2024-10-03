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