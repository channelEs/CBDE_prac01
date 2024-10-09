from configparser import ConfigParser
from connect import postgres_conn as postG
from datasets import load_dataset

# script to load PostGres DataBase configuration
def load_config(filename='database.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)

    # get section, default to postgresql
    config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return config

if __name__ == '__main__':
    config = load_config()
    connection = postG()
    print(config)
    print(connection)
    connection.post_connect(config)

    # dataset = load_dataset("bookcorpus", split="train")
    # print(dataset)
    # print(dataset[4])
    # print(dataset[20])

    connection.close_connection()