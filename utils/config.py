import configparser


def get_openapi_key():
    # get open api key
    config = configparser.ConfigParser()
    config.read('config.ini')
    """ config.int
    [OPENAPI_KEY]
    API_KEY = xxx
    Organization_ID = ooo
    """
    OPENAI_API_KEY = config["OPENAPI_KEY"]["API_KEY"]
    ORGANIZATION_ID = config["OPENAPI_KEY"]["Organization_ID"]
    return OPENAI_API_KEY, ORGANIZATION_ID
