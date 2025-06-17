
from datetime import datetime


def get_timestamp():
    """
    Retorna la Fecha-Hora actual en el formato "%Y-%m-%d_%H.%M.%S". 
    """
    current_datetime = datetime.now()

    formatted_timestamp = current_datetime.strftime("%Y-%m-%d_%H.%M.%S")

    return formatted_timestamp