import pandas as pd

from app.tabio_engine import convert_to_json


def test_convert_to_list():
    """
    tests converting table_extract dataframe to json
    """
    dummy_data = [((1,2,3,4), [pd.DataFrame([10,10,10,10])])]
    result = convert_to_json(dummy_data)
    assert(len(result) == 1)
    result = result[0]
    assert(result["top"] == 1)
    assert(result["left"] == 2)
    assert(result["bottom"] == 3)
    assert(result["right"] == 4)
    assert(len(result["table_data"]) == 1)
