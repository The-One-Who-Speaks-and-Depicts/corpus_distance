"""
A wrapping module for the default data,
which reads the Croatian/Slovak/Slovenian dataset
in .csv format, and transfers it
to the main module as pandas dataframe;
Also a wrapping module for the default configuration
"""


import json
from pandas import read_csv
from importlib import resources


data_stream = resources.files(__name__).joinpath("gospels.csv")
data_df = read_csv(data_stream)

config_stream = resources.files(__name__).joinpath("config.json").read_text(encoding='utf-8')
config = json.loads(config_stream)
