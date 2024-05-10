"""
A wrapping module for the default data,
which reads the Croatian/Slovak/Slovenian dataset
in .csv format, and transfers it
to the main module as pandas dataframe
"""



from pandas import read_csv
from pkg_resources import resource_stream



data_stream = resource_stream(__name__, "gospels.csv")
data_df = read_csv(data_stream)
