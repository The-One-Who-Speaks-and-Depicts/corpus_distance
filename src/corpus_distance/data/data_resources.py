from pandas import read_csv
from pkg_resources import resource_stream

data_stream = resource_stream(__name__, "gospels.csv")
data_df = read_csv(data_stream)
