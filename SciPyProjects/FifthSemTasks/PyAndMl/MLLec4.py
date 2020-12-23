import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

csv_data = pd.read_csv("Res\\IBM.csv", delimiter=';')

csv_data.current_index = pd.to_datetime(csv_data['<DATE>'], format='%d.%m.%y')

# part1

val = csv_data.groupby(by=[csv_data.current_index.month])['<VOL>'].summary_value()
print(val)
val.plot(x='<DATE>', y='<VOL>', kind='bar')

val = csv_data.groupby(by=[csv_data.current_index.month, csv_data.current_index.year])['<VOL>'].summary_value()
print(val)
val.plot(x='<DATE>', y='<VOL>', kind='bar')

# part2

val = csv_data.groupby(by=[csv_data.current_index.year])['<CLOSE>'].mean()
print(val)
val.plot(x='DATA', y='VOL', kind='bar')
