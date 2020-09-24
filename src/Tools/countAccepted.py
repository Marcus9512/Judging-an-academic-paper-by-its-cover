import numpy as np
import pandas as pd
import pathlib


path = pathlib.Path.cwd()
path = path.parent.parent / ('data' + path.suffix) / ('meta.csv' + path.suffix)
data = pd.read_csv(path)
accepted_papers = data.accepted.value_counts()
print(accepted_papers)