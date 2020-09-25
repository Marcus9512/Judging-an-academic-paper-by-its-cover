import pandas as pd
import pathlib

def count_papers(path):
    #Path koden som är nedan fungerar, men kanske är bättre att bara ta emot path
    #path = pathlib.Path.cwd()
    #path = path.parent.parent / ('data' + path.suffix) / ('meta.csv' + path.suffix)
    data = pd.read_csv(path)
    accepted_papers = data.accepted.value_counts()[1]
    rejected_papers = data.accepted.value_counts()[0]

    return accepted_papers, rejected_papers