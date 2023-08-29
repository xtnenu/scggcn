import torch
import pandas as pd
import Dataprocess

def result2csv():
    pred=torch.load("./data/gene_pred.pkl")
    pred=pred.flatten()
    gene_name=Dataprocess.get_genenames()
    df = pd.DataFrame({'Col1': gene_name, 'Col2': pred})
    sorted_df = df.sort_values('Col2')
    sorted_df.to_csv("casestudy.csv",index=False)