import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from sklearn import metrics

from Dataprocess import Node_dect,k_set,maskall
from Network import Gcn,Gcn_bayes,Gat,ChebNet,Gat_bayes


def main(pred=True,epoch=600):

    data = torch.load("./data/CPDB_data.pkl")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)
    mask_all, Y = Node_dect()
    Y = Y.to(device)
    mask_all, amask_all = maskall()

    datas = torch.load("./data/PPI_lst.pickle")
    # datas=torch.load("./data/sdne_ppi.pickle")
    # datas=torch.load("./data/Node_ppi.pickle")
    # datas=torch.load("./data/struc_ppi.pickle")
    data.x = datas
    data = data.to(device)

    if pred==False:
        k_sets = k_set()
        AUC = np.zeros(shape=(10, 1))
        AUPR = np.zeros(shape=(10, 1))
        for i in range(10):
            print(i)
            for cv_run in range(5):
                _, _, tr_mask, te_mask = k_sets[i][cv_run]
                model = ChebNet(128,data).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(epoch):
                    train(tr_mask)

                AUC[i] = test2(te_mask,model,Y)

        print(AUC.mean())
        print(AUC.var())
        print(AUPR.mean())
        print(AUPR.var())
    else:
        model = ChebNet(128, data).to(device)
        for epoch in range(epoch):
            print(epoch)
            train(mask_all,model,Y)
        torch.save(model, "ssggcn.pkl")

        x, _, _, _ = model()
        pred_result = torch.sigmoid(x[amask_all]).cpu().detach().numpy()
        print(pred_result)
        torch.save(pred_result, 'gene_pred.pkl')


def train(mask,model,Y):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    pred, rl, c1, c2 = model()
    loss = F.binary_cross_entropy_with_logits(pred[mask], Y[mask]) / (c1 * c1) + rl / (c2 * c2) + 2 * torch.log(c2 * c1)
    #loss=F.binary_cross_entropy_with_logits(pred[mask], Y[mask]) #Without Bayes
    loss.backward()
    optimizer.step()


def train_baseline(mask,model):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    pred=model()
    loss = F.binary_cross_entropy_with_logits(pred[mask], Y[mask])
    loss.backward()
    optimizer.step()

@torch.no_grad()
def test(mask,model,Y):
    model.eval()
    x, _, _, _ = model()
    pred = torch.sigmoid(x[mask]).cpu().detach().numpy()
    Yn = Y[mask].cpu().numpy()
    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(Yn, pred), area

@torch.no_grad()
def test_baseline(mask,model):
    model.eval()
    x= model()
    #pred = torch.sigmoid(x[mask]).cpu().detach().numpy()
    pred=x[mask].cpu().detach().numpy()
    Yn = Y[mask].cpu().numpy()
    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(Yn, pred), area

@torch.no_grad()
def test2(mask,model,Y):
    model.eval()
    x, _, _, _ = model()

    pred = torch.sigmoid(x[mask]).cpu().detach().numpy()
    Yn = Y[mask].cpu().numpy()
    MSE = metrics.mean_squared_error(Yn, pred)
    RS=metrics.r2_score(Yn, pred)
    print(RS)
    #area = metrics.auc(recall, precision)

    return MSE,RS

if __name__=="__main__":
    main()




