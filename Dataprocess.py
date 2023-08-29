import csv
import torch
import numpy as np
import random

def random_partition(arr, num_partitions):
    random.seed(42)
    random.shuffle(arr)
    partition_size = len(arr) // num_partitions
    partitions = [arr[i:i+partition_size] for i in range(0, len(arr), partition_size)]
    remainder = len(arr) % num_partitions
    if remainder > 0:
        for i in range(remainder):
            partitions[i].append(arr[-(i+1)])
    return partitions


def get_genenames():
    f = open("./data/node_names.txt", 'r')
    gene_lst=[]
    gene_lst2=[]
    for i in f:
        lst=i.rstrip("\n").split(",")
        gene_lst.append(lst[1])
    _,amask=maskall()
    for i in range(len(amask)):
        if amask[i]==True:
            gene_lst2.append(gene_lst[i])
    return gene_lst2

def Node_dect(filepath="./data/C0036341_disease_gda_summary.csv",threshold=0.6):
    f = open("./data/node_names.txt", 'r')
    f2 = open(filepath, 'r')
    scz=csv.reader(f2)
    sczlst=[]
    mask=[]
    label={}
    labelst=[]
    for i in scz:
        sczlst.append(i[2])
        label[i[2]]=i[8]
    for i in f:
        lst=i.rstrip("\n").split(",")
        if lst[1] in sczlst:
            mask.append(1)
            if float(label[lst[1]])> threshold:
                labelst.append([1.0])
            else:
                labelst.append([0.0])
        else:
            if len(lst)>2 and lst[2] in sczlst:
                print(lst[2])
                mask.append(1)
                if float(label[lst[2]])>threshold:
                    labelst.append([1.0])
                else:
                    labelst.append([0.0])
            else:
                mask.append(0)
                labelst.append([0.0])
    labelst=torch.tensor(labelst)
    return mask,labelst

def mask2bools(tr,te,length):
    trlst=[]
    telst=[]
    for i in range(length):

        if i in tr:
            trlst.append(True)
            telst.append(False)
        elif i in te:
            trlst.append(False)
            telst.append(True)
        else:
            trlst.append(False)
            telst.append(False)
    return trlst,telst

def k_set():
    kset={}
    mask,_=Node_dect()
    data=[]
    i=0
    for j in mask:
        if j==0:
            i=i+1
        else:
            data.append(i)
            i=i+1
    print(len(data))
    partitions = random_partition(data, 10)
    for i, partition in enumerate(partitions):
        print(f"Partition {i+1}: {partition}")
        tr=list(set(data)-set(partition))
        print(len(partition))
        print(len(tr))
        t,t2=mask2bools(tr,partition,13627)
        kset[i]=(t,t2)
    return kset

def maskall():
    lst=[]
    lst2=[]
    mask, _ = Node_dect()
    data = []
    i = 0
    for j in mask:
        if j == 0:
            i = i + 1
        else:
            data.append(i)
            i = i + 1
    for k in range(13627):
        if k in data:
            lst.append(True)
            lst2.append(False)
        else:
            lst.append(False)
            lst2.append(True)
    return lst,lst2

