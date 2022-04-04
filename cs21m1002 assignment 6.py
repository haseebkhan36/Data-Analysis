#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ## Candidate Itemset Generation

# In[2]:


from itertools import combinations

def genCandidateSet(level_k, level_fi):
    n_fi = len(level_fi)
    candidate_fi = []

    for i in range(n_fi):
        j = i+1
        while(j < n_fi) and (level_fi[i][:level_k-1] == level_fi[j][:level_k-1]):
            candidate_set = level_fi[i][:level_k-1] + [level_fi[i][level_k-1]] + [level_fi[j][level_k-1]]
            candidate_set_pass = False

            if(level_k == 1):
                candidate_set_pass = True
            elif(level_k == 2) and (candidate_set[-2:] in level_fi):
                candidate_set_pass = True
            elif(all((list(a) + candidate_set[-2:]) in level_fi for a in combinations(candidate_set[:-2], level_k))):
                candidate_set_pass = True
            
            if(candidate_set_pass):
                candidate_fi.append(candidate_set)
            j+=1

    return candidate_fi


# # Transaction Reduction Technique

# In[3]:


def transactionReduction(transactions, min_supp, verbose=False):
    
    items = set()
    
    print("Transactions")
    for t in transactions:
        print(t)
        items.update(t)
    print("Number of transaction = {}".format(len(transactions)))

    items = sorted(list(items))

    fi = []

    level_k = 1
    level_fi = []

    candidate_fi = [[item] for item in items]

    while candidate_fi:
        if verbose:
            print("\n-----------------------------------------------------------------------------")
            print("Level {}: Frequent Itemsets".format(level_k-1))
            for i in level_fi:
                print(i)
            print()
            print("Level {}: Pruned Transactions".format(level_k))

        candidate_fi_count = [0] * len(candidate_fi)
        current_index = 0
        for t in transactions:
            # number of candidate itemsets part of the transaction
            n_ci_transaction = 0

            for i, itemset in enumerate(candidate_fi):
                if(all(_item in t for _item in itemset)):
                    candidate_fi_count[i] += 1
                    n_ci_transaction += 1

            if(n_ci_transaction > 1):
                transactions[current_index] = t
                current_index += 1

            elif verbose:
                print(t)

        if verbose:
            print("Number of transactions pruned = {}".format(len(transactions)-current_index), end='\n\n')

        transactions = transactions[:current_index]
        print("After level {}: Number of transactions = {}".format(level_k, current_index))

        if(verbose):
            print("-----------------------------------------------------------------------------")
        
        level_fi = [itemset for itemset, support in zip(candidate_fi, candidate_fi_count) if support >= min_supp]
        fi.extend([set(i) for i in level_fi])

        candidate_fi = genCandidateSet(level_k, level_fi)
        level_k += 1
    return fi


# ### Dataset

# In[4]:


transactions = [
                    ['A', 'B'],
                    ['B', 'C', 'D'],
                    ['A', 'C', 'D', 'E'],
                    ['A', 'D', 'E'],
                    ['A', 'B', 'C'],
                    ['A', 'B', 'C', 'D'],
                    ['B', 'A'],
                    ['A', 'B', 'C'],
                    ['A', 'B', 'D'],
                    ['B', 'C', 'E']
                ]

min_supp = 4
verbose = True
fi = transactionReduction(transactions=transactions, min_supp=min_supp, verbose=verbose)
print("\nFrequent Itemsets (Min Support Count = {})".format(min_supp))
for i in fi:
    print(i)


# ### Benchmark dataset

# In[6]:


file = open("mushrooms.csv", "r")

dataset = []                              
for line in file:
  dataset.append(line.strip().split(' ')) 
                                      
file.close()


# In[7]:


min_supp = 4
verbose = False
fi = transactionReduction(transactions=dataset, min_supp=len(dataset)/2, verbose=verbose)
print("\nFrequent Itemsets (Min Support Count = {})".format(min_supp))
for i in fi:
    print(i)


# ## 2.  Vertical Transaction Aproach (Intersection)

# In[9]:


def basic_eclat(vertical_db, items, min_supp):
    fi = []
    level_k = 1
    level_fi = []
    candidate_fi = [[item] for item in items]

    while candidate_fi:
        candidate_fi_count = [0] * len(candidate_fi)

        if(level_k == 1):
            for i, itemset in enumerate(candidate_fi):
                candidate_fi_count[i] = len(vertical_db[itemset[0]])
        else:
            for i, itemset in enumerate(candidate_fi):
                t = set(vertical_db[itemset[0]])
                for item in itemset[1:]:
                    t = t.intersection(set(vertical_db[itemset[0]]).intersection(vertical_db[item]))#) #set(vertical_db[item]) for item in itemset))
                candidate_fi_count[i] = len(t)

        level_fi = [itemset for itemset, support in zip(candidate_fi, candidate_fi_count) if support >= min_supp]
        fi.extend([set(i) for i in level_fi])

        candidate_fi = genCandidateSet(level_k, level_fi)
        level_k += 1

    return fi


# In[10]:


data = {
    'A' : [1, 3, 4, 5, 6, 7, 8, 9],
    'B' : [1, 2, 5, 6, 7, 8, 9, 10],
    'C' : [2, 3, 5, 6, 8, 10],
    'D' : [2, 3, 4, 6, 9],
    'E' : [3, 4, 10]
}
print(len(data['A']))

items = ['A', 'B', 'C', 'D', 'E']
min_supp = 4
fi = basic_eclat(data, items, min_supp)
print("\nFrequent Itemsets (Min Support Count = {})".format(min_supp))
for i in fi:
    print(i)


# ## 3 Hash based Apriori

# In[11]:


class hashtable:
    def __init__(self, hash_table_size):
        self.size = hash_table_size
        self.hash_table = [0] * hash_table_size

    def add_itemset(self, itemset):
        hash_index = (itemset[0] * 10 + itemset[1]) % self.size
        self.hash_table[hash_index] += 1

    def get_itemset_count(self, itemset):
        hash_index = (itemset[0] * 10 + itemset[1]) % self.size
        return self.hash_table[hash_index]


# In[12]:


def apriori_hash(transactions, min_supp):
    items = set()
    for t in transactions:
        items.update(t)

    items = sorted(list(items))

    fi = []
    level_k = 1
    level_fi = []
    candidate_fi = [[item] for item in items]

    hash_tb = hashtable(7)

    while(candidate_fi):
        candidate_fi_count = [0] * len(candidate_fi)
        
        for t in transactions:
            if level_k == 1:
                for itemset in combinations(t, 2):
                    hash_tb.add_itemset(itemset)
            
            for i, itemset in enumerate(candidate_fi):
                if(all(a in t for a in itemset)):
                    candidate_fi_count[i] += 1

        level_fi = [itemset for itemset, support in zip(candidate_fi, candidate_fi_count) if support >= min_supp]
        fi.extend([set(f_item) for f_item in level_fi])

        candidate_fi = genCandidateSet(level_k, level_fi)
        level_k += 1

        if(level_k == 2):
            for itemset in candidate_fi:
                if(hash_tb.get_itemset_count(itemset) < min_supp):
                    print("Pruned itemset", itemset)
                    candidate_fi.remove(itemset)
    
    return fi


# In[23]:


transactions = [
                    [1, 2],
                    [2, 3, 4],
                    [1, 3, 4, 5],
                    [1, 4, 5],
                    [1, 2, 3],
                    [1, 2, 3, 4],
                    [2, 1],
                    [1, 2, 3],
                    [1, 2, 4],
                    [2, 3, 5]
                ]

min_supp = 4
verbose = True
fi = apriori_hash(transactions=transactions, min_supp=min_supp)
print("\nFrequent Itemsets (Min Support Count = {})".format(min_supp))
for i in fi:
    print(i)


# ## 4. FP-Max Algorithm

# In[1]:


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpmax

te = TransactionEncoder()

df = pd.read_csv("mushroom_encoded.csv")
freq_itemsets = fpmax(df, min_support=0.4, use_colnames=True)
freq_itemsets


# In[ ]:




