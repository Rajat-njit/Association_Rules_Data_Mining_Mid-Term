#!/usr/bin/env python
# coding: utf-8

# ## Data Mining : MidTerm Project

# <b>Name:</b> Rajat Pednekar<br>
# <b>Professor:</b> Yasser Abduallah<br>
# <b>UCID:</b> rp2348<br>
# <b>Email_Id:</b> rp2348@njit.edu<br>

# <b>Topic:</b> A Comparative Study of Brute-Force, Apriori, and FP-Growth Algorithms

# <b>Introduction:</b><br>This project details the implementation of three algorithms—Brute-Force, Apriori, and FP-Growth—for mining frequent itemsets and generating association rules from transactional data. The purpose is to compare the efficiency and effectiveness of each algorithm in generating frequent itemsets and rules while measuring their execution time under various support and confidence thresholds.

# <b>Import Statements</b><br>
# This cell imports the necessary libraries:
# 1. itertools for generating combinations
# 2. time for measuring execution time
# 3. pandas for data manipulation and analysis
# 4. apriori, fpgrowth, and association_rules from mlxtend.frequent_patterns for implementing the Apriori and FP-Growth algorithms

# <I>Note: Before executing the cell install pandas and mlxtend libraries.<I>


# In[50]:


import itertools
import time
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


# <b>Calculate Support Function</b><br>
# This function calculates the support for a given itemset in a set of transactions. Support is the fraction of transactions that contain the itemset. It's a crucial metric in association rule mining.

# In[52]:


# Function to calculate support for an itemset.

def calculate_support(transactions, itemset):
    return sum(1 for transaction in transactions if set(itemset).issubset(set(transaction))) / len(transactions)


# <b>Brute Force Frequent Itemsets Function</b><br>
# This function implements a brute force approach to find the frequent itemsets.
# 1. It starts with individual items and progressively increases the itemset size.
# 2. For each size, it generates all possible combinations of items.
# 3. It calculates the support for each combination and keeps those that meet the minimum support threshold.
# 4. The process continues until no new frequent itemsets are found.

# In[54]:


# Brute Force algorithm to generate frequent itemset.

def brute_force_frequent_itemsets(transactions, min_support):
    items = sorted(set(item for transaction in transactions for item in transaction)) # get the unique items from the transactions.
    frequent_itemsets = []
    itemset_size = 1
    
    while True:
        candidate_itemsets = list(itertools.combinations(items, itemset_size))     # Generate all combinations of itemset from current size.  
        current_frequent_itemsets = []
        
        for itemset in candidate_itemsets:
            support = calculate_support(transactions, itemset)
            if support >= min_support:
                current_frequent_itemsets.append((itemset, support))
        
        if not current_frequent_itemsets:
            break
        
        frequent_itemsets.extend(current_frequent_itemsets)
        itemset_size += 1

    return frequent_itemsets


# <b>Generate Association Rules Function</b><br>
# This function generates association rules from the frequent itemsets:
# 1. For each frequent itemset with more than one item, it generates all possible antecedent-consequent pairs.
# 2. It calculates the confidence for each rule.
# 3. Rules that meet the minimum confidence threshold are kept and returns a list of rules with <br>
# their antecedents, consequents, confidence and support.

# In[56]:


def generate_association_rules(frequent_itemsets, transactions, min_confidence):
    rules = []
    for itemset, itemset_support in frequent_itemsets:
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, i):
                    consequent = tuple(item for item in itemset if item not in antecedent)
                    antecedent_support = calculate_support(transactions, antecedent)
                    confidence = itemset_support / antecedent_support
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence, itemset_support))
    return rules


# <b>Prepare Transaction DataFrame Function</b><br>This function prepares the transaction data for use with the Apriori and FP-Growth algorithms from mlxtend:
# 
# 1. It creates a list of all unique items across all transactions.<br>
# 2. t then creates a boolean DataFrame where each row represents a transaction and each column represents an item.<br>
# 3. A True value indicates the presence of an item in a transaction, while False indicates its absence.<br>

# This format is required for the mlxtend implementation of Apriori and FP-Growth algorithms.

# In[59]:


def prepare_transaction_df(transactions):
    items = sorted(set(item for transaction in transactions for item in transaction))
    return pd.DataFrame([[item in transaction for item in items] for transaction in transactions], columns=items).astype(bool)


# <b>Run All Algorithms Function</b><br>
# This function runs all three algorithms (Brute Force, Apriori, and FP-Growth) on the given transactions:
# 1. It first prepares the transaction data in the required format.
# 2. For each algorithm, it:
# - Measures the execution time
# - Finds frequent itemsets
# - Generates association rules
# - Prints the results
# 3. It handles potential errors in the Apriori and FP-Growth algorithms.
# 4. Finally, it returns the execution times for all three algorithms.

# In[61]:


def run_all_algorithms(transactions, min_support, min_confidence):
    transaction_df = prepare_transaction_df(transactions)

    # Brute force algorithm
    print("\nRunning Brute Force Algorithm...")
    start_time = time.time()
    frequent_itemsets_brute = brute_force_frequent_itemsets(transactions, min_support)
    rules_brute = generate_association_rules(frequent_itemsets_brute, transactions, min_confidence)
    brute_time = time.time() - start_time
    print(f"Brute Force Time: {brute_time:.4f} seconds")
    print_results("Brute Force", frequent_itemsets_brute, rules_brute)

    # Apriori Algorithm
    print("\nRunning Apriori Algorithm...")
    start_time = time.time()
    try:
        frequent_itemsets_apriori = apriori(transaction_df, min_support=min_support, use_colnames=True)
        apriori_time = time.time() - start_time
        
        if frequent_itemsets_apriori.empty:
            print("Apriori did not find any frequent itemsets.")
        else:
            rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence)
            print(f"Apriori Time: {apriori_time:.4f} seconds")
            print_results("Apriori", frequent_itemsets_apriori, rules_apriori)
    except Exception as e:
        print(f"An error occurred during Apriori algorithm execution: {e}")
        apriori_time = time.time() - start_time

    # FP-Growth Algorithm
    print("\nRunning FP-Growth Algorithm...")
    start_time = time.time()
    try:
        frequent_itemsets_fpgrowth = fpgrowth(transaction_df, min_support=min_support, use_colnames=True)
        fpgrowth_time = time.time() - start_time
        
        if frequent_itemsets_fpgrowth.empty:
            print("FP-Growth did not find any frequent itemsets.")
        else:
            rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=min_confidence)
            print(f"FP-Growth Time: {fpgrowth_time:.4f} seconds")
            print_results("FP-Growth", frequent_itemsets_fpgrowth, rules_fpgrowth)
    except Exception as e:
        print(f"An error occurred during FP-Growth algorithm execution: {e}")
        fpgrowth_time = time.time() - start_time
    
    return brute_time, apriori_time, fpgrowth_time


# <b> Read CSV and Prepare Transactions Function</b><br>
# This function reads transaction and itemset data from CSV files and prepares it for analysis:
# 1. It reads both the transaction and itemset CSV files.
# 2. It creates a mapping of item numbers to item names if available.
# 3. It processes each transaction, handling both comma-separated strings and individual items.
# 4. It returns a list of transactions, where each transaction is a list of items.

# <I>Note: Make sure the Datasets are kept in the same directory with the same naming conventions.<I>

# In[64]:


def read_csv_and_prepare_transactions(transaction_file, itemset_file):
    try:
        df_trans = pd.read_csv(transaction_file)
        df_items = pd.read_csv(itemset_file)
        
        if 'Item #' in df_items.columns and 'Item Name' in df_items.columns:
            item_map = dict(zip(df_items['Item #'], df_items['Item Name']))
        else:
            item_map = None
        
        transactions = []
        for _, row in df_trans.iterrows():
            transaction = []
            for item in row:
                if isinstance(item, str):
                    items = [i.strip() for i in item.split(',') if i.strip()]
                    transaction.extend(items)
                elif pd.notna(item):
                    transaction.append(str(item))
            if transaction:
                transactions.append(transaction)
        
        return transactions
    
    except Exception as e:
        print(f"Error reading the CSV files: {str(e)}")
        raise


# <b>Print Results Function</b><br>
# This function prints the results of the algorithms in a readable format:
# 1. It first prints the frequent itemsets with their support values.
# 2. Then it prints the association rules, showing the antecedent, consequent, confidence, and support for each rule.
# 3. It handles both DataFrame and list formats, accommodating the different output formats of the algorithms.

# In[66]:


def print_results(algorithm_name, frequent_itemsets, rules):
    print(f"\n{algorithm_name} Results:")
    print("Frequent Itemsets:")
    if isinstance(frequent_itemsets, pd.DataFrame):
        for _, row in frequent_itemsets.iterrows():
            print(f"Items: {set(row['itemsets'])}, Support: {row['support']*100:.2f}%")
    else:
        for itemset, support in frequent_itemsets:
            print(f"Items: {set(itemset)}, Support: {support*100:.2f}%")
    
    print("\nAssociation Rules:")
    if isinstance(rules, pd.DataFrame):
        for _, rule in rules.iterrows():
            print(f"Rule: {set(rule['antecedents'])} -> {set(rule['consequents'])}")
            print(f"Confidence: {rule['confidence']*100:.2f}%, Support: {rule['support']*100:.2f}%")
            print()
    else:
        for antecedent, consequent, confidence, support in rules:
            print(f"Rule: {set(antecedent)} -> {set(consequent)}")
            print(f"Confidence: {confidence*100:.2f}%, Support: {support*100:.2f}%")
            print()


# <b>Main Function</b><br>
# This is the main function that orchestrates the entire process:
# 1. It defines a dictionary of available stores and their corresponding CSV files.
# 2. It prompts the user to select a store for analysis.
# 3. It reads and prepares the transaction data for the selected store.
# 4. It prompts the user for minimum support and confidence thresholds.
# 5. It runs all three algorithms on the data and measures their execution times.
# 6. Finally, it prints the execution times and identifies the fastest algorithm.

# <b>Tutorial Note:</b> The General Store is small dataset so if you want to try low support values, try on that dataset otherwise the brute force takes time to execute but it works.

# In[69]:


def main():
    # Define available stores and their corresponding files
    stores = {
        "1": ("Amazon", "Amazon_Transaction.csv", "Amazon_Itemset.csv"),
        "2": ("Bestbuy", "Bestbuy_Transaction.csv", "Bestbuy_Itemset.csv"),
        "3": ("General", "General_Transaction.csv", "General_Itemset.csv"),
        "4": ("K-mart", "K_mart_Transaction.csv", "K_mart_Itemset.csv"),
        "5": ("Nike", "Nike_Transaction.csv", "Nike_Itemset.csv")
    }
    
    print("\nAvailable stores:")
    for key, (store_name, _, _) in stores.items():
        print(f"{key}. {store_name}")

    # Get user input for store selection.
    while True:
        choice = input("Enter the number of the store you want to analyze: ")
        if choice in stores:
            store_name, transaction_file, itemset_file = stores[choice]
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

    # Read and prepare transaction data.
    try:
        transactions = read_csv_and_prepare_transactions(transaction_file, itemset_file)
        if not transactions:
            print("No valid transactions found. Please check your CSV files.")
            return
    except Exception as e:
        print(f"Error reading the CSV files: {e}")
        print("Please ensure that the CSV files are properly formatted.")
        return

# Get user input for minimum support and minimum confidence.
    while True:
        try:
            min_support = float(input("Enter the minimum support (as a percentage between 0 and 100): "))
            if 0 <= min_support <= 100:
                min_support /= 100  # Convert to decimal
                break
            else:
                print("Please enter a value between 0 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            min_confidence = float(input("Enter the minimum confidence (as a percentage between 0 and 100): "))
            if 0 <= min_confidence <= 100:
                min_confidence /= 100  # Convert to decimal
                break
            else:
                print("Please enter a value between 0 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    brute_time, apriori_time, fpgrowth_time = run_all_algorithms(transactions, min_support, min_confidence)

    print(f"\nExecution Times:")
    print(f"Brute Force: {brute_time:.4f} seconds")
    print(f"Apriori: {apriori_time:.4f} seconds")
    print(f"FP-Growth: {fpgrowth_time:.4f} seconds")

    # Print the fastest algorithm.
    fastest_algorithm = min((brute_time, 'Brute Force'), (apriori_time, 'Apriori'), (fpgrowth_time, 'FP-Growth'))[1]
    print(f"\nThe fastest algorithm is: {fastest_algorithm}")

if __name__ == "__main__":
    main()


# In[ ]:




