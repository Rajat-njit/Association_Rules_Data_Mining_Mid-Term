import itertools
import time
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from tqdm import tqdm

def calculate_support(transactions, itemset):
    return sum(1 for transaction in transactions if set(itemset).issubset(set(transaction))) / len(transactions)

def brute_force_frequent_itemsets(transactions, min_support, progress_callback=None):
    items = sorted(set(item for transaction in transactions for item in transaction))
    frequent_itemsets = []
    itemset_size = 1
    total_combinations = sum(len(list(itertools.combinations(items, r))) for r in range(1, len(items) + 1))
    
    with tqdm(total=total_combinations, desc="Brute Force Progress", disable=progress_callback is None) as pbar:
        while True:
            candidate_itemsets = list(itertools.combinations(items, itemset_size))
            current_frequent_itemsets = []
            
            for itemset in candidate_itemsets:
                support = calculate_support(transactions, itemset)
                if support >= min_support:
                    current_frequent_itemsets.append((itemset, support))
                pbar.update(1)
                if progress_callback:
                    progress_callback(pbar.n / total_combinations)
            
            if not current_frequent_itemsets:
                break
            
            frequent_itemsets.extend(current_frequent_itemsets)
            itemset_size += 1

    return frequent_itemsets

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

def prepare_transaction_df(transactions):
    items = sorted(set(item for transaction in transactions for item in transaction))
    return pd.DataFrame([[item in transaction for item in items] for transaction in transactions], columns=items).astype(bool)

def run_algorithm(algorithm_name, algorithm_func, transactions, min_support, min_confidence, progress_callback=None):
    print(f"\nRunning {algorithm_name} Algorithm...")
    start_time = time.time()
    
    if algorithm_name == "Brute Force":
        frequent_itemsets = algorithm_func(transactions, min_support, progress_callback)
        rules = generate_association_rules(frequent_itemsets, transactions, min_confidence)
    else:
        transaction_df = prepare_transaction_df(transactions)
        frequent_itemsets = algorithm_func(transaction_df, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    algorithm_time = time.time() - start_time
    print(f"{algorithm_name} Time: {algorithm_time:.4f} seconds")
    print_results(algorithm_name, frequent_itemsets, rules)
    return algorithm_time

def print_results(algorithm_name, frequent_itemsets, rules):
    print(f"\n{algorithm_name} Results:")
    print("Frequent Itemsets:")
    if isinstance(frequent_itemsets, pd.DataFrame):
        for _, row in frequent_itemsets.iterrows():
            print(f"Items: {set(row['itemsets'])}, Support: {row['support']:.4f}")
    else:
        for itemset, support in frequent_itemsets:
            print(f"Items: {set(itemset)}, Support: {support:.4f}")
    
    print("\nAssociation Rules:")
    if isinstance(rules, pd.DataFrame):
        for _, rule in rules.iterrows():
            print(f"Rule: {set(rule['antecedents'])} -> {set(rule['consequents'])}")
            print(f"Confidence: {rule['confidence']:.4f}, Support: {rule['support']:.4f}")
            print()
    else:
        for antecedent, consequent, confidence, support in rules:
            print(f"Rule: {set(antecedent)} -> {set(consequent)}")
            print(f"Confidence: {confidence:.4f}, Support: {support:.4f}")
            print()

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

def main():
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

    while True:
        choice = input("Enter the number of the store you want to analyze: ")
        if choice in stores:
            store_name, transaction_file, itemset_file = stores[choice]
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

    try:
        transactions = read_csv_and_prepare_transactions(transaction_file, itemset_file)
        if not transactions:
            print("No valid transactions found. Please check your CSV files.")
            return
    except Exception as e:
        print(f"Error reading the CSV files: {e}")
        print("Please ensure that the CSV files are properly formatted.")
        return

    try:
        min_support = float(input("Enter the minimum support (as a percentage, e.g., 20 for 20%): ")) / 100
        min_confidence = float(input("Enter the minimum confidence (as a percentage, e.g., 50 for 50%): ")) / 100
    except ValueError:
        print("Invalid input for minimum support or confidence. Please run the script again and enter valid numbers.")
        return

    algorithms = {
        "1": ("Brute Force", brute_force_frequent_itemsets),
        "2": ("Apriori", apriori),
        "3": ("FP-Growth", fpgrowth)
    }

    print("\nAvailable algorithms:")
    for key, (algo_name, _) in algorithms.items():
        print(f"{key}. {algo_name}")

    chosen_algos = input("Enter the numbers of the algorithms you want to run (comma-separated): ").split(",")

    execution_times = {}

    for choice in chosen_algos:
        if choice in algorithms:
            algo_name, algo_func = algorithms[choice]
            progress_callback = None if algo_name != "Brute Force" else lambda progress: print(f"\rProgress: {progress:.2%}", end="", flush=True)
            execution_times[algo_name] = run_algorithm(algo_name, algo_func, transactions, min_support, min_confidence, progress_callback)
        else:
            print(f"Invalid algorithm choice: {choice}")

    print(f"\nExecution Times:")
    for algo_name, exec_time in execution_times.items():
        print(f"{algo_name}: {exec_time:.4f} seconds")

    if execution_times:
        fastest_algorithm = min(execution_times, key=execution_times.get)
        print(f"\nThe fastest algorithm is: {fastest_algorithm}")

if __name__ == "__main__":
    main()
