# -*-coding:utf-8-*-
# Ref: https://www.cnblogs.com/llhthinker/p/6719779.html


def loadDataSet():
    """
    Load dataSet
    Returns:
        A dataSet: A list of transactions. Each transaction contains several items.
    """
    dataSet = [
        ['11', '12', '15'], ['12', '14'], ['12', '13'],
        ['11', '12', '14'], ['11', '13'], ['12', '13'],
        ['11', '13'], ['11', '12', '13', '15'], ['11', '12', '13']]
    return dataSet


def is_apriori(Ck_item, Lksub1):
    """
    Judge whether a frequent candidate k-items' set satisfy Apriori property.
    Args:
        Ck_item: a frequent candidate k-items' set in Ck which contains all frequent
                candidate k-items' sets.
        Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-items' set.
    Returns:
        True or False.
    """
    for item in Ck_item:
        sub_Ck = Ck_item - frozenset([item])
        if sub_Ck not in Lksub1:
            return False
    return True


def create_C1(dataSet):
    """
    Create frequent candidate 1-itemset C1
    Args:
        A dataSet: A list of transactions. Each transaction contains several items.
    Returns:
        C1: A set which contains all frequent candidate 1-itemsets
    """
    C1 = set()
    for transaction in dataSet:
        for item in transaction:
            C1.add(frozenset([item]))
    return C1


def create_Ck(Lksub1, k):
    """
    Create Ck, a set which contains all frequent candidate k-items' sets by Lk-1's own connection operation.
    Args:
        Lksub1: Lk-1, a set which contains all frequent candidate (k-1)-items' sets
        k: The item's number of a frequent itemset.
    Returns:
        Ck: a set which contains all frequent candidate k-items' set.
    """
    Ck = set()
    len_Lksub1 = len(Lksub1)
    list_Lksub1 = list(Lksub1)
    for i in range(len_Lksub1-1):
        for j in range(i+1, len_Lksub1):
            a, b = list(list_Lksub1[i]), list(list_Lksub1[j])
            a.sort()
            b.sort()
            if a[0:k-2] == b[0:k-2]:
                Ck_item = list_Lksub1[i] | list_Lksub1[j]
                # pruning
                if is_apriori(Ck_item, Lksub1):
                    Ck.add(Ck_item)

    return Ck


def generate_Lk_by_Ck(dataSet, Ck, min_support, support_data):
    """
    Generate Lk by excuting a delete policy from Ck.
    Args:
        dataSet: A dataSet: A list of transactions. Each transaction contains several items.
        Ck: A set which contains all frequent candidate k-items' sets
        min_support: The minimum support.
        support_data: A dictionary. The key is frequent itemset and value is support.
    Returns:
        Lk: A set which contains all frequent k-items' sets
    """
    Lk = set()
    item_count = {}
    for transaction in dataSet:
        for item in Ck:
            if item.issubset(transaction):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    t_num = len(dataSet)
    for item in item_count:
        if item_count[item]/t_num >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item]/t_num

    return Lk


def generate_L(dataSet, k, min_support):
    """
    Generate all frequent itemsets.
    Args:
        dataSet: A dataSet: A list of transactions. Each transaction contains several items.
        k: Maximum number of items for all frequent itemsets.
        min_support: The minimum support.
    Returns:
        L: The list of Lk.
        support_data: A dictionary. The key is frequent itemset and value is support.
    """
    support_data = {}
    C1 = create_C1(dataSet)
    L1 = generate_Lk_by_Ck(dataSet, C1, min_support, support_data)
    Lksub1 = L1.copy()
    L = []
    L.append(Lksub1)
    for i in range(2, k+1):
        Ci = create_Ck(Lksub1, i)
        Li = generate_Lk_by_Ck(dataSet, Ci, min_support, support_data)
        Lksub1 = Li.copy()
        L.append(Lksub1)

    return L, support_data


def generate_big_rules(L, support_data, min_conf):
    """
    Generate big rules from frequent itemsets.
    Args:
        L: The list of Lk.
        support_data: A dictionary. The key is frequent itemset and value is support.
        min_conf: Minimal confidence/
    Returns：
        big_rules_list: A list which contains all big rules. Each big rule is represented
        as a 3-tuple.
    """
    big_rule_list = []
    sub_set_list = []
    for i in range(len(L)):
        for freq_set in L[i]:
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[freq_set-sub_set]
                    big_rule = (freq_set - sub_set, sub_set, conf)
                    if conf >= min_conf and big_rule not in big_rule_list:
                        big_rule_list.append(big_rule)
            sub_set_list.append(freq_set)

    return big_rule_list


def main():
    dataSet = loadDataSet()
    L, support_data = generate_L(dataSet, 3, 0.2)
    big_rules_list = generate_big_rules(L, support_data, 0.7)
    for Lk in L:
        print('*' * 50)
        print('frequent' + str(len(list(Lk)[0])) + '-itemsets\t\tsupport')
        print('*' * 50)
        for freq_set in Lk:
            print(freq_set, support_data[freq_set])
    print('Big Rules')
    for item in big_rules_list:
        print(item[0], '=>', item[1], 'conf:', item[2])


if __name__ == '__main__':
    main()

