# -*-coding:utf-8-*-
# Ref: https://pypi.org/project/efficient-apriori/

from efficient_apriori import apriori


def loadDataSet():
    """
    Load dataSet
    Returns:
        A dataSet: A list of transactions. Each transaction contains several items.
    """
    dataSet = [('eggs', 'bacon', 'soup'), ('eggs', 'bacon', 'apple'), ('soup', 'bacon', 'banana')]
    return dataSet


def main():
    dataSet = loadDataSet()
    itemsets, rules = apriori(dataSet, min_support=0.5, min_confidence=1)
    print(rules)


if __name__ == '__main__':
    main()


