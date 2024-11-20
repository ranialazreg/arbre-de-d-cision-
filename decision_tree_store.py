import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


data_store = pd.read_csv('store_sales.csv')


X = data_store[['Category', 'Price', 'Discount']]
X['Category'] = X['Category'].astype('category').cat.codes 


y = data_store['Sales']


clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X, y)

plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=['Not Sold', 'Sold'],
    filled=True
)
plt.title("Arbre de décision - Ventes de magasin")
plt.show()
