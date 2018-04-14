import digit_config
from digit_helpers import *

########################################
## Loading data and some EDA
########################################
# data = pd.read_csv("raw_data/train.csv")
# data = data.sample(10000)
# file_name = 'sampled_digit.csv'
# data.to_csv(file_name, sep='\t')

data = pd.read_csv("sampled_digit.csv", sep='\t')

print('>>>>> raw dataset info:')
print("shape of raw dataset:", data.shape)
print(data.columns.values)

print(">>>>> missing data check in raw data:")
print(data.isnull().sum())

f1,ax1=plt.subplots(1,2,figsize=(18,8))
data['label'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax1[0],shadow=True)
ax1[0].set_title('Observation percentage of different classes')
ax1[0].set_ylabel('')
sns.countplot('label',data=data,ax=ax1[1])
ax1[1].set_title('Observation counts of different classes')


labels = data.pop('label')
X_train, X_test, y_train, y_test= train_test_split(data.iloc[:,1:], labels, test_size=0.3, random_state=0)

# plt.figure()
# sns.heatmap(X_train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
# fig=plt.gcf()
# fig.set_size_inches(10,8)

# pca=PCA(n_components=30)
# X_train_pca=pca.fit_transform(X_train)
# X_test_pca=pca.transform(X_test)
# var_ratio = sum(pca.explained_variance_ratio_)
# print("Explained variance: {:.3%}".format(var_ratio))
# print(X_train_pca)
# plt.figure()
# sns.heatmap(X_train_pca.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
# fig=plt.gcf()
# fig.set_size_inches(10,8)
# plt.show()