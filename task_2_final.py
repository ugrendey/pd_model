from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import riskpy
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter('ignore')
from riskpy.modeling.binning import Binner
from riskpy.graphs.graphs import binning_barchart
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve



#%%
class data_combine:

  def __init__(self, train_df):
      self.train_df = train_df

  def key_faktor(self):
      self['churn'] = self['churn'].replace(False, 0)
      self['churn'] = self['churn'].replace(True, 1)
      self.drop(['phone number'], axis=1, inplace=True)
      return self
  
  def col_types(self):
      categorical_columns = [c for c in self.columns if self[c].dtype.name == 'object']
      numerical_columns   = [c for c in self.columns if self[c].dtype.name != 'object']
      return categorical_columns, numerical_columns
      
#%%
class categoric:
  def __init__(self, binary_columns, nonbinary_columns):
      self.binary_columns = binary_columns
      self.nonbinary_columns = nonbinary_columns
  
  def vect(self):      
      categorical_columns = data_combine.col_types(self)[0]
      data_describe = self.describe(include=[object])
      binary_columns    = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
      nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
      return binary_columns, nonbinary_columns
      
  def bina(self):   #руками посмотрел на бинарные признаки, здесь просто преобразовал их
      binary_columns = categoric.vect(self)[0]
      self.loc[self['international plan'] == 'no', 'international plan'] = 0
      self.loc[self['international plan'] == 'yes', 'international plan'] = 1    
      self.loc[self['voice mail plan'] == 'no', 'voice mail plan'] = 0
      self.loc[self['voice mail plan'] == 'yes', 'voice mail plan'] = 1   
      data_binary = self[binary_columns]
      return data_binary

  def nonbina(self):
      nonbinary_columns = categoric.vect(self)[1]
      data_nonbinary = pd.get_dummies(train_df[nonbinary_columns])
      return data_nonbinary

#%%
class numeric: 
  def __init__(self, train_df):
      self.train_df = train_df 
  
  def feat_custom(self):
    self['f1']=self['total eve minutes'] / self['total intl charge']
    self['f2']=self['total eve minutes'] / self['customer service calls']
    self['f3']=self['total eve calls'] / self['total eve charge']
    self['f4']=self['total eve calls'] / self['total night minutes']
    self['f5']=self['total eve calls'] / self['total night calls']
    self['f6']=self['total eve calls'] / self['total night charge']
    self['f7']=self['total eve calls'] / self['total intl minutes']
    self['f8']=self['total eve calls'] / self['total intl calls']
    self['f9']=self['total eve calls'] / self['total intl charge']
    self['f10']=self['total eve calls'] / self['customer service calls']
    self['f11']=self['total eve charge'] / self['total night minutes']
    self['f12']=self['total eve charge'] / self['total night calls']
    self['f13']=self['total eve charge'] / self['total night charge']
    self['f14']=self['total eve charge'] / self['total intl minutes']
    self['f15']=self['total eve charge'] / self['total intl calls']
    self['f16']=self['total eve charge'] / self['total intl charge']
    self['f17']=self['total eve charge'] / self['customer service calls']
    self['f18']=self['total night minutes'] / self['total night calls']
    self['f19']=self['total night minutes'] / self['total night charge']
    self['f20']=self['total night minutes'] / self['total intl minutes']
    self['f21']=self['total night minutes'] / self['total intl calls']
    self['f22']=self['total night minutes'] / self['total intl charge']
    self['f23']=self['total night minutes'] / self['customer service calls']
    self['f24']=self['total night calls'] / self['total night charge']
    self['f25']=self['total night calls'] / self['total intl minutes']
    self['f26']=self['total night calls'] / self['total intl calls']
    self['f27']=self['total night calls'] / self['total intl charge']
    self['f28']=self['total night calls'] / self['customer service calls']
    self['f29']=self['total night charge'] / self['total intl minutes']
    self['f30']=self['total night charge'] / self['total intl calls']
    self['f31']=self['total night charge'] / self['total intl charge']
    self['f32']=self['total night charge'] / self['customer service calls']
    self['f33']=self['total intl minutes'] / self['total intl calls']
    self['f34']=self['total intl minutes'] / self['total intl charge']
    self['f35']=self['total intl minutes'] / self['customer service calls']
    self['f36']=self['total intl calls'] / self['total intl charge']
    self['f37']=self['total intl calls'] / self['customer service calls']
    self['f38']=self['total intl charge'] / self['customer service calls']
    return self

  def finalprep(self):
    numerical_columns = data_combine.col_types(self)[1]
    categorical_columns = data_combine.col_types(self)[0]
    
    self = numeric.feat_custom(self)
    self.fillna(0, inplace = True)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    newdf = self.select_dtypes(include=numerics)
    b = newdf.columns.to_series()[np.isinf(newdf).any()]
    
    df = self[b]
    for i in b:
        max_value = np.nanmax(df[i][df[i] != np.inf ])*100
        self[i].replace(np.inf, max_value, inplace= True )
    
    
    numerical_new = [c for c in self.columns if c not in (categorical_columns + numerical_columns)]
    numerical_columns = numerical_columns + numerical_new
    
    data_numerical = train_df[numerical_columns]
    data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()
    return data_numerical
      
#%%
class woe_trans:
  def __init__(self, data):
      self.data = data
      
  def trans(self, data_nonbinary, data_binary, train_df):
    column_list = list(self.columns)
    subset = column_list.copy()
    self.drop_duplicates(subset=subset, inplace = True)
    
    X_ip = self.drop(['churn'], axis = 1)
    y_ip = train_df['churn']
     
    X_train_ip, X_test_ip, y_train_ip, y_test_ip = train_test_split(X_ip, y_ip, test_size=0.3, random_state=42)
        
    X_train_ip['churn'] = y_train_ip
    X_test_ip['churn'] = y_test_ip
    X_ip['churn'] = y_ip
    
    binner_ip = Binner()
    bins = binner_ip.fit(X_train_ip, 'churn')
    X_train_woe_ip = binner_ip.transform(X_train_ip)
    X_test_woe_ip = binner_ip.transform(X_test_ip)
    
    
    res_col = list(data_nonbinary.columns) + list(data_binary.columns)
    X_train_woe_ip_rest = X_train_ip[res_col]
    X_test_woe_ip_rest = X_test_ip[res_col]
    X_train_woe_ip = pd.concat((X_train_woe_ip, X_train_woe_ip_rest), axis=1)
    X_test_woe_ip = pd.concat((X_test_woe_ip, X_test_woe_ip_rest), axis=1)
    
    y_train_woe_ip = X_train_woe_ip['churn']
    y_test_woe_ip = X_test_woe_ip['churn']
    X_train_woe_ip.drop(['churn'], axis = 1, inplace = True)
    X_test_woe_ip.drop(['churn'], axis = 1, inplace = True)
    return X_train_woe_ip, X_test_woe_ip, y_train_woe_ip, y_test_woe_ip

  def otbor_param(FEATURE_COUNT, X_train_woe_ip, y_train_woe_ip):
    logreg = LogisticRegression(random_state = 42, solver = 'liblinear', penalty = 'l1')
    skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42)
    
    selector = SFS(logreg, 
               k_features = FEATURE_COUNT , 
               forward = True,  
               scoring ='roc_auc',
               cv = skf)
    
    selector = selector.fit(X_train_woe_ip, y_train_woe_ip)
      
    top_feat = list(selector.subsets_[FEATURE_COUNT]['feature_names'])
    gini_features = pd.DataFrame(index=top_feat, columns = ['Gini'])
    for feat in top_feat:
        logreg = LogisticRegression(random_state = 42, penalty = 'l2', C = 1.5)
        logreg.fit(X_train_woe_ip.loc[:, feat].values.reshape(-1,1), y_train_woe_ip)
        preds_train = logreg.predict_proba(X_train_woe_ip.loc[:, feat].values.reshape(-1,1))[:, 1]
        gini_features.loc[feat, 'Gini'] = 2*roc_auc_score(y_train_woe_ip, preds_train) - 1
        
    good_gini = gini_features.loc[gini_features['Gini'] > 0.05]
    gg = good_gini.index.tolist()
    X_train_woe_ip_final = X_train_woe_ip[gg]      
    
    
    corrtab = X_train_woe_ip_final[gg].corr()
    #тут я ручками поудалял корреляционные (не стал кодом заморачиваться)
    X_train_woe_ip_final.drop(['customer service calls_woe', 'f23_woe', 'f10_woe', 'f2_woe'], axis=1, inplace=True)
    corrtab = X_train_woe_ip_final[X_train_woe_ip_final.columns].corr()
    
    X_train_woe_ip_final = X_train_woe_ip[X_train_woe_ip_final.columns]
    X_test_woe_ip_final = X_test_woe_ip[X_train_woe_ip_final.columns]
    return X_train_woe_ip_final, X_test_woe_ip_final

  def regularization(RANDOM_STATE, X_train_woe_ip_final, y_train_woe_ip):
    cv_scores_train = []
    cv_scores_test = []
    alphas = np.arange(0.1, 4, 0.2)
    kfold = StratifiedKFold(n_splits = 3, shuffle = True, random_state = RANDOM_STATE)
    for alpha in alphas:
        scores_one_alpha_train = []
        scores_one_alpha_test = []
        coeff = []
        for train_index, test_index in kfold.split(X_train_woe_ip_final, y_train_woe_ip):
     
            X_train, X_test = X_train_woe_ip_final.iloc[train_index,:], X_train_woe_ip_final.iloc[test_index,:]
            y_train, y_test = y_train_woe_ip.iloc[train_index], y_train_woe_ip.iloc[test_index]
            logreg = LogisticRegression(random_state = RANDOM_STATE, penalty = 'l2', C = alpha)
            logreg.fit(X_train, y_train)
            pred_train = logreg.predict_proba(X_train)[:, 1]
            pred_test = logreg.predict_proba(X_test)[:, 1]
            scores_one_alpha_train.append(roc_auc_score(y_train, pred_train))
            scores_one_alpha_test.append(roc_auc_score(y_test, pred_test))
            coeff.append(logreg.coef_)
        mean_coeff = np.mean(coeff, axis = 0)
        cv_scores_train.append(np.mean(scores_one_alpha_train))
        cv_scores_test.append(np.mean(scores_one_alpha_test))
    
    return alphas[np.argmax(cv_scores_test)]
    
  def model(reg_coef, RANDOM_STATE,X_train_woe_ip_final, y_train_woe_ip,X_test_woe_ip_final, y_test_woe_ip):
    logreg_ip = LogisticRegression(random_state = RANDOM_STATE, penalty = 'l2', C = reg_coef)
    logreg_ip.fit(X_train_woe_ip_final, y_train_woe_ip)
    y_pred = logreg_ip.predict_proba(X_test_woe_ip_final)[:,1]
    y_pred_train = logreg_ip.predict_proba(X_train_woe_ip_final)[:,1]
    print("ROC AUC val:", roc_auc_score(y_test_woe_ip, y_pred))
    print("ROC AUC train:", roc_auc_score(y_train_woe_ip, y_pred_train))
    
    print(logreg_ip.coef_)
    

    logit_roc_auc = roc_auc_score(y_test_woe_ip, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test_woe_ip, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

    gini_=2*logit_roc_auc-1
    print(gini_)

#%%
if __name__ == "__main__":
  train_df = pd.read_csv(r'D:\2019-03-16\МСП Банк\Тестовое задание\telecom_churn.csv', sep=',') 
  train_df = data_combine.key_faktor(train_df)
  data_binary = categoric.bina(train_df)
  data_nonbinary = categoric.nonbina(train_df)
  data_numerical = numeric.finalprep(train_df)
  
  data = pd.concat((data_numerical, data_nonbinary, data_binary), axis=1)
  data = pd.DataFrame(data, dtype=float)
  
  woe_transform = woe_trans.trans(data, data_nonbinary, data_binary, train_df)
  X_train_woe_ip = woe_transform[0]
  X_test_woe_ip = woe_transform[1]
  y_train_woe_ip = woe_transform[2]
  y_test_woe_ip = woe_transform[3]
  
  X_final = woe_trans.otbor_param(20, X_train_woe_ip, y_train_woe_ip)
  X_train_woe_ip_final = X_final[0]
  X_test_woe_ip_final = X_final[1]
  
  reg_coef = woe_trans.regularization(42, X_train_woe_ip_final, y_train_woe_ip)
  woe_trans.model(reg_coef, 42, X_train_woe_ip_final, y_train_woe_ip, X_test_woe_ip_final, y_test_woe_ip)
  
  