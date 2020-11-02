
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Importando base de dados
base = pd.read_csv('census.csv')


# In[3]:


# Dividir entre previsores e classe
previsores = base.iloc[:, 0:14].values

classe = base.iloc[:, 14].values


# In[4]:


# Transformando atributos nominais em atributos discretos
from sklearn.preprocessing import LabelEncoder


# In[5]:


labelencoder_previsores = LabelEncoder()


# In[6]:


previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])


# In[ ]:


# Adicionando as variáveis dummies para as variáveis categóricas
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


onehotencoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotencoder.fit_transform(previsores).toarray()


# In[ ]:


# labelencoder_classe = LabelEncoder()


# In[ ]:


# classe = labelencoder_classe.fit_transform(classe)


# In[7]:


# Verificando números de linhas e colunas das colunas previsoras e da classe
print(previsores.shape)
print(classe.shape)


# In[8]:


# Escalonamento de vetores por padronização
from sklearn.preprocessing import StandardScaler


# In[9]:


scaler = StandardScaler()


# In[10]:


previsores = scaler.fit_transform(previsores)


# In[11]:


# Dividindo os dados em treino e teste
from sklearn.model_selection import train_test_split


# In[12]:


previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.15, random_state=0)


# In[13]:


# Criando modelo redes neurais
from sklearn.neural_network import MLPClassifier


# In[14]:


classificador = MLPClassifier(verbose=True, max_iter=1000, tol=0.00001)


# In[15]:


classificador.fit(previsores_train, classe_train)


# In[16]:


previsoes = classificador.predict(previsores_test)


# In[17]:


# Calculando a precisão do modelo
from sklearn.metrics import confusion_matrix, accuracy_score


# In[18]:


precisao = accuracy_score(classe_test, previsoes)
precisao


# In[19]:


matriz = confusion_matrix(classe_test, previsoes)
matriz


# ## Resultados
# ### Redes neurais (MLP Classifier)
# 0.8360 - (labelencoder + onehotencoder + escalonamento) - max_iter=1000, tol=0.00001  
# 0.2440 - (labelencoder) - max_iter=1000, tol=0.00001  
# 0.7881 - (labelencoder + onehotencoder) - max_iter=1000, tol=0.00001  
# 0.8473 - (labelencoder + escalonamento) - max_iter=1000, tol=0.00001 
