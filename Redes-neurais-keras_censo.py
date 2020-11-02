
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


# In[7]:


# Adicionando as variáveis dummies para as variáveis categóricas
from sklearn.preprocessing import OneHotEncoder


# In[8]:


# onehotencoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
# previsores = onehotencoder.fit_transform(previsores).toarray()


# In[9]:


labelencoder_classe = LabelEncoder()


# In[10]:


classe = labelencoder_classe.fit_transform(classe)


# In[11]:


# Verificando números de linhas e colunas das colunas previsoras e da classe
print(previsores.shape)
print(classe.shape)


# In[12]:


# Escalonamento de vetores por padronização
from sklearn.preprocessing import StandardScaler


# In[13]:


scaler = StandardScaler()


# In[14]:


previsores = scaler.fit_transform(previsores)


# In[15]:


# Dividindo os dados em treino e teste
from sklearn.model_selection import train_test_split


# In[16]:


previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.15, random_state=0)


# In[17]:


# Importando pacote para Deep Learning
import tensorflow


# In[18]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[19]:


# Criando nossa rede neural inicial
classificador = Sequential()


# In[20]:


# Atribuindo primeira camada oculta para nossa rede, com 55 neurônios
# classificador.add(Dense(units=55, activation='relu', input_dim=108))

# Opção sem utilizar o onehotencoder
classificador.add(Dense(units=8, activation='relu', input_dim=14))


# In[21]:


# Atribuindo segunda camada oculta para nossa rede, com 55 neurônios
# classificador.add(Dense(units=55, activation='relu'))

# Opção sem utilizar o onehotencoder
classificador.add(Dense(units=8, activation='relu'))


# In[22]:


# Atribuindo camada de saída para nossa rede, com 1 neurônios (problema binário)
classificador.add(Dense(units=1, activation='sigmoid'))


# In[23]:


# Compilar rede neural
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[24]:


# Treinando o modelo, ajustando a cada 10 registros, ajustando o peso por 100 vezes
classificador.fit(previsores_train, classe_train, batch_size=10, epochs=100)


# In[25]:


previsoes = classificador.predict(previsores_test)


# In[26]:


previsoes = (previsoes > 0.5)


# In[27]:


# Calculando a precisão do modelo
from sklearn.metrics import confusion_matrix, accuracy_score


# In[28]:


precisao = accuracy_score(classe_test, previsoes)
precisao


# In[29]:


matriz = confusion_matrix(classe_test, previsoes)
matriz


# ## Resultados
# ### Redes neurais Tensorflow Keras
# 0.8133 - (labelencoder + onehotencoder + escalonamento)  
# 0.7559 - (labelencoder)  
# 0.2440 - (labelencoder + onehotencoder)  
# 0.8448 - (labelencoder + escalonamento)  
