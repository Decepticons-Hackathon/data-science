#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def match_processing(parser_data, product_data, n_matchs):

  get_ipython().system('pip install fuzzywuzzy -q')
  get_ipython().system('pip install python-Levenshtein -q')

  import pandas as pd
  from fuzzywuzzy import fuzz
  from fuzzywuzzy import process
  import numpy as np
  import torch
  from transformers import AutoTokenizer, AutoModel, logging
  import re
  import pickle

  from sklearn.model_selection import train_test_split, GridSearchCV
  from sklearn.metrics import precision_score
  from lightgbm import LGBMClassifier


  import warnings
  warnings.filterwarnings('ignore')
  logging.set_verbosity_error()

  RANDOM_STATE=12345


#--------------------------------------------------Загружаем предобученную модель--------------------------------------------------------------------

  with open('pickle_model.pkl', "rb") as file:
    unpickler = pickle.Unpickler(file)
    pretrained_model = unpickler.load()

 #--------------------------------------------------загружаем данные из csv--------------------------------------------------------------------

  parser_data = pd.read_csv('/content/marketing_dealerprice.csv', sep=';', index_col='id')
  product_data = pd.read_csv('/content/marketing_product.csv', sep=';')

#--------------------------------------------------Функция предобработки текста.--------------------------------------------------------------------
  def re_name(name):
    """
    Функция предобработки текста.
    Разделяет некоторые слитные слова.
    мылоPROSEPT -> мыло PROSEPT
    """

    return re.sub(r'(?<=[а-яa-z])(?=[A-Z])|(?=[а-я])(?<=[A-Za-z])|(?<=[a-z])(?=[0-9])', ' ', str(name))

  parser_data['product_name'] = parser_data['product_name'].apply(re_name)
  product_data['name'] = product_data['name'].apply(re_name)
#----------------------------------------------------------------------------------------------------------------------
#-----------------------------------ВЕКТОРИЗАЦИЯ ПРИЗНАКОВ-------------------------------------------------------------
  tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
  model = AutoModel.from_pretrained("cointegrated/rubert-tiny")

#--------------------------------- Эмбеддинг для признака name---------------------------------------------------------
  tokenized = product_data['name'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
  max_len = 0
  for i in tokenized.values:
    if len(i) > max_len:
      max_len = len(i)

  padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
  attention_mask = np.where(padded != 0, 1, 0)
  input_ids = torch.tensor(padded)
  attention_mask = torch.tensor(attention_mask)
  with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)
  vector_name = last_hidden_states[0][:,0,:].numpy()

#--------------------------------- Эмбеддинг для признака product_name---------------------------------------------------------

  tokenized = parser_data['product_name'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
  max_len = 0
  for i in tokenized.values:
    if len(i) > max_len:
      max_len = len(i)

  padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
  attention_mask = np.where(padded != 0, 1, 0)
  input_ids = torch.tensor(padded)
  attention_mask = torch.tensor(attention_mask)

  with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)
  vector_product_name = last_hidden_states[0][:,0,:].numpy()

#--------------------------------- Скалярное произведение---------------------------------------------------------

  scalar = list()
  for i in range(vector_name.shape[0]):
    scalar.append(np.dot(vector_product_name[i], vector_name[i]))
  product_data['scalar_feach'] = pd.Series(scalar)

#--------------------------------- Объединение датасетов---------------------------------------------------------

  product_data = product_data.fillna(0)

# Удаляем полные дубликаты
  product_data = product_data.drop_duplicates().reset_index(drop=True)
  parser_data = parser_data.drop_duplicates().reset_index(drop=True)

# Добавляем кололнку для объединения и заполняем их одинаковым значением(для корректной склейки)
  product_data['merge'] = 1
  parser_data['merge'] = 1

# Объединяем таблицы так, чтобы к каждому product_key все товары производителя.
  union_data = pd.merge_ordered(parser_data,product_data, fill_method = 'ffill',
                           right_by='id').drop('merge',axis=1).reset_index(drop=True)
# добавляем расстояние Левенштейна как признак.
  def fuzzywuzzy_name(dataframe):
    return fuzz.token_set_ratio(dataframe['product_name'], dataframe['name'])
  union_data['fuzz'] = union_data.apply(fuzzywuzzy_name, axis=1)

# В таблице ids будут хранится айдишники для сопоставления с предиктивом расстояние левенштейна для ранжирования
  ids=pd.DataFrame()
  ids[['product_dealer_id','dealer_id','manufacturer']] =  union_data[['product_key','dealer_id','id']]

# Удаляем лишние айдишники  из датасета с признаками(те что остались нужны, т.к. модель их использует)
  union_data = union_data[['scalar_feach','dealer_id','id','category_id','fuzz']]

#--------------------------------- Обработка данных предобученной моделью---------------------------------------------------------

  predictions = pd.DataFrame(pretrained_model.predict(union_data))
  predictions = predictions.rename(columns={0:'is_matching'})
  predictions_threats = pd.DataFrame(pretrained_model.predict_proba(union_data))
  predictions_threats = predictions_threats.rename(columns={0:'match_threats'})

#--------------------------------- Обработка полученных предсказаний ---------------------------------------------------------

  matchs = pd.concat((ids,predictions,predictions_threats['match_threats']),axis=1)
  matchs = matchs[matchs['is_matching'] == 1]
  final_table = matchs.sort_values(by=['product_dealer_id','dealer_id','match_threats'],
                                 ascending=True).drop('is_matching',axis=1)

#--------------------------------- обрезка до нужного количества предсказаний. ---------------------------------------------------------

  def to_nmatchs(data):
    new_data = pd.DataFrame()
    temp_data = data.set_index(['product_dealer_id','dealer_id'],drop=True)
    for i in temp_data.index.unique():
      new_data = pd.concat((new_data,temp_data.loc[i,:].head(n_matchs)))
    return new_data

  finish = to_nmatchs(final_table)
#--------------------------------- Убираем не нужные колонки (оставляем только сматченные айдишники) ---------------------------------------------------------
  finish = finish.drop(['fuzz','match_threats'],axis=1)

#--------------------------------- Убираем не нужные колонки (оставляем только сматченные айдишники) ---------------------------------------------------------
  finish = finish.to_csv('matched_ids.csv')
  finish = csv.reader('marketing_product.csv',delimiter = ";")

  return finish

