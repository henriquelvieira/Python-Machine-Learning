from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd

df = pd.read_csv('https://github.com/henriquelvieira/Python-Pandas/raw/main/data/wine_dataset.csv')
print(df.shape)

#TRANSFORMAR OS VALORES DA COLUNA style P/ NÚMEROS (RED - 0 / WHITE - 1) 
df['style'] = (df['style'].replace('red', 0)).replace('white', 1)

y = df['style'] #DF COM AS RESPOSTAS (COLUNA STYLE)
x = df.drop('style', axis=1) #DF SEM A RESPOSTA (COLUNA STYLE)

#CRIAÇÃO DOS CONJUNTOS DE DADOS P/ TREINO (70%) E TESTE (30%)
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)

modelo = ExtraTreesClassifier(n_estimators=100) #CRIAÇÃO DO MODELO
modelo.fit(x_treino, y_treino) #TREINO DO MODELO

resultado = modelo.score(x_teste, y_teste) #CALCULAR A ACURÂCIA DO MODELO
print(f'Taxa de acurâcia: {resultado}')

print(y_teste[400:403])
print(x_teste[400:403])

previsoes = modelo.predict(x_teste[400:403]) #CHAMADA DO MODELO P/ CLASSIFICAR OS REGISTROS DOS DADOS DE TESTE
# print(previsoes)

for index, resposta in enumerate(previsoes):
  if resposta == 0:
      resultado = 'Red'
  else:
      resultado = 'White'

  print(f'{str(index)} - {resultado}')