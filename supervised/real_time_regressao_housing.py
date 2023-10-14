import joblib
modelo = joblib.load('modelo_regressao.pkl')
taxa_criminalidade = 3
media_quartos = 1.0
media_sala_estar = 5.0
populacao = 700
media_idade_casas = 40.0
renda_media_bairro = 2.0

# Organização das características em um array bidimensional (um exemplo de entrada)
entrada_teste = [[taxa_criminalidade, media_idade_casas, media_sala_estar, media_quartos, populacao, renda_media_bairro, 44, -122]]
print("Preço Estimado: ", modelo.predict(entrada_teste))