import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

class PrevisaoProdutoRequest(BaseModel):
    peso: int
    tipo_embalagem: Literal["Caixa de Papelão", "Plástico Bolha"]

class PrevisaoProdutoResponse(BaseModel):
    produto: Literal["Smartphone", "Tablet"]

app = FastAPI()

modelo_dsa = joblib.load('modelos/modelo_logistica.pkl')
le_tipo_embalagem = joblib.load('modelos/transformador_tipo_embalagem.pkl')
le_tipo_produto = joblib.load('modelos/transformador_tipo_produto.pkl')


@app.post("/prever/", response_model=PrevisaoProdutoResponse)
def prever(produto: PrevisaoProdutoRequest):

    dados_entrada = [[produto.peso, le_tipo_embalagem.transform([produto.tipo_embalagem])[0]]]    
    previsao = modelo_dsa.predict(dados_entrada)[0]
    print(previsao)
    tipo_produto = le_tipo_produto.inverse_transform([previsao])[0]
    print(tipo_produto)
    return {
        'produto': tipo_produto
    }

