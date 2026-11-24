from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class AnalisadorDados:
    def __init__(self, pasta_saida: str | Path) -> None:
        self.pasta_saida = str(pasta_saida)
        Path(self.pasta_saida).mkdir(parents=True, exist_ok=True)

    def carregar_csv(self, caminho_arquivo: str | Path) -> pd.DataFrame:
        dataframe = pd.read_csv(caminho_arquivo)
        print(
            f"Arquivo '{caminho_arquivo}' carregado "
            f"({dataframe.shape[0]} linhas)."
        )
        return dataframe

    def remover_outliers_iqr(
        self,
        df: pd.DataFrame,
        coluna: str,
        minimo_valido: float = 0,
    ) -> pd.Series:
        serie = df[coluna].copy()
        valores_validos = serie[serie > minimo_valido]
        primeiro_quartil = valores_validos.quantile(0.25)
        terceiro_quartil = valores_validos.quantile(0.75)
        intervalo_interquartil = terceiro_quartil - primeiro_quartil
        limite_superior = terceiro_quartil + 1.5 * intervalo_interquartil
        condicao_invalida = (serie <= minimo_valido) | (serie > limite_superior)
        quantidade_invalidos = int(condicao_invalida.sum())
        serie.loc[condicao_invalida] = np.nan
        print(
            f"[{coluna}] Valores inválidos/outliers convertidos em NaN: "
            f"{quantidade_invalidos}"
        )
        return serie

    def exportar_csv(self, dataframe: pd.DataFrame, nome_arquivo: str) -> None:
        caminho_completo = Path(self.pasta_saida) / nome_arquivo
        caminho_completo.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(caminho_completo, index=False, encoding="utf-8-sig")
        print(f"Relatório CSV salvo em: {caminho_completo}")


class AnalisadorSupermercado(AnalisadorDados):
    def __init__(
        self,
        caminho_produtos: str | Path,
        caminho_vendas: str | Path,
        pasta_saida: str | Path,
    ) -> None:
        super().__init__(pasta_saida)
        self.caminho_produtos = str(caminho_produtos)
        self.caminho_vendas = str(caminho_vendas)
        self.df_completo: pd.DataFrame | None = None
        self._auditoria: list[dict[str, Any]] = []

    def limpar_produtos(self, df_produtos: pd.DataFrame) -> pd.DataFrame:
        df = df_produtos.copy()
        preco_original = df["preco_unitario"].copy()
        df["preco_unitario"] = self.remover_outliers_iqr(
            df, "preco_unitario", minimo_valido=0
        )
        medianas_categoria = df.groupby("categoria")["preco_unitario"].transform(
            "median"
        )
        df["preco_unitario"] = (
            df["preco_unitario"].fillna(medianas_categoria).round(2)
        )

        if df["preco_unitario"].isna().any():
            categorias_sem_preco = df.loc[
                df["preco_unitario"].isna(), "categoria"
            ].unique()
            raise ValueError(
                "Não foi possível imputar preços para as categorias: "
                f"{categorias_sem_preco.tolist()}"
            )

        indices_corrigidos = preco_original.isna() | (
            preco_original != df["preco_unitario"]
        )
        for indice in df.index[indices_corrigidos]:
            self._auditoria.append(
                {
                    "venda_id": None,
                    "product_id": df.at[indice, "product_id"],
                    "coluna_corrigida": "preco_unitario",
                    "valor_original": preco_original.at[indice],
                    "valor_corrigido": df.at[indice, "preco_unitario"],
                }
            )
        return df

    def limpar_vendas(self, df_vendas: pd.DataFrame) -> pd.DataFrame:
        df = df_vendas.copy()
        quantidade_original = df["quantidade"].copy()
        df["quantidade"] = self.remover_outliers_iqr(
            df, "quantidade", minimo_valido=0
        )
        mediana_quantidade = df["quantidade"].median()
        if pd.isna(mediana_quantidade):
            raise ValueError("Não há quantidades válidas para calcular a mediana.")
        df["quantidade"] = df["quantidade"].fillna(mediana_quantidade).astype(int)

        indices_corrigidos = quantidade_original.isna() | (
            quantidade_original != df["quantidade"]
        )
        for indice in df.index[indices_corrigidos]:
            self._auditoria.append(
                {
                    "venda_id": df.at[indice, "venda_id"],
                    "product_id": df.at[indice, "product_id"],
                    "coluna_corrigida": "quantidade",
                    "valor_original": quantidade_original.at[indice],
                    "valor_corrigido": df.at[indice, "quantidade"],
                }
            )
        return df

    def preparar_dados(self) -> None:
        self._auditoria = []
        df_produtos_bruto = self.carregar_csv(self.caminho_produtos)
        df_vendas_bruto = self.carregar_csv(self.caminho_vendas)
        df_produtos_limpo = self.limpar_produtos(df_produtos_bruto)
        df_vendas_limpo = self.limpar_vendas(df_vendas_bruto)

        self.df_completo = pd.merge(
            df_vendas_limpo,
            df_produtos_limpo,
            on="product_id",
            how="left",
            validate="many_to_one",
        )
        self.df_completo["valor_total"] = (
            self.df_completo["quantidade"] * self.df_completo["preco_unitario"]
        )
        valores_ausentes = self.df_completo.isna().sum()
        if valores_ausentes.any():
            raise ValueError(
                "Há valores ausentes após a limpeza: "
                f"{valores_ausentes[valores_ausentes > 0].to_dict()}"
            )
        print("Verificação de qualidade: nenhuma célula ausente após a limpeza.")

    def faturamento_por_categoria(self) -> pd.DataFrame:
        if self.df_completo is None:
            raise ValueError("Execute preparar_dados() antes de gerar métricas.")
        resumo = (
            self.df_completo.groupby("categoria", as_index=False)["valor_total"]
            .sum()
            .rename(
                columns={
                    "categoria": "Categoria",
                    "valor_total": "Faturamento_Total",
                }
            )
            .sort_values(by="Faturamento_Total", ascending=False)
        )
        resumo["Faturamento_Total"] = resumo["Faturamento_Total"].round(2)
        return resumo

    def gerar_grafico_categorias(self, df_resumo: pd.DataFrame) -> None:
        plt.figure(figsize=(10, 6))
        plt.barh(
            df_resumo["Categoria"],
            df_resumo["Faturamento_Total"],
            color="#2b5c8f",
        )
        plt.xlabel("Faturamento Total (R$)")
        plt.title("Faturamento por Categoria - Supermercado (Dados Tratados)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        caminho_imagem = Path(self.pasta_saida) / "grafico_faturamento.png"
        plt.savefig(caminho_imagem, dpi=300)
        plt.close()
        print(f"Gráfico salvo em: {caminho_imagem}")

    def exportar_auditoria(self) -> None:
        colunas = [
            "venda_id",
            "product_id",
            "coluna_corrigida",
            "valor_original",
            "valor_corrigido",
        ]
        auditoria = pd.DataFrame(self._auditoria, columns=colunas)
        self.exportar_csv(auditoria, "auditoria_limpeza.csv")
