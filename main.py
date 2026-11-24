from pathlib import Path

from src.analisador import AnalisadorSupermercado


def executar_pipeline() -> None:
    pasta_projeto = Path(__file__).resolve().parent
    analisador = AnalisadorSupermercado(
        caminho_produtos=pasta_projeto / "dados" / "produtos.csv",
        caminho_vendas=pasta_projeto / "dados" / "vendas.csv",
        pasta_saida=pasta_projeto / "relatorios",
    )
    analisador.preparar_dados()
    relatorio = analisador.faturamento_por_categoria()
    analisador.exportar_csv(relatorio, "faturamento_por_categoria.csv")
    analisador.gerar_grafico_categorias(relatorio)
    analisador.exportar_auditoria()
    print("\nFaturamento por categoria:")
    print(relatorio.to_string(index=False))


if __name__ == "__main__":
    executar_pipeline()
