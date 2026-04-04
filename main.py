from utils.clustering import (
    processar_ano,
    rodar_todos_os_anos,
)
from utils.carregamento_dados import (
    ANOS_DISPONIVEIS,
)

def main():
	entrada = input("Digite o ano para análise (2015-2023) ou 'todos': ").strip().lower()

	if entrada == "todos":
		rodar_todos_os_anos()
	else:
		try:
			ano_escolhido = int(entrada)
			if ano_escolhido not in ANOS_DISPONIVEIS:
				raise ValueError
			processar_ano(ano_escolhido)
		except ValueError:
			print("Entrada inválida. Digite um ano entre 2015 e 2023 ou 'todos'.")


if __name__ == "__main__":
	main()