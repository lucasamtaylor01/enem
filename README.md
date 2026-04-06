# 📄 Análise de dados Exame Nacional do Ensino Médio (ENEM) (2015-2023)

## 📝 Descrição
Neste repositório, realizamos o tratamento e a análise das notas do ENEM entre 2015 e 2023 por município e UF.

Aplicamos técnicas de clusterização para agrupar municípios com perfis semelhantes de desempenho, identificando padrões regionais. Além disso, construímos um gráfico de dispersão para analisar a relação entre nota e renda familiar.

## Instalação
1. Clone o repositório
   ```bash
   git clone https://github.com/lucasamtaylor01/enem.git
   ```

2. ⚙️ Instalar dependências
    
   **Linux/macOS:**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```
   **Windows (PowerShell)**
   ```bash
   python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
   ```

3. Executar `main.py`

## ⚠️ Atenção 
Os dados utilizados neste projeto são públicos e podem ser obtidos diretamente no site oficial do INEP:
[www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem)

Devido ao grande volume desses arquivos, eles não foram incluídos neste repositório do GitHub. Portanto, para executar o código corretamente, é necessário fazer o download dos microdados manualmente a partir do link acima.

Após o download, certifique-se de organizar os arquivos na estrutura esperada pelo projeto (conforme indicado no código ou na documentação). Ressalta-se que os dados são públicos, anonimizados e disponibilizados pelo INEP, devendo ser utilizados com a devida citação da fonte e em conformidade com a legislação vigente, incluindo a Lei Geral de Proteção de Dados (LGPD).

## 🤝 Contribuições

Contribuições são muito bem-vindas 😊. 

Este projeto foi desenvolvido com base em dados extensos e etapas de tratamento que podem estar sujeitas a imprecisões ou erros. Caso identifique algum problema, inconsistência ou tenha sugestões de melhoria, sinta-se à vontade para abrir uma issue ou enviar um pull request. 

Toda colaboração é importante para aprimorar a qualidade, a reprodutibilidade e a utilidade do projeto.

## 🔒 Licença

O código deste repositório está licenciado sob os termos da [licença MIT](LICENSE).

Os dados, por sua vez, são derivados de fontes públicas do INEP e não estão cobertos pela licença MIT, mantendo-se sob as condições de uso definidas pelo órgão responsável.

## 🤖 Uso ético de IA
Este projeto foi desenvolvido com a ajuda do [GitHub Copilot](https://github.com/features/copilot).

## 📚 Documentação
A documentação completa do projeto está disponível na wiki: [https://github.com/lucasamtaylor01/enem/wiki](https://github.com/lucasamtaylor01/enem/wiki)