# 📄 Análise de dados do ENEM entre os anos de 2015 e 2023 

## 📝 Descrição
Neste repositório, realizamos o tratamento e a análise das notas do ENEM entre 2015 e 2023 por município e UF.

Aplicamos técnicas de clusterização para agrupar municípios com perfis semelhantes de desempenho, identificando padrões regionais. Além disso, construímos um gráfico de dispersão para analisar a relação entre nota e renda familiar.

## ⚙️ Instalação
1. Clone o repositório
   ```bash
   git clone https://github.com/lucasamtaylor01/enem.git
   ```

2. Instalar dependências
    
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
Os dados utilizados neste projeto são públicos e podem ser obtidos no site oficial do INEP:  
https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem

Devido ao grande volume, os arquivos não estão incluídos neste repositório. Para executar o projeto, é necessário realizar o download manual dos microdados e organizá-los conforme a estrutura esperada pelo código. Os dados são anonimizados e devem ser utilizados com a devida citação da fonte, em conformidade com a legislação vigente (LGPD).

## 🤝 Contribuições
Contribuições são bem-vindas 😊. Caso encontre erros, inconsistências ou tenha sugestões de melhoria, sinta-se à vontade para abrir uma issue ou enviar um pull request. Toda colaboração ajuda a aprimorar o projeto.

## 🔒 Licença

O código deste repositório está licenciado sob os termos da [licença MIT](LICENSE). Os dados, por sua vez, são derivados de fontes públicas do INEP e não estão cobertos pela licença MIT, mantendo-se sob as condições de uso definidas pelo órgão responsável.

## 🤖 Uso ético de IA
Este projeto foi desenvolvido com a ajuda do [GitHub Copilot](https://github.com/features/copilot).

## 📚 Documentação
A documentação completa do projeto está disponível na wiki: [https://github.com/lucasamtaylor01/enem/wiki](https://github.com/lucasamtaylor01/enem/wiki)
