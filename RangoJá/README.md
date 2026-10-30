# 🍔 RangoJá — assistente com RAG

Assistente de atendimento para um **aplicativo fictício de delivery**, adaptado de um exercício em Python/Colab. Responde a perguntas usando busca semântica em uma documentação própria e geração com uma LLM local.

## Organização

```text
RangoJá/
├── rag-python/
│   ├── app.py
│   └── requirements.txt
└── base-de-conhecimento/
    ├── README.md
    ├── documentacao.md
    ├── comportamento.md
    └── avaliacao.md
```

## Como funciona

1. Cada seção `##` da documentação vira um trecho.
2. O `multilingual-e5-small` gera embeddings normalizados com prefixo `passage:`.
3. A pergunta recebe o prefixo `query:` e é comparada aos trechos por produto escalar, equivalente à similaridade de cosseno neste caso.
4. Os dois trechos mais próximos entram no contexto, junto da pergunta.
5. O Qwen gera a resposta seguindo as instruções de comportamento.

Tecnologias: Python, Sentence Transformers, NumPy, Hugging Face Transformers, PyTorch e Gradio. O índice fica em memória e é reconstruído ao iniciar; não há banco vetorial externo.

## Executar

Recomendado: Python 3.11 ou 3.12 e espaço para baixar os modelos. GPU é opcional; em CPU a geração pode ser lenta. O primeiro uso exige internet. A instalação do PyTorch deve corresponder ao hardware, conforme o [guia oficial](https://pytorch.org/get-started/locally/).

No PowerShell, a partir da raiz do repositório:

```powershell
cd "RangoJá/rag-python"
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

Abra o endereço local exibido pelo Gradio. A interface não cria link público automaticamente. No Linux/macOS, crie o ambiente com `python3 -m venv .venv` e use `.venv/bin/python` nos comandos equivalentes.

```powershell
# Somente recuperação (baixa apenas o modelo de embeddings):
.venv\Scripts\python.exe app.py --buscar "Posso pagar com dinheiro?"

# Resposta completa no terminal:
.venv\Scripts\python.exe app.py --pergunta "Como funciona o reembolso?" --k 2

# Baixar e salvar ambos os modelos antes de ficar sem internet:
.venv\Scripts\python.exe app.py --salvar-modelos
.venv\Scripts\python.exe app.py --offline
```

O modo offline depende de um salvamento completo anterior. Os pesos ficam em `rag-python/modelos/` e não entram no Git. As dependências têm faixas de versão; não há lockfile nem garantia de resultados idênticos entre ambientes.

## Conhecimento e comportamento

Edite as políticas em [`documentacao.md`](base-de-conhecimento/documentacao.md) e reinicie o aplicativo para reconstruir o índice. Edite as instruções em [`comportamento.md`](base-de-conhecimento/comportamento.md). Apenas a documentação é indexada; instruções e avaliações ficam separadas do conteúdo recuperável.

O Markdown também pode ser reutilizado em plataformas que aceitam arquivos de conhecimento e instruções de assistente. Consulte o [guia da base](base-de-conhecimento/README.md).

## Isso já é um agente?

Este projeto implementa um **assistente com RAG e regras de comportamento**. Ele é uma base para evoluir para um agente. Em um uso comum do termo, um agente também decide como usar ferramentas e executar etapas para atingir um objetivo, por exemplo consultar um pedido e solicitar seu cancelamento com os controles adequados. Aqui não há ferramentas de pedidos, planejamento autônomo nem ações no aplicativo.

## Limitações e avaliação

- Dados, marca e contatos fazem parte de um exemplo fictício.
- A busca sempre retorna os `k` trechos mais próximos, mesmo para perguntas sem relação; não há limiar de relevância calibrado.
- Instruções reduzem respostas inventadas, mas não garantem fidelidade ou proteção contra injeção de prompt.
- Cada pergunta é independente; a interface exibe o histórico, mas o modelo não o utiliza.
- O assistente não acessa contas, pedidos, pagamentos ou canais reais de suporte.
- Não houve avaliação quantitativa de qualidade nem validação completa com os pesos dos modelos neste ambiente.

Use os [casos de avaliação](base-de-conhecimento/avaliacao.md) para verificar recuperação, respostas e limites antes de qualquer uso além da demonstração.

## Adaptação do exercício

O script original foi reorganizado em aplicação executável, com carregamento de modelos sob demanda, documentação externa, opções de terminal e modo offline explícito. O prefixo `passege:` foi corrigido para `passage:`, a redação ambígua sobre dinheiro foi esclarecida e o compartilhamento público automático do Gradio foi desativado.

O carregamento do Qwen3.5 foi adaptado para a tarefa `image-text-to-text`, conforme sua arquitetura multimodal, mantendo somente perguntas textuais neste projeto.

## Referências

- [Modelo de embeddings e prefixos E5](https://huggingface.co/intfloat/multilingual-e5-small)
- [Qwen3.5-0.8B e instruções de carregamento](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Gradio ChatInterface](https://www.gradio.app/docs/gradio/chatinterface)
