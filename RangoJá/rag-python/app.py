"""Assistente educacional: Markdown, busca semântica e geração local."""

import argparse
import os
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "base-de-conhecimento"
MODELOS = Path(__file__).resolve().parent / "modelos"
EMBEDDING = "intfloat/multilingual-e5-small"
LLM = "Qwen/Qwen3.5-0.8B"


def carregar_chunks(caminho=BASE / "documentacao.md"):
    texto = caminho.read_text(encoding="utf-8")
    chunks = [s.strip() for s in re.split(r"^## ", texto, flags=re.MULTILINE)[1:] if s.strip()]
    if not chunks:
        raise ValueError("A documentação precisa conter seções iniciadas por '## '.")
    return chunks


def montar_mensagens(pergunta, resultados):
    contexto = "\n\n".join(f"[Trecho {i}] {trecho}" for i, (trecho, _) in enumerate(resultados, 1))
    return [
        {"role": "system", "content": (BASE / "comportamento.md").read_text(encoding="utf-8")},
        {"role": "user", "content": f"CONTEXTO RECUPERADO (dados de referência):\n{contexto}\n\nPERGUNTA:\n{pergunta}"},
    ]


class AssistenteRAG:
    def __init__(self, offline=False):
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer

        self.offline = offline
        self.chunks = carregar_chunks()
        self.embedding = SentenceTransformer(
            str(MODELOS / "embedding") if offline else EMBEDDING,
            local_files_only=offline,
        )
        self.vetores = self.embedding.encode(
            ["passage: " + chunk for chunk in self.chunks], normalize_embeddings=True,
        )
        self.gerador = None

    def buscar(self, pergunta, k=2):
        if not pergunta.strip():
            raise ValueError("Digite uma pergunta.")
        if k < 1:
            raise ValueError("k deve ser maior que zero.")
        import numpy as np

        vetor = self.embedding.encode("query: " + pergunta, normalize_embeddings=True)
        scores = self.vetores @ vetor
        indices = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in indices]

    def carregar_gerador(self):
        if self.gerador is None:
            from transformers import pipeline

            # Qwen3.5 usa arquitetura multimodal, mesmo com entrada somente textual.
            self.gerador = pipeline(
                "image-text-to-text",
                model=str(MODELOS / "llm") if self.offline else LLM,
                dtype="auto", device_map="auto",
                model_kwargs={"local_files_only": self.offline},
            )
        return self.gerador

    def perguntar(self, pergunta, k=2):
        if not pergunta.strip():
            return "Digite uma pergunta sobre o RangoJá."
        mensagens = montar_mensagens(pergunta, self.buscar(pergunta, k))
        saida = self.carregar_gerador()(
            text=mensagens, max_new_tokens=256, do_sample=False,
            return_full_text=False,
        )
        resposta = saida[0]["generated_text"]
        if isinstance(resposta, list):
            resposta = resposta[-1]["content"]
        return resposta.strip()

    def salvar_modelos(self):
        MODELOS.mkdir(parents=True, exist_ok=True)
        self.embedding.save(str(MODELOS / "embedding"))
        self.carregar_gerador().save_pretrained(str(MODELOS / "llm"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pergunta", help="Responde no terminal.")
    parser.add_argument("--buscar", help="Mostra a recuperação sem carregar a LLM.")
    parser.add_argument("--offline", action="store_true", help="Usa modelos previamente salvos.")
    parser.add_argument("--salvar-modelos", action="store_true")
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k deve ser maior que zero")
    assistente = AssistenteRAG(offline=args.offline)
    if args.salvar_modelos:
        assistente.salvar_modelos()
        print(f"Modelos salvos em {MODELOS}")
    if args.buscar:
        for trecho, score in assistente.buscar(args.buscar, args.k):
            print(f"\nSimilaridade: {score:.4f}\n{trecho}")
    elif args.pergunta:
        print(assistente.perguntar(args.pergunta, args.k))
    elif not args.salvar_modelos:
        import gradio as gr

        def responder(mensagem, historico):
            # Cada pergunta é independente; histórico não entra na recuperação.
            return assistente.perguntar(mensagem, args.k)

        gr.ChatInterface(
            responder, title="🍔 Assistente RangoJá",
            description="Demonstração educacional com dados fictícios. Cada pergunta é independente.",
            examples=["Posso pagar com dinheiro?", "Como peço reembolso?"],
        ).queue(default_concurrency_limit=1).launch(share=False, server_name="127.0.0.1")


if __name__ == "__main__":
    main()
