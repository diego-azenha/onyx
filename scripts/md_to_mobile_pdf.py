#!/usr/bin/env python
"""Markdown -> PDF de página pequena, legível em celular sem zoom.

## O problema que isto resolve

PDF é layout FIXO. Um A4 no telefone obriga a dar zoom e arrastar lateralmente a cada linha — a
experiência de leitura fica pior que a do próprio Markdown. A correção não é diminuir a fonte: é
diminuir a PÁGINA. Com `@page { size: 95mm 170mm }` cada página enche a tela de um celular moderno na
vertical, e o leitor apenas passa páginas.

## Como funciona

1. As expressões matemáticas (`$...$` e `$$...$$`) são extraídas para marcadores ANTES da conversão.
   Sem isso o Markdown interpreta `_` e `*` dentro da matemática como ênfase e corrompe as fórmulas —
   `\hat\sigma_i` vira itálico no meio de um subscrito.
2. Converte-se o Markdown com extensões de tabela e nota de rodapé.
3. A matemática é reinserida e renderizada por MathJax no navegador.
4. Um navegador headless (Edge ou Chrome) imprime em PDF com a geometria de página pequena.
   `--virtual-time-budget` é essencial: sem ele o navegador imprime antes de o MathJax terminar e o
   PDF sai com a matemática em código-fonte.

Requer apenas `markdown` (pip) e um navegador baseado em Chromium — sem LaTeX, sem pandoc.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

NAVEGADORES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

CSS = """
@page {{
  size: {largura} {altura};
  margin: {margem};
}}
html {{ -webkit-print-color-adjust: exact; }}
body {{
  font-family: "Georgia", "Times New Roman", serif;
  font-size: {fonte};
  line-height: 1.55;
  color: #17181c;
  margin: 0;
  hyphens: auto;
  -webkit-hyphens: auto;
  text-align: justify;
}}
h1, h2, h3, h4 {{
  font-family: "Segoe UI", "Helvetica Neue", sans-serif;
  line-height: 1.25;
  text-align: left;
  break-after: avoid-page;
  page-break-after: avoid;
  margin: 1.1em 0 0.45em;
}}
h1 {{ font-size: 1.45em; margin-top: 0; }}
h2 {{ font-size: 1.18em; border-bottom: 1px solid #d8dade; padding-bottom: 0.2em; }}
h3 {{ font-size: 1.02em; }}
h4 {{ font-size: 0.95em; font-style: italic; }}
p {{ margin: 0 0 0.7em; orphans: 2; widows: 2; }}
strong {{ color: #000; }}
code {{
  font-family: "Cascadia Mono", Consolas, monospace;
  font-size: 0.82em;
  background: #f2f3f5;
  padding: 0.1em 0.25em;
  border-radius: 3px;
}}
pre {{
  background: #f6f7f9;
  border-left: 2px solid #c4c8cf;
  padding: 0.5em 0.6em;
  overflow-x: auto;
  font-size: 0.74em;
  line-height: 1.4;
  break-inside: avoid-page;
}}
pre code {{ background: none; padding: 0; }}
/* Tabelas são o ponto crítico numa página estreita: fonte menor, quebra de palavra permitida,
   e nunca partidas entre páginas quando couberem. */
table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 0.70em;
  font-family: "Segoe UI", sans-serif;
  margin: 0.7em 0 1em;
  break-inside: avoid-page;
  page-break-inside: avoid;
}}
th, td {{
  border-bottom: 1px solid #dfe1e5;
  padding: 0.32em 0.35em;
  text-align: left;
  vertical-align: top;
  word-break: break-word;
  hyphens: auto;
}}
th {{ background: #eef0f3; font-weight: 600; border-bottom: 1.5px solid #b9bec6; }}
tr:last-child td {{ border-bottom: 1.5px solid #b9bec6; }}
blockquote {{
  margin: 0.7em 0;
  padding-left: 0.8em;
  border-left: 2.5px solid #c4c8cf;
  color: #43464d;
}}
hr {{ border: none; border-top: 1px solid #d8dade; margin: 1.4em 0; }}
ul, ol {{ padding-left: 1.25em; margin: 0 0 0.7em; }}
li {{ margin-bottom: 0.28em; }}
/* Blocos matemáticos numa página estreita. `overflow: auto` seria inútil num PDF — o excesso
   simplesmente some. A solução é escalar o que não couber (ver o script de ajuste no final). */
mjx-container[display="true"] {{
  font-size: 0.92em !important;
  margin: 0.7em 0 !important;
  break-inside: avoid-page;
  display: block;
  overflow: visible;
}}
.mjx-fit {{ transform-origin: left center; }}
mjx-container {{ font-size: 0.95em !important; }}
"""

HTML = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>{titulo}</title>
<style>{css}</style>
<script>
  window.MathJax = {{
    tex: {{ inlineMath: [['$','$']], displayMath: [['$$','$$']] }},
    options: {{ enableMenu: false }},
    svg: {{ fontCache: 'global' }}
  }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head><body>
{corpo}
<script>
  // Numa página estreita, uma equação larga simplesmente EXTRAVASA e é cortada na impressão —
  // `overflow: auto` não ajuda porque num PDF não há rolagem. Aqui, depois que o MathJax termina,
  // cada bloco mais largo que a coluna é reduzido por `transform: scale` até caber. A alternativa
  // seria diminuir a fonte de TODAS as fórmulas, penalizando as que já cabiam.
  MathJax.startup.promise.then(function () {{
    document.querySelectorAll('mjx-container[display="true"]').forEach(function (el) {{
      var disponivel = el.parentElement.clientWidth;
      var necessario = el.scrollWidth || el.getBoundingClientRect().width;
      if (necessario > disponivel + 1) {{
        var f = Math.max(0.45, disponivel / necessario);
        el.classList.add('mjx-fit');
        el.style.transform = 'scale(' + f + ')';
        el.style.height = (el.getBoundingClientRect().height * f) + 'px';
      }}
    }});
    document.title = document.title + ' ✓';   // marcador de que o ajuste rodou
  }});
</script>
</body></html>
"""


def _proteger_matematica(texto: str) -> tuple[str, list[str]]:
    """Extrai `$$...$$` e `$...$` para marcadores. Sem isto o Markdown come `_` e `*` dentro das
    fórmulas e transforma subscritos em itálico."""
    guardadas: list[str] = []

    def guardar(m: re.Match) -> str:
        guardadas.append(m.group(0))
        return f"\x00MATH{len(guardadas)-1}\x00"

    texto = re.sub(r"\$\$.+?\$\$", guardar, texto, flags=re.S)
    texto = re.sub(r"(?<!\$)\$(?!\s)(?:[^$\n]|\n(?!\n))+?(?<!\s)\$(?!\$)", guardar, texto)
    return texto, guardadas


def _restaurar_matematica(html: str, guardadas: list[str]) -> str:
    for i, expr in enumerate(guardadas):
        html = html.replace(f"\x00MATH{i}\x00", expr)
    return html


def _navegador() -> str:
    for p in NAVEGADORES:
        if Path(p).exists():
            return p
    achado = shutil.which("chrome") or shutil.which("msedge")
    if achado:
        return achado
    raise SystemExit("nenhum navegador baseado em Chromium encontrado (Edge ou Chrome)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada")
    ap.add_argument("--out", default=None, help="default: mesmo nome, extensão .pdf")
    ap.add_argument("--largura", default="95mm", help="largura da página (default 95mm, ~tela de celular)")
    ap.add_argument("--altura", default="170mm")
    ap.add_argument("--margem", default="7mm 6mm 9mm 6mm")
    ap.add_argument("--fonte", default="10.2pt")
    ap.add_argument("--espera", type=int, default=20000, help="ms para o MathJax renderizar antes de imprimir")
    ap.add_argument("--manter-html", action="store_true")
    args = ap.parse_args()

    entrada = Path(args.entrada)
    saida = Path(args.out) if args.out else entrada.with_suffix(".pdf")

    texto = entrada.read_text(encoding="utf-8")
    texto, guardadas = _proteger_matematica(texto)
    corpo = markdown.markdown(texto, extensions=["tables", "fenced_code", "footnotes", "attr_list", "sane_lists"])
    corpo = _restaurar_matematica(corpo, guardadas)

    css = CSS.format(largura=args.largura, altura=args.altura, margem=args.margem, fonte=args.fonte)
    titulo = next((l.lstrip("# ").strip() for l in texto.splitlines() if l.startswith("# ")), entrada.stem)
    html = HTML.format(titulo=titulo, css=css, corpo=corpo)

    destino_html = saida.with_suffix(".html") if args.manter_html else Path(tempfile.mkdtemp()) / "doc.html"
    destino_html.write_text(html, encoding="utf-8")

    cmd = [
        _navegador(), "--headless=new", "--disable-gpu", "--no-sandbox",
        f"--virtual-time-budget={args.espera}",       # espera o MathJax; sem isto sai LaTeX cru
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={saida.resolve()}",
        destino_html.resolve().as_uri(),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not saida.exists():
        sys.exit(f"falhou ao gerar o PDF:\n{r.stderr[-1500:]}")

    kb = saida.stat().st_size / 1024
    print(f"gerado {saida}  ({kb:.0f} KB, pagina {args.largura}x{args.altura}, fonte {args.fonte})")
    print(f"  {len(guardadas)} expressoes matematicas renderizadas")
    if args.manter_html:
        print(f"  HTML preservado em {destino_html}")


if __name__ == "__main__":
    main()
