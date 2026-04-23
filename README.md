# OCR Benchmark

Framework de avaliação comparativa de modelos OCR, medindo precisão de extração e desempenho entre diferentes engines sobre os mesmos documentos PDF.

---

## Visão Geral

Este projeto executa benchmark dos modelos listados abaixo, rodando cada um sobre os mesmos PDFs e medindo **Character Error Rate (CER)**, **Word Error Rate (WER)** e **tempo de execução**, gerando ao final um relatório visual consolidado.

Toda execução de OCR salva automaticamente o resultado em `data/extracted/`, seguindo o padrão:

```
{modelo_ocr}_output_{nome_do_arquivo_extraido}.json
```

---

## Modelos Avaliados

| Modelo | Versão | Backend |
|---|---|---|
| GLM-OCR | `glm-ocr:bf16` | Ollama 0.16.3 / `ollama` 0.6.1 |
| Mistral OCR | `mistral-ocr-2512` | Mistral API |
| EasyOCR | 1.7.2 | — |
| PaddleOCR | 3.4.0 | PaddlePaddle 3.3.0 |
| Tesseract OCR | 5.3.4 | `pytesseract` 0.3.13 |

### Modelos Descartados

**`mistral-small3.2`** foi excluído do benchmark. Embora suporte entrada de imagem via Ollama, exige mais VRAM do que a máquina de teste oferece, o que aciona execução híbrida CPU/GPU e torna os resultados de tempo incomparáveis com os demais modelos.

---

## Máquina de Teste

Todos os benchmarks foram executados na seguinte configuração:

| Componente | Especificação |
|---|---|
| **CPU** | Intel Core i7-13700 (Raptor Lake, 13ª geração) |
| **GPU** | NVIDIA GeForce RTX 4060 (8 GB VRAM) |
| **RAM** | 32 GB DDR5 |

> **Observação:** A GPU dedicada (RTX 4060) possui 8 GB de VRAM. Modelos que excedam esse limite foram descartados do benchmark para garantir comparabilidade.

---

## Pré-requisitos

Todos os benchmarks dependem de `pdf2image`, que requer o binário `pdftoppm` do **Poppler** disponível no `PATH`.

Cada modelo roda em seu próprio ambiente virtual isolado em `ocr_models/<modelo>/`.

---

## Configuração e Execução

### GLM

```bash
cd ocr_models/GLM
python3 -m venv glmvenv
source glmvenv/bin/activate
pip install -r glm_requirements.txt
ollama pull glm-ocr:bf16
python3 glm_benchmark.py ../../data/pdf1.pdf
```

> **Observação:** Requer o Ollama em execução (`ollama serve`) e o modelo baixado antes da execução.

---

### EasyOCR

```bash
cd ocr_models/easyocr
python3 -m venv easyocrvenv
source easyocrvenv/bin/activate
pip install -r easyocr_requirements.txt
python3 easyocr_benchmark.py ../../data/pdf1.pdf
```

---

### PaddleOCR

```bash
cd ocr_models/paddleocr
python3 -m venv paddleocrvenv
source paddleocrvenv/bin/activate

# CPU
pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# GPU (CUDA 13.0) — substitua o comando acima por:
# pip install paddlepaddle-gpu==3.3.0 \
#     --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/

pip install -r paddleocr_requirements.txt
python3 paddleocr_benchmark.py ../../data/pdf1.pdf
```

> **Observação:** Instale o `paddlepaddle` (CPU ou GPU) **antes** de rodar `pip install -r`, pois o pacote é omitido intencionalmente do `requirements.txt` para evitar conflitos de índice.

---

### Tesseract

```bash
cd ocr_models/tesseract
python3 -m venv tesseractvenv
source tesseractvenv/bin/activate
pip install -r tesseract_requirements.txt
python3 tesseract_benchmark.py ../../data/pdf1.pdf
```

> **Observação:** O script usa automaticamente o wrapper local em `ocr_models/tesseract/local/bin/tesseract` quando disponível. Se necessário, também aceita um binário `tesseract` instalado no sistema.

---

### Mistral

```bash
cd ocr_models/mistral
python3 -m venv mistralvenv
source mistralvenv/bin/activate
pip install -r mistral_requirements.txt
python3 mistral_benchmark.py ../../data/pdf1.pdf
```

> **Observação:** Requer uma chave de API da Mistral configurada via `.env`, variável de ambiente (`MISTRAL_API_KEY` ou `API_KEY`) ou pelo argumento `--api-key`.

---

## Avaliação

### Métricas CER / WER

Para avaliar todos os resultados extraídos contra o gabarito `data/pdf2_gabarito.json`:

```bash
python3 evaluate_ocr_metrics.py data/pdf2_gabarito.json
```

O script descobre automaticamente todos os arquivos em `data/extracted/*_output_*.json`, calcula CER e WER por documento e por página, imprime um ranking no terminal e salva os resultados completos em `data/pdf2_metrics.json`.

### Visualização

Para gerar um painel visual comparando CER, WER, tempo total de execução e erro por página:

```bash
cd plot
python3 -m pip install -r plot_requirements.txt
python3 plot_ocr_results.py ../data/pdf2_metrics.json
```

A imagem gerada é salva por padrão em `plot/<nome_do_documento>_benchmark.png` (por exemplo, `plot/pdf2_gabarito_benchmark.png`).
