# De Cero a Preentrenado: NMT Inglés → Español con 4 arquitecturas Seq2Seq

Proyecto Final Integrador · **Procesamiento de Lenguaje Natural (TC3009C)** · Tecnológico de Monterrey
Integrantes: _Bruno Arturo Goñi FLores_ (A01769630) · _Nombre 2_ (A0XXXXXXX)

| Modelo | Arquitectura | Entrenamiento |
|---|---|---|
| **A** | BiLSTM Encoder–Decoder + atención dinámica (Luong, producto punto) | desde cero |
| **B** | Transformer compacto (d=128, 4 heads, d_ff=512, 2 capas) | desde cero |
| **C** | Transformer Base (d=512, 8 heads, d_ff=2048, 6 capas, Adam 1e-4) | desde cero |
| **D** | `Helsinki-NLP/opus-mt-en-es` (MarianMT) | fine-tuning |

## Auditoría de datos

Corpus Tatoeba/ManyThings `spa.txt`, filtrado (≥ 8 palabras EN, ≤ 120 palabras), barajado con `random.Random(23)` y partido 80/10/10, exactamente como en el bloque del enunciado.

| Train | Val | Test | **SHA-256 Test Audit ID** |
|---|---|---|---|
| _—_ | _—_ | _—_ | **`____________`** |

## Resultados en `test_pairs`

_(Pegar aquí `results/tabla1.csv` cuando estén los resultados del notebook 06.)_

| Métrica | A | B | C | D |
|---|---|---|---|---|
| Corpus BLEU global | | | | |
| BLEU medias (8–18) | | | | |
| BLEU largas (>18) | | | | |
| Loss / PPL test | | | | † |
| Parámetros entrenables | | | | |
| Tiempo / época (GPU) | | | | |
| Latencia (ms/oración) | | | | |

† En D la pérdida se mide por subpalabra (SentencePiece), así que no se puede comparar directo con A/B/C (por palabra).

**Hallazgos principales:** _(3–5 viñetas: sesgo inductivo recurrente vs. autoatención, sobreajuste de C, ganancia del transfer learning, comportamiento en oraciones largas.)_

## Cómo reproducir

### En Google Colab (recomendado, GPU T4 o superior)
1. Cambiar `REPO_URL` en la primera celda de cada notebook por la URL de este repositorio.
2. Ejecutar los notebooks **en orden**: cada uno clona el repo, instala `requirements.txt`, descarga `spa.txt` y vuelve a correr el protocolo de partición.
3. Con `USE_DRIVE = True`, los checkpoints y resultados se guardan en `MyDrive/nmt-en-es-tc3009c/results/`. Así, el notebook 06 puede leer lo que generaron los demás aunque se hayan corrido en sesiones distintas.

| Notebook | Contenido | Tiempo aprox. (T4) |
|---|---|---|
| `01_datos_y_auditoria` | protocolo, Audit ID, vocabularios, verificación de fuga, normalización | < 1 min |
| `02_modelo_A_bilstm_atencion` | entrenamiento, curvas, test, latencia, heatmaps | ~15–25 min |
| `03_modelo_B_transformer_compacto` | corrección de `PositionalEmbedding`, entrenamiento, test, heatmaps | ~10 min |
| `04_modelo_C_transformer_base` | ídem, con alta capacidad (diagnóstico de sobreajuste) | ~30–45 min |
| `05_modelo_D_finetuning` | zero-shot, 2+ configuraciones (lr/warmup, weight decay, congelamiento), test | ~20–30 min |
| `06_comparacion_y_analisis` | Tabla 1, curvas de los 4 modelos, heatmaps, análisis cualitativo | < 1 min |

### Local
```bash
pip install -r requirements.txt
wget -q https://www.manythings.org/anki/spa-eng.zip && unzip -q -o spa-eng.zip
jupyter lab notebooks/
```

## Estructura

```
src/
  data.py            protocolo obligatorio, normalize() simétrica, vocabularios (solo train), DataLoaders
  models/
    lstm_attention.py  Modelo A
    transformer.py     Modelos B y C (PositionalEmbedding corregido, MHA desde cero)
  train.py           bucle común A/B/C (Teacher Forcing, AMP, early stopping por val_loss)
  finetune.py        Modelo D (congelamiento parcial, warmup, weight decay, cross-attention)
  evaluate.py        BLEU global/medias/largas (sacrebleu), latencia, heatmaps
  viz.py             curvas Train vs. Val con diagnóstico de sobreajuste, heatmaps
notebooks/           01 → 06
results/             métricas, historias y predicciones en JSON + figuras (sin checkpoints)
```

## Reglas de integridad experimental

1. **Cero fuga de datos.** `eng_word2idx` y `spa_word2idx` se construyen solo con `train_pairs`. Cualquier token de val/test que no esté en el vocabulario se mapea a `<unk>`. La tasa de OOV se reporta en el notebook 01.
2. **Aislamiento de test.** La selección de época y el early stopping usan solo `val_loss`, y la configuración de D se elige con el BLEU de validación. `test_pairs` se evalúa una sola vez por modelo.
3. **Normalización simétrica.** La misma función `src.data.normalize` (minúsculas, sin acentos, sin puntuación) se aplica a las predicciones y referencias de los 4 modelos antes de `sacrebleu`. D se entrena con texto crudo y se normaliza solo para la métrica.
4. **Decodificación greedy** en los 4 modelos, para que la comparación sea justa.
