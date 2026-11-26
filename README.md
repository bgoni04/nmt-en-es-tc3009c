# De Cero a Preentrenado: NMT Inglés → Español con 4 arquitecturas Seq2Seq

Proyecto Final Integrador · **Procesamiento de Lenguaje Natural (TC3009C)** · Tecnológico de Monterrey

**Integrantes:** Bruno Arturo Goñi Flores (A01769630) · Manuel Antonio Pérez Fonseca (A01741957)

Comparamos cuatro modelos de traducción automática inglés → español entrenados y evaluados sobre exactamente la misma partición del corpus Tatoeba/ManyThings (`spa.txt`).

| Modelo | Arquitectura | Entrenamiento | Estado |
|---|---|---|---|
| **A** | BiLSTM Encoder–Decoder + atención dinámica (Luong, producto punto) | desde cero | pendiente |
| **B** | Transformer compacto (d=128, 4 heads, d_ff=512, 2 capas) | desde cero | pendiente |
| **C** | Transformer Base (d=512, 8 heads, d_ff=2048, 6 capas, Adam 1e-4) | desde cero | pendiente |
| **D** | `Helsinki-NLP/opus-mt-en-es` (MarianMT) | fine-tuning | pendiente |

## Avance

- [x] Estructura del repositorio y `requirements.txt`
- [x] Protocolo obligatorio de carga, filtrado y partición 80/10/10 (`src/data.py`)
- [x] Normalización simétrica para BLEU
- [x] Vocabularios construidos solo con `train_pairs`, con `<unk>` para palabras desconocidas
- [x] DataLoaders con padding (`PAD_IDX = 0`)
- [x] Notebook 01: auditoría de datos y verificación de las reglas de integridad
- [ ] Modelo A — BiLSTM + atención
- [ ] Modelos B y C — Transformer (con corrección de `PositionalEmbedding`)
- [ ] Modelo D — fine-tuning con al menos 2 configuraciones de hiperparámetros
- [ ] Evaluación: BLEU global / medias / largas, perplejidad, parámetros, tiempo y latencia
- [ ] Curvas Train vs. Validation, heatmaps de atención y análisis cualitativo
- [ ] Reporte PDF y video

## Auditoría de datos

Corpus filtrado (≥ 8 palabras en inglés, ≤ 120 palabras), barajado con `random.Random(23)` y partido 80/10/10, exactamente como el bloque de la Sección 3 del enunciado.

| Train | Val | Test | SHA-256 Test Audit ID |
|---|---|---|---|
| 29,130 | 3,641 | 3,642 | **`dbac04ac5648`** |

| Dato | Valor |
|---|---|
| Vocabulario EN / ES (`min_freq = 2`, solo train) | 5,529 / 8,026 |
| Palabras fuera de vocabulario en test (EN / ES) | 2.16 % / 4.37 % |
| Test: oraciones medias (8–18) / largas (> 18) | 3,611 / 31 |
| Oraciones de test cuya fuente EN también aparece en train | 564 (15.49 %) |

Estos valores salen al ejecutar `notebooks/01_datos_y_auditoria.ipynb` y quedan guardados en `results/data_audit.json`. En Tatoeba una misma oración en inglés puede tener varias traducciones, por eso algunas fuentes se repiten entre train y test. No es fuga de datos según el protocolo, pero lo tomamos en cuenta al interpretar el BLEU.

## Resultados en `test_pairs`

_Pendiente: se llenará cuando estén entrenados los cuatro modelos._

| Métrica | A | B | C | D |
|---|---|---|---|---|
| Corpus BLEU global (sacrebleu) | | | | |
| BLEU oraciones medias (8–18) | | | | |
| BLEU oraciones largas (> 18) | | | | |
| Loss / Perplejidad en test | | | | |
| Parámetros entrenables | | | | |
| Tiempo / época (GPU) | | | | |
| Latencia (ms/oración) | | | | |

## Cómo reproducir

### Google Colab (recomendado, GPU T4 o superior)

1. Abrir el notebook en Colab y activar GPU.
2. Ejecutar todas las celdas. La primera celda clona este repositorio, instala `requirements.txt` y la segunda descarga `spa.txt`.
3. Con `USE_DRIVE = True` los resultados se guardan en `MyDrive/nmt-en-es-tc3009c/results/`, para poder usarlos entre sesiones.

### Local

```bash
pip install -r requirements.txt
wget -q https://www.manythings.org/anki/spa-eng.zip && unzip -q -o spa-eng.zip
jupyter lab notebooks/
```

### Pruebas

Con `spa.txt` en la raíz del repositorio:

```bash
python -m pytest
```

Verifican la partición y el Test Audit ID, la normalización, los vocabularios, `encode`/`decode` y los DataLoaders.

## Estructura

```
src/
  data.py      protocolo de partición, normalize(), vocabularios (solo train), DataLoaders
  utils.py     semilla, dispositivo, conteo de parámetros, cronómetro, guardado de JSON
notebooks/
  01_datos_y_auditoria.ipynb
tests/
  test_data.py pruebas de src/data.py y src/utils.py
results/       métricas y figuras (los checkpoints .pt no se suben)
```

## Reglas de integridad experimental

1. **Cero fuga de datos.** `eng_word2idx` y `spa_word2idx` se construyen solo con `train_pairs`; cualquier palabra de val/test fuera del vocabulario se mapea a `<unk>`.
2. **Aislamiento de test.** La selección de época y el early stopping usan solo validación. `test_pairs` se evalúa una sola vez por modelo, al final.
3. **Normalización simétrica.** La misma función `src.data.normalize` (minúsculas, sin acentos, sin puntuación) se aplica a predicciones y referencias de los cuatro modelos antes de calcular BLEU.
