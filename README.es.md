<h1 align="center">Strata para Mac</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · **Español** · [Português](README.pt-BR.md)

<p align="center"><b>Ejecuta modelos de IA de 100 mil millones de parámetros en tu propio Mac</b><br>
Apple Silicon · probado en un M5 Max con 128 GB · gratis y de código abierto</p>

> **Este es un fork de [Strata](https://github.com/Niko1221/Strata) mantenido de forma independiente, solo para Mac
> con Apple Silicon**, por [dennis-akimov](https://github.com/dennis-akimov). Añade un motor Metal y la instalación en
> Mac. Los motores de Windows y Linux del original (NVIDIA CUDA, AMD HIP, Intel SYCL) siguen en este repositorio tal
> como estaban al hacer el fork, pero aquí no se compilan, prueban ni actualizan.
> **Para un PC con Windows o Linux, usa el original: [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata).**

Strata ejecuta grandes modelos de IA en tu propio ordenador: conversan, escriben código, leen imágenes y trabajan con
tus aplicaciones y agentes de programación a través de las mismas API que los servicios en la nube. El modelo se
ejecuta en tu Mac; lo que sale de él depende de ti (las descargas de modelos y cada aplicación, herramienta o servidor
MCP que conectes).

También crea imágenes a partir de un prompt de texto, con su propio motor de imágenes ([Nucleus-Image](docs/MACOS.md#pictures-from-a-prompt-nucleus-image) o [HunyuanImage-3.0](docs/MACOS.md#pictures-from-a-prompt-hunyuanimage-30)). Y videos cortos a partir de un prompt ([Wan 2.2](docs/MACOS.md#videos-from-a-prompt-wan-22-a14b)).

## ¿Qué tan rápido es?

Medido en un MacBook Pro con M5 Max (GPU de 40 núcleos, 128 GB), macOS 26.4, en el modo de energía *Alto rendimiento*,
2026-10-08, sin razonamiento. Un token es aproximadamente ¾ de una palabra.

| Modelo | Respuestas de 900 palabras | Respuestas cortas | Archivo del modelo en memoria |
| --- | ---: | ---: | ---: |
| **[GPT-OSS 120B](docs/MACOS.md#gpt-oss-120b)** (MXFP4) | 93-100 tokens/s | 93-103 tokens/s | 63 GB |
| **[GPT-OSS 120B con borrador DFlash](docs/MACOS.md#gpt-oss-120b-with-a-dflash-draft)**‡ | 122-135 tokens/s en matemáticas, 71-77 en chat | - | 65 GB |
| **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) Q2_0**, con `--mtp on` | 60-69 tokens/s | 92-102 tokens/s | 38 GB |
| **Qwen3.8-Flash-Next IQ3_XXS**, con `--mtp on` | 46-48 tokens/s | 73-92 tokens/s | 47 GB |
| **[GLM-5.3-Flash](docs/MACOS.md#glm-53-flash) Maya-S24**\* | 13-26 tokens/s | 29-32 tokens/s | 86 GB |
| **[GPT-OSS 20B](docs/MACOS.md#gpt-oss-20b-and-qwen36-35b-a3b)** (MXFP4)† | 128-139 tokens/s | - | 12 GB |
| **[Qwen3.6-35B-A3B](docs/MACOS.md#gpt-oss-20b-and-qwen36-35b-a3b)** (MXFP4_MOE)† | 89-97 tokens/s | - | 22 GB |

\* Medido el 2026-10-10 mientras Docker usaba unos 40 GB y el swap estaba lleno (la GPU a baja frecuencia), así que es un mínimo; GLM además lee unas 3 veces más parámetros activos por token que los demás.

† Medido el 2026-10-12 con respuestas de 600 palabras (800 tokens), con la GPU limitada a 1,2-1,5 GHz tras ejecuciones anteriores, así que es un mínimo.

‡ Medido el 2026-10-13 con respuestas de 300 tokens, frente a 91-99 tokens/s sin el borrador en las mismas ejecuciones: el borrador ayuda en matemáticas, no en el chat.

Imágenes y videos en el mismo Mac (2026-10-11/12):

| Modelo | Salida | Tiempo | Memoria |
| --- | --- | ---: | ---: |
| **[HunyuanImage-3.0](docs/MACOS.md#pictures-from-a-prompt-hunyuanimage-30)** (80B, expertos Q4_0) | imagen 1024x1024, 8 pasos | 88 s | 0,3 GB, su archivo de 54 GB mapeado |
| **[Nucleus-Image](docs/MACOS.md#pictures-from-a-prompt-nucleus-image)** (17B, expertos Q4_0) | imagen 1024x1024, 20 pasos | 45 s | 2 GB, sus 12 GB de archivos mapeados |
| **[Wan 2.2 A14B](docs/MACOS.md#videos-from-a-prompt-wan-22-a14b)** + LoRAs Lightning | video 832x480, 2 s (33 fotogramas), 4 pasos | 430 s | 30 GB en el pico |
| Wan 2.2 A14B sin los LoRAs | el mismo video, 18 pasos | 2907 s | unos 28 GB |

Los tiempos de video se midieron mientras Docker ocupaba 76 GB y macOS usaba swap (la GPU a 600-750 MHz): son un mínimo.

Las cifras de Qwen usan su capa de borrador MTP, que es opcional (`python3 bootstrap.py --setup --mtp on`, unos 12 GB más de
disco). La caché del contexto se suma al archivo del modelo (unos pocos GB; más con contextos largos). En el modo de
energía *Automático*, el mismo Mac escribía las respuestas largas de 2 a 3 veces más lento. No se midieron otros Mac.
Todas las cifras y cómo se midieron: [docs/MACOS.md](docs/MACOS.md#measured) (en inglés).

## Qué necesitas

| | |
| --- | --- |
| **Mac** | Apple Silicon (M1 o posterior). Solo se probó un M5 Max con 128 GB; otros chips y tamaños deberían funcionar, pero no se probaron. |
| **Memoria** | Suficiente para el archivo del modelo y su caché, dentro de lo que macOS permite usar a la GPU (en el Mac probado de 128 GB: 107,5 GB). La comprobación de la instalación cuenta 64 GB como mínimo. `make check` estima qué cabe en tu Mac; mira la tabla de modelos más abajo. |
| **Disco** | La descarga del modelo más unos pocos GB (70-100 GB por modelo), en el SSD interno. |
| **Software** | Las Command Line Tools de Apple (`xcode-select --install`, espera a que termine) y Python 3.10 o posterior (si falta, la instalación lo añade con Homebrew cuando Homebrew está instalado). Probado en macOS 26.4. |

## Instalación

En Terminal:

```sh
git clone https://github.com/dennis-akimov/Strata.git
cd Strata
make check                 # qué puede ejecutar este Mac; no instala nada
make pull MODEL=Q2_0       # compila el motor y descarga el modelo (66 GB)
make run                   # carga el modelo y abre http://127.0.0.1:8080 cuando está listo
```

`make pull` hace algunas preguntas (tamaño del contexto, imágenes); Intro elige la respuesta recomendada. Si la descarga
se interrumpe, vuelve a ejecutarlo: continúa donde se quedó. La carga tarda alrededor de un minuto en cada inicio.
`make run` ocupa la Terminal mientras Strata funciona; Ctrl+C lo detiene, o `make start` / `make stop` lo ejecuta en
segundo plano. Todas las opciones, límites y advertencias: **[docs/MACOS.md](docs/MACOS.md)** (en inglés).

**Actualizar:** `git pull`, luego `make run` (la instalación vuelve a compilar el motor si cambiaron las fuentes; el
modelo se queda). **Dónde está cada cosa:** los modelos en `Strata-data/` junto a la carpeta `Strata`, los ajustes en
`strata-<modelo>.json`, las conversaciones en tu navegador.

## ¿Qué modelo?

| Modelo | Descarga | En memoria, contexto de 128K | Notas |
| --- | ---: | ---: | --- |
| **Qwen3.8-Flash-Next Q2_0** | 66 GB | unos 60 GB (estimación) | El predeterminado y el más probado en Mac. `make pull MODEL=Q2_0` |
| Qwen3.8-Flash-Next IQ3_XXS | 76 GB | 65 GB (medido) | Menos comprimido que Q2_0; un 10-30 % más lento en nuestras pruebas. `make pull MODEL=IQ3_XXS` |
| Qwen3.8-Flash-Next IQ3_S | 84 GB | 74 GB (medido) | Aún menos comprimido; un 10 % más lento que Q2_0 en una prueba A/B. `make pull MODEL=IQ3_S` |
| **GPT-OSS 120B** | 63 GB | unos 70 GB (estimación) | Unos 5 mil millones de parámetros activos por token. Instalar: `make pull MODEL=GPT-OSS-120B`, luego [pasos](docs/MACOS.md#gpt-oss-120b) |
| GPT-OSS 120B + borrador DFlash | 1,6 GB más | unos 72 GB (estimación) | Más rápido en matemáticas y respuestas paso a paso (unos +35 %), más lento en el chat (unos -20 %). `make pull MODEL=GPT-OSS-120B-DFLASH`, luego [pasos](docs/MACOS.md#gpt-oss-120b-with-a-dflash-draft) |
| GPT-OSS 20B | 12 GB | 19 GB (medido) | El más rápido aquí (unos 3.600 millones de parámetros activos por token), menos capaz que el 120B. `make pull MODEL=GPT-OSS-20B`, luego [pasos](docs/MACOS.md#gpt-oss-20b-and-qwen36-35b-a3b) |
| Qwen3.6-35B-A3B | 22 GB | 29 GB (medido) | Unos 3.000 millones de parámetros activos por token, pero aquí no es más rápido que GPT-OSS 120B. `make pull MODEL=QWEN3.6-35B`, luego [pasos](docs/MACOS.md#gpt-oss-20b-and-qwen36-35b-a3b) |
| GLM-5.3-Flash (Maya-S24) | 95 GB | 89 GB con 32K (medido) | El más grande aquí (320 mil millones de parámetros, unos 18 mil millones activos por token) y el más lento. El chat, el razonamiento y las llamadas a herramientas funcionan. Instalar: `make pull MODEL=GLM-5.3`, luego [pasos](docs/MACOS.md#glm-53-flash) |
| HunyuanImage-3.0 (imágenes) | 169 GB, en streaming (quedan 54 GB) | 54 GB mapeados + 0,3 GB | 80B; unos 90 s por imagen. Su licencia no se aplica en la UE, el Reino Unido ni Corea del Sur. `make pull MODEL=HUNYUAN` (unas 2 horas) |
| Nucleus-Image (imágenes) | unos 50 GB (quedan 21 GB) | 12 GB mapeados + 2 GB | 17B, Apache-2.0; unos 45 s por imagen. `make pull MODEL=NUCLEUS` |
| Wan 2.2 T2V A14B (videos) | 29 GB | 30 GB en el pico | Dos expertos de 14B + los LoRAs Lightning; un video de 2 s en unos 7 minutos. `make pull MODEL=WAN2.2` |

«En memoria» es el motor con la capa de borrador MTP de los modelos Qwen y 2 espacios para peticiones, en el Mac
probado; un contexto más pequeño necesita menos (`make run CONTEXT=32768`). Debe caber en lo que macOS permite usar a
la GPU, que `make check` muestra para tu Mac (107,5 GB en el Mac probado de 128 GB; menos en Mac más pequeños). Los
modelos Qwen además leen una tabla de 28 GB del SSD mientras funcionan. Los demás tamaños de Qwen (IQ2_XS, los de
Unsloth) no se probaron en Mac.

## Uso

- **En el navegador:** `http://127.0.0.1:8080` tiene **Chat**, un **Monitor** en vivo del modelo y del Mac (carga de
  la GPU, memoria, potencia, temperatura) y **About**. El esfuerzo de razonamiento (**Thinking**, de *Off* a *High*)
  decide cuánto piensa el modelo; un **Thinking budget** en los ajustes **Sampling** del chat lo limita en tokens.
- **Tus aplicaciones y agentes de programación:** un proveedor «compatible con OpenAI» con la URL base
  `http://127.0.0.1:8080/v1` (cualquier nombre de modelo; cualquier clave de API mientras no definas una), la API de
  Anthropic en `http://127.0.0.1:8080/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`) y la API
  Responses en `/v1/responses`. Las herramientas y MCP funcionan. Ejemplos de clientes:
  [docs/DETAILS.md](docs/DETAILS.md#using-it).
- **Imágenes:** responde que sí a las imágenes en la instalación (modelos Qwen) y luego adjúntalas en el chat o en tu
  aplicación.
- **Crear imágenes:** `/image <prompt>` en el chat, o `POST /v1/images/generations` (la API de imágenes de OpenAI), con
  [Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) (17B) en el motor de imágenes propio de Strata. Una
  imagen de 1024x1024 tarda unos 45-70 s (20-30 pasos) y usa unos pocos GB de memoria además de los 20 GB de archivos
  del modelo, que macOS mapea desde el SSD. Instalar: `make pull MODEL=NUCLEUS`, luego [pasos](docs/MACOS.md#pictures-from-a-prompt-nucleus-image).
  O [HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil) (Tencent, mezcla de expertos de 80B; un solo modelo para el prompt y la imagen, sin
  codificador de texto): 8 pasos, unos 90 s a 1024x1024, 0,3 GB de memoria privada además de su archivo de 54 GB, que
  macOS mapea desde el SSD. Instalar: `make pull MODEL=HUNYUAN` (unas 2 horas), luego [pasos](docs/MACOS.md#pictures-from-a-prompt-hunyuanimage-30). Su licencia no
  se aplica en la UE, el Reino Unido ni Corea del Sur.
- **Crear videos:** `/video <prompt>` en el chat, o `POST /v1/videos` (la API de video de OpenAI: un trabajo que se
  consulta), con [Wan 2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) (dos expertos de 14B) en el backend Metal de [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) y
  [los LoRAs Lightning de lightx2v](https://huggingface.co/lightx2v/Wan2.2-Lightning) (4 pasos en lugar de 18, una pasada por paso). Un video de 2 s a 832x480
  tarda unos 7 minutos (49 sin los LoRAs) y usa 30 GB de memoria en el pico. Instalar: `make pull MODEL=WAN2.2`
  (29 GB, con los LoRAs), luego [pasos](docs/MACOS.md#videos-from-a-prompt-wan-22-a14b).
- **Una petición cada vez** de forma predeterminada; las demás esperan. Un mensaje de seguimiento en la misma
  conversación solo lee lo nuevo, así que empieza rápido; un prompt nuevo y largo tarda un rato en leerse antes de que
  empiece la respuesta.
- **Desde otro dispositivo de la red:** en `strata-<modelo>.json` pon `"host": "0.0.0.0"` y una `"api_key"`, reinicia
  Strata y usa `http://<IP de tu Mac>:8080` con esa clave. El tráfico no va cifrado: úsalo en una red de confianza (o
  una VPN).

## ¿Algo salió mal?

- **Va lento.** Usa el modo de energía *Alto rendimiento* si tu Mac lo tiene (portátiles: *Ajustes del Sistema >
  Batería > Modo de energía*; ordenadores de sobremesa: *Ajustes del Sistema > Energía*) y mantén el portátil
  enchufado. Para ver la frecuencia de la GPU: `brew install macmon` y luego `macmon`.
- **Piensa durante mucho tiempo.** Baja el esfuerzo de razonamiento o pon un **Thinking budget** (4096-8192): cuando
  se agota, el servidor cierra el razonamiento y el modelo escribe su respuesta, si max tokens deja espacio.
- **El puerto está en uso.** Otro programa escucha en 8080. Si es otro Strata, `make stop` lo detiene, o al iniciar un
  `run-<modelo>.sh` se ofrece a detenerlo; si no, usa otro puerto (`make run PORT=8090`). Dos modelos a la vez
  necesitan memoria para ambos.
- **El modelo no cabe.** Cierra aplicaciones grandes, elige un modelo o un contexto más pequeño (`make run CONTEXT=32768`).

Más: [docs/MACOS.md](docs/MACOS.md#limits-and-warnings) (en inglés). Problemas con este fork para Mac:
[issues](https://github.com/dennis-akimov/Strata/issues). Un problema de seguridad: repórtalo en privado, consulta
[SECURITY.md](SECURITY.md).

## ¿Cómo funciona?

En Mac, el motor propio de Strata (`strata-metal`, en [`metal/`](metal/)) ejecuta el modelo en la GPU mediante el
backend Metal de [llama.cpp](https://github.com/ggml-org/llama.cpp), y el servidor de Strata va por encima: la
aplicación web, las API de OpenAI / Anthropic / Responses / MCP, la reutilización de conversaciones, la capa de
borrador del modelo que adivina algunos tokens por adelantado y los formatos de chat de Qwen, GPT-OSS (harmony) y GLM.
La CPU y la GPU comparten la memoria del Mac, así que los pesos del modelo están en ella una sola vez, para ambas.
Diseño y mediciones: [docs/MACOS.md](docs/MACOS.md#how-it-fits-together) (en inglés). Los documentos del original se
conservan en [`docs/`](docs/README.md), indicando cuáles se aplican en Mac.

Las imágenes a partir de un prompt pasan por un segundo motor, `strata-image` (en [`metal/image/`](metal/image/)): el
codificador de texto de Nucleus-Image, su modelo de imágenes de mezcla de expertos de 17B y su decodificador en el mismo
backend Metal, un proceso por imagen. HunyuanImage-3.0 también corre ahí: las claves y valores del prompt una vez, luego 8 pasos de los tokens de
imagen por el transformer de 80B y su VAE 3D.

Los videos corren en `sd-cli` de stable-diffusion.cpp, compilado con un parche de Strata que hace 13 veces más
rápidas en Metal las convoluciones 3D del VAE de Wan; el servidor ejecuta cada clip como un trabajo.

## Créditos y licencia

**[Strata](https://github.com/Niko1221/Strata) es obra de Niko1221 y de los colaboradores de Strata**
([todos los créditos](docs/HOW_IT_WORKS.md#credits)); este fork añade el soporte para Mac. Los modelos son del equipo
de Qwen ([Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), comprimido por
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)), de OpenAI
([GPT-OSS](https://huggingface.co/openai/gpt-oss-120b)) y de Z.ai (GLM-5.3-Flash, comprimido por
[Project Maya](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)). Strata usa partes de
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Código abierto bajo la [licencia MIT](LICENSE) (Copyright
(c) 2026 Niko1221 and the Strata contributors); algunas partes y cada modelo tienen sus propias licencias
([cuáles](docs/HOW_IT_WORKS.md#license)).

Imágenes: [Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) de NucleusAI (Apache-2.0), convertido desde
el port [mlx-nucleus-image](https://huggingface.co/treadon/mlx-nucleus-image).
[HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil) de Tencent ([Tencent Hunyuan Community License](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE): sin licencia en la UE, el Reino Unido ni Corea del Sur).
Videos: [Wan 2.2](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) del equipo Wan de Alibaba (Apache-2.0), ejecutado con [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (MIT).

Si este fork para Mac te resulta útil, puedes apoyar a su mantenedor:
[buymeacoffee.com/dennisakimov](https://buymeacoffee.com/dennisakimov).
