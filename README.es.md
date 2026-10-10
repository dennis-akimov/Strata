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

## ¿Qué tan rápido es?

Medido en un MacBook Pro con M5 Max (GPU de 40 núcleos, 128 GB), macOS 26.4, en el modo de energía *Alto rendimiento*,
2026-10-08, sin razonamiento. Un token es aproximadamente ¾ de una palabra.

| Modelo | Respuestas de 900 palabras | Respuestas cortas | Archivo del modelo en memoria |
| --- | ---: | ---: | ---: |
| **[GPT-OSS 120B](docs/MACOS.md#gpt-oss-120b)** (MXFP4) | 93-100 tokens/s | 93-103 tokens/s | 63 GB |
| **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) Q2_0**, con `--mtp on` | 60-69 tokens/s | 92-102 tokens/s | 38 GB |
| **Qwen3.8-Flash-Next IQ3_XXS**, con `--mtp on` | 46-48 tokens/s | 73-92 tokens/s | 47 GB |
| **[GLM-5.3-Flash](docs/MACOS.md#glm-53-flash) Maya-S24**\* | 13-26 tokens/s | 29-32 tokens/s | 86 GB |

\* Medido el 2026-10-10 mientras Docker usaba unos 40 GB y el swap estaba lleno (la GPU a baja frecuencia), así que es un mínimo; GLM además lee unas 3 veces más parámetros activos por token que los demás.

Las cifras de Qwen usan su capa de borrador MTP, que es opcional (`./setup.sh --setup --mtp on`, unos 12 GB más de
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
| **GPT-OSS 120B** | 63 GB | unos 70 GB (estimación) | El más rápido aquí (unos 5 mil millones de parámetros activos por token). Se instala a mano: [pasos](docs/MACOS.md#gpt-oss-120b) |
| GLM-5.3-Flash (Maya-S24) | 95 GB | 89 GB con 32K (medido) | El más grande aquí (320 mil millones de parámetros, unos 18 mil millones activos por token) y el más lento. El chat, el razonamiento y las llamadas a herramientas funcionan. Se instala a mano: [pasos](docs/MACOS.md#glm-53-flash) |

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

Si este fork para Mac te resulta útil, puedes apoyar a su mantenedor:
[buymeacoffee.com/dennisakimov](https://buymeacoffee.com/dennisakimov).
