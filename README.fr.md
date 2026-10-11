<h1 align="center">Strata pour Mac</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · **Français** · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Faites tourner des modèles d'IA de 100 milliards de paramètres sur votre propre Mac</b><br>
Apple Silicon · testé sur un M5 Max avec 128 Go · gratuit et open source</p>

> **Ceci est un fork de [Strata](https://github.com/Niko1221/Strata) maintenu de façon indépendante, uniquement pour
> les Mac Apple Silicon**, par [dennis-akimov](https://github.com/dennis-akimov). Il ajoute un moteur Metal et
> l'installation sur Mac. Les moteurs Windows et Linux de l'original (NVIDIA CUDA, AMD HIP, Intel SYCL) restent dans ce
> dépôt tels qu'ils étaient au moment du fork, mais ils ne sont ni compilés, ni testés, ni mis à jour ici.
> **Pour un PC Windows ou Linux, utilisez l'original : [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata).**

Strata fait tourner de grands modèles d'IA sur votre propre ordinateur : ils discutent, écrivent du code, lisent des
images et travaillent avec vos applications et vos agents de code via les mêmes API que les services cloud. Le modèle
tourne sur votre Mac ; ce qui en sort dépend de vous (les téléchargements de modèles, et chaque application, outil ou
serveur MCP que vous connectez).

## Quelle vitesse ?

Mesuré sur un MacBook Pro avec M5 Max (GPU 40 cœurs, 128 Go), macOS 26.4, en mode d'énergie *Haute puissance*,
2026-10-08, sans réflexion. Un token vaut environ ¾ d'un mot.

| Modèle | Réponses de 900 mots | Réponses courtes | Fichier du modèle en mémoire |
| --- | ---: | ---: | ---: |
| **[GPT-OSS 120B](docs/MACOS.md#gpt-oss-120b)** (MXFP4) | 93-100 tokens/s | 93-103 tokens/s | 63 Go |
| **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) Q2_0**, avec `--mtp on` | 60-69 tokens/s | 92-102 tokens/s | 38 Go |
| **Qwen3.8-Flash-Next IQ3_XXS**, avec `--mtp on` | 46-48 tokens/s | 73-92 tokens/s | 47 Go |
| **[GLM-5.3-Flash](docs/MACOS.md#glm-53-flash) Maya-S24**\* | 13-26 tokens/s | 29-32 tokens/s | 86 Go |

\* Mesuré le 2026-10-10 alors que Docker utilisait environ 40 Go et que le swap était plein (GPU à basse fréquence) : c'est un minimum ; GLM lit aussi environ 3x plus de paramètres actifs par token que les autres.

Les chiffres de Qwen utilisent sa couche de brouillon MTP, qui est optionnelle (`./setup.sh --setup --mtp on`, environ
12 Go de disque en plus). Le cache du contexte s'ajoute au fichier du modèle (quelques Go ; davantage pour les longs
contextes). En mode d'énergie *Automatique*, le même Mac écrivait les longues réponses 2 à 3 fois plus lentement.
Aucun autre Mac n'a été mesuré. Tous les chiffres et leur méthode : [docs/MACOS.md](docs/MACOS.md#measured) (en anglais).

## Ce qu'il vous faut

| | |
| --- | --- |
| **Mac** | Apple Silicon (M1 ou plus récent). Seul un M5 Max avec 128 Go a été testé ; les autres puces et tailles devraient fonctionner, mais n'ont pas été essayées. |
| **Mémoire** | Assez pour le fichier du modèle et son cache, dans la limite que macOS accorde au GPU (sur le Mac testé de 128 Go : 107,5 Go). La vérification de l'installation compte 64 Go comme minimum. `make check` estime ce qui tient sur votre Mac ; voir le tableau des modèles ci-dessous. |
| **Disque** | Le téléchargement du modèle plus quelques Go (70 à 100 Go par modèle), sur le SSD interne. |
| **Logiciels** | Les Command Line Tools d'Apple (`xcode-select --install`, attendez la fin) et Python 3.10 ou plus récent (s'il manque, l'installation l'ajoute avec Homebrew si Homebrew est présent). Testé sous macOS 26.4. |

## Installation

Dans le Terminal :

```sh
git clone https://github.com/dennis-akimov/Strata.git
cd Strata
make check                 # ce que ce Mac peut faire tourner ; n'installe rien
make pull MODEL=Q2_0       # compile le moteur et télécharge le modèle (66 Go)
make run                   # charge le modèle et ouvre http://127.0.0.1:8080 quand il est prêt
```

`make pull` pose quelques questions (taille du contexte, images) ; Entrée choisit la réponse recommandée. Si le
téléchargement s'interrompt, relancez-le : il reprend là où il s'est arrêté. Le chargement prend environ une minute à
chaque démarrage. `make run` occupe le Terminal tant que Strata tourne ; Ctrl+C l'arrête, ou `make start` /
`make stop` le fait tourner en arrière-plan. Toutes les options, limites et mises en garde :
**[docs/MACOS.md](docs/MACOS.md)** (en anglais).

**Mise à jour :** `git pull`, puis `make run` (l'installation recompile le moteur si les sources ont changé ; le modèle
reste). **Où sont les choses :** les modèles dans `Strata-data/` à côté du dossier `Strata`, les réglages dans
`strata-<modèle>.json`, les conversations dans votre navigateur.

## Quel modèle ?

| Modèle | Téléchargement | En mémoire, contexte 128K | Remarques |
| --- | ---: | ---: | --- |
| **Qwen3.8-Flash-Next Q2_0** | 66 Go | environ 60 Go (estimation) | Le choix par défaut, le plus testé sur Mac. `make pull MODEL=Q2_0` |
| Qwen3.8-Flash-Next IQ3_XXS | 76 Go | 65 Go (mesuré) | Moins compressé que Q2_0 ; 10 à 30 % plus lent dans nos essais. `make pull MODEL=IQ3_XXS` |
| Qwen3.8-Flash-Next IQ3_S | 84 Go | 74 Go (mesuré) | Encore moins compressé ; environ 10 % plus lent que Q2_0 dans un test A/B. `make pull MODEL=IQ3_S` |
| **GPT-OSS 120B** | 63 Go | environ 70 Go (estimation) | Le plus rapide ici (environ 5 milliards de paramètres actifs par token). À installer à la main : [étapes](docs/MACOS.md#gpt-oss-120b) |
| GLM-5.3-Flash (Maya-S24) | 95 Go | 89 Go à 32K (mesuré) | Le plus gros ici (320 milliards de paramètres, environ 18 milliards actifs par token) et le plus lent. Discussion, réflexion et appels d'outils fonctionnent. À installer à la main : [étapes](docs/MACOS.md#glm-53-flash) |

« En mémoire » est le moteur avec la couche de brouillon MTP des modèles Qwen et 2 emplacements de requête, sur le Mac
testé ; un contexte plus petit demande moins (`make run CONTEXT=32768`). Cela doit tenir dans ce que macOS accorde au
GPU, que `make check` affiche pour votre Mac (107,5 Go sur le Mac testé de 128 Go ; moins sur les Mac plus petits).
Les modèles Qwen lisent aussi une table de 28 Go sur le SSD pendant qu'ils tournent. Les autres tailles de Qwen
(IQ2_XS, celles d'Unsloth) n'ont pas été essayées sur Mac.

## Utilisation

- **Dans le navigateur :** `http://127.0.0.1:8080` propose **Chat**, un **Monitor** en direct du modèle et du Mac
  (charge du GPU, mémoire, puissance, température) et **About**. L'effort de réflexion (**Thinking**, de *Off* à *High*)
  règle la durée de réflexion ; un **Thinking budget** dans les réglages **Sampling** du chat la plafonne en tokens.
- **Vos applications et agents de code :** un fournisseur « compatible OpenAI » avec l'URL de base
  `http://127.0.0.1:8080/v1` (n'importe quel nom de modèle ; n'importe quelle clé d'API tant que vous n'en définissez
  pas), l'API Anthropic sur `http://127.0.0.1:8080/v1/messages` (Claude Code : `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`)
  et l'API Responses sur `/v1/responses`. Les outils et MCP fonctionnent. Exemples de clients :
  [docs/DETAILS.md](docs/DETAILS.md#using-it).
- **Images :** répondez oui aux images à l'installation (modèles Qwen), puis joignez-les dans le chat ou votre application.
- **Une requête à la fois** par défaut ; les autres attendent. Un message de suivi dans la même conversation ne lit que
  le nouveau texte et démarre donc vite ; un long nouveau prompt met un moment à être lu avant le début de la réponse.
- **Depuis un autre appareil du réseau :** dans `strata-<modèle>.json`, définissez `"host": "0.0.0.0"` et une
  `"api_key"`, relancez Strata, puis utilisez `http://<IP de votre Mac>:8080` avec cette clé. Le trafic n'est pas
  chiffré : restez sur un réseau de confiance (ou un VPN).

## Un problème ?

- **C'est lent.** Utilisez le mode d'énergie *Haute puissance* si votre Mac le propose (portables : *Réglages Système >
  Batterie > Mode d'énergie* ; ordinateurs de bureau : *Réglages Système > Énergie*) et gardez un portable branché. Pour
  voir la fréquence du GPU : `brew install macmon`, puis `macmon`.
- **Il réfléchit très longtemps.** Baissez l'effort de réflexion ou fixez un **Thinking budget** (4096-8192) : une fois
  le budget épuisé, le serveur clôt la réflexion et le modèle écrit sa réponse, si max tokens laisse de la place.
- **Le port est déjà utilisé.** Un autre programme écoute sur 8080. Si c'est un autre Strata, `make stop` l'arrête, ou
  lancer un `run-<modèle>.sh` propose de l'arrêter ; sinon, prenez un autre port (`make run PORT=8090`). Deux modèles à
  la fois demandent la mémoire des deux.
- **Le modèle ne tient pas.** Fermez les grosses applications, choisissez un modèle ou un contexte plus petit
  (`make run CONTEXT=32768`).

Plus : [docs/MACOS.md](docs/MACOS.md#limits-and-warnings) (en anglais). Problèmes avec ce fork pour Mac :
[issues](https://github.com/dennis-akimov/Strata/issues). Un problème de sécurité : signalez-le en privé, voir
[SECURITY.md](SECURITY.md).

## Comment ça marche ?

Sur Mac, le moteur propre de Strata (`strata-metal`, dans [`metal/`](metal/)) fait tourner le modèle sur le GPU via le
backend Metal de [llama.cpp](https://github.com/ggml-org/llama.cpp), et le serveur Strata se place au-dessus :
l'application web, les API OpenAI / Anthropic / Responses / MCP, la réutilisation des conversations, la couche de
brouillon du modèle qui devine quelques tokens à l'avance, et les formats de discussion de Qwen, GPT-OSS (harmony) et
GLM. Le CPU et le GPU partagent la mémoire du Mac, donc les poids du modèle n'y sont qu'une fois, pour les deux.
Conception et mesures : [docs/MACOS.md](docs/MACOS.md#how-it-fits-together) (en anglais). Les documents de l'original
sont conservés dans [`docs/`](docs/README.md), avec ceux qui s'appliquent sur Mac.

## Crédits et licence

**[Strata](https://github.com/Niko1221/Strata) est l'œuvre de Niko1221 et des contributeurs de Strata**
([tous les crédits](docs/HOW_IT_WORKS.md#credits)) ; ce fork ajoute la prise en charge du Mac. Les modèles sont de
l'équipe Qwen ([Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), compressé par
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)), d'OpenAI
([GPT-OSS](https://huggingface.co/openai/gpt-oss-120b)) et de Z.ai (GLM-5.3-Flash, compressé par
[Project Maya](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)). Strata utilise des parties de
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Open source sous [licence MIT](LICENSE) (Copyright (c) 2026
Niko1221 and the Strata contributors) ; quelques parties et chaque modèle ont leur propre licence
([lesquelles](docs/HOW_IT_WORKS.md#license)).

Si ce fork pour Mac vous est utile, vous pouvez soutenir son mainteneur :
[buymeacoffee.com/dennisakimov](https://buymeacoffee.com/dennisakimov).
