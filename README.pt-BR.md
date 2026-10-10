<h1 align="center">Strata para Mac</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · **Português**

<p align="center"><b>Rode modelos de IA de 100 bilhões de parâmetros no seu próprio Mac</b><br>
Apple Silicon · testado em um M5 Max com 128 GB · gratuito e de código aberto</p>

> **Este é um fork do [Strata](https://github.com/Niko1221/Strata) mantido de forma independente, só para Macs com
> Apple Silicon**, por [dennis-akimov](https://github.com/dennis-akimov). Ele adiciona um motor Metal e a instalação no
> Mac. Os motores de Windows e Linux do original (NVIDIA CUDA, AMD HIP, Intel SYCL) continuam neste repositório como
> estavam no momento do fork, mas não são compilados, testados nem atualizados aqui.
> **Para um PC com Windows ou Linux, use o original: [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata).**

O Strata roda grandes modelos de IA no seu próprio computador: eles conversam, escrevem código, leem imagens e
trabalham com seus aplicativos e agentes de programação pelas mesmas APIs dos serviços na nuvem. O modelo roda no seu
Mac; o que sai dele depende de você (os downloads de modelos e cada aplicativo, ferramenta ou servidor MCP que você
conectar).

## Qual a velocidade?

Medido em um MacBook Pro com M5 Max (GPU de 40 núcleos, 128 GB), macOS 26.4, no modo de energia *Alto desempenho*,
2026-10-08, sem raciocínio. Um token é cerca de ¾ de uma palavra.

| Modelo | Respostas de 900 palavras | Respostas curtas | Arquivo do modelo na memória |
| --- | ---: | ---: | ---: |
| **[GPT-OSS 120B](docs/MACOS.md#gpt-oss-120b)** (MXFP4) | 93-100 tokens/s | 93-103 tokens/s | 63 GB |
| **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) Q2_0**, com `--mtp on` | 60-69 tokens/s | 92-102 tokens/s | 38 GB |
| **Qwen3.8-Flash-Next IQ3_XXS**, com `--mtp on` | 46-48 tokens/s | 73-92 tokens/s | 47 GB |

Os números do Qwen usam a camada de rascunho MTP dele, que é opcional (`./setup.sh --setup --mtp on`, cerca de 12 GB a
mais de disco). O cache do contexto se soma ao arquivo do modelo (alguns GB; mais em contextos longos). No modo de
energia *Automático*, o mesmo Mac escrevia respostas longas de 2 a 3 vezes mais devagar. Outros Macs não foram
medidos. Todos os números e como foram medidos: [docs/MACOS.md](docs/MACOS.md#measured) (em inglês).

## O que você precisa

| | |
| --- | --- |
| **Mac** | Apple Silicon (M1 ou mais novo). Só um M5 Max com 128 GB foi testado; outros chips e tamanhos devem funcionar, mas não foram testados. |
| **Memória** | O suficiente para o arquivo do modelo e seu cache, dentro do que o macOS deixa a GPU usar (no Mac testado de 128 GB: 107,5 GB). A verificação da instalação considera 64 GB o mínimo. `make check` estima o que cabe no seu Mac; veja a tabela de modelos abaixo. |
| **Disco** | O download do modelo mais alguns GB (70-100 GB por modelo), no SSD interno. |
| **Software** | As Command Line Tools da Apple (`xcode-select --install`, espere terminar) e Python 3.10 ou mais novo (se faltar, a instalação o adiciona com o Homebrew quando o Homebrew está instalado). Testado no macOS 26.4. |

## Instalação

No Terminal:

```sh
git clone https://github.com/dennis-akimov/Strata.git
cd Strata
make check                 # o que este Mac consegue rodar; não instala nada
make pull MODEL=Q2_0       # compila o motor e baixa o modelo (66 GB)
make run                   # carrega o modelo e abre http://127.0.0.1:8080 quando estiver pronto
```

`make pull` faz algumas perguntas (tamanho do contexto, imagens); Enter escolhe a resposta recomendada. Se o download
parar, rode de novo: ele continua de onde parou. O carregamento leva cerca de um minuto a cada início. `make run` ocupa
o Terminal enquanto o Strata roda; Ctrl+C o encerra, ou `make start` / `make stop` o roda em segundo plano. Todas as
opções, limites e avisos: **[docs/MACOS.md](docs/MACOS.md)** (em inglês).

**Atualizar:** `git pull`, depois `make run` (a instalação recompila o motor se as fontes mudaram; o modelo fica).
**Onde ficam as coisas:** os modelos em `Strata-data/` ao lado da pasta `Strata`, as configurações em
`strata-<modelo>.json`, as conversas no seu navegador.

## Qual modelo?

| Modelo | Download | Na memória, contexto de 128K | Observações |
| --- | ---: | ---: | --- |
| **Qwen3.8-Flash-Next Q2_0** | 66 GB | cerca de 60 GB (estimativa) | O padrão e o mais testado no Mac. `make pull MODEL=Q2_0` |
| Qwen3.8-Flash-Next IQ3_XXS | 76 GB | 65 GB (medido) | Menos comprimido que o Q2_0; 10-30 % mais lento nos nossos testes. `make pull MODEL=IQ3_XXS` |
| Qwen3.8-Flash-Next IQ3_S | 84 GB | 74 GB (medido) | Ainda menos comprimido; cerca de 10 % mais lento que o Q2_0 em um teste A/B. `make pull MODEL=IQ3_S` |
| **GPT-OSS 120B** | 63 GB | cerca de 70 GB (estimativa) | O mais rápido aqui (cerca de 5 bilhões de parâmetros ativos por token). Instalação manual: [passos](docs/MACOS.md#gpt-oss-120b) |
| GLM-5.3-Flash (Maya-S24) | 95 GB | cerca de 100 GB com 32K (estimativa) | Só código: o Strata lê o tokenizador, o formato de conversa e as chamadas de ferramentas, mas o modelo ainda não foi executado. |

"Na memória" é o motor com a camada de rascunho MTP dos modelos Qwen e 2 vagas de requisição, no Mac testado; um
contexto menor precisa de menos (`make run CONTEXT=32768`). Isso precisa caber no que o macOS deixa a GPU usar, que o
`make check` mostra para o seu Mac (107,5 GB no Mac testado de 128 GB; menos em Macs menores). Os modelos Qwen também
leem uma tabela de 28 GB do SSD enquanto rodam. Os outros tamanhos do Qwen (IQ2_XS, os da Unsloth) não foram testados
no Mac.

## Como usar

- **No navegador:** `http://127.0.0.1:8080` tem **Chat**, um **Monitor** ao vivo do modelo e do Mac (carga da GPU,
  memória, potência, temperatura) e **About**. O esforço de raciocínio (**Thinking**, de *Off* a *High*) define quanto
  o modelo pensa; um **Thinking budget** nas configurações **Sampling** do chat limita isso em tokens.
- **Seus aplicativos e agentes de programação:** um provedor "compatível com OpenAI" com a URL base
  `http://127.0.0.1:8080/v1` (qualquer nome de modelo; qualquer chave de API enquanto você não definir uma), a API da
  Anthropic em `http://127.0.0.1:8080/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`) e a API
  Responses em `/v1/responses`. Ferramentas e MCP funcionam. Exemplos de clientes:
  [docs/DETAILS.md](docs/DETAILS.md#using-it).
- **Imagens:** responda sim às imagens na instalação (modelos Qwen) e depois anexe-as no chat ou no seu aplicativo.
- **Uma requisição por vez** por padrão; as outras esperam. Uma mensagem de continuação na mesma conversa lê só o que é
  novo, então começa rápido; um prompt novo e longo leva um tempo para ser lido antes de a resposta começar.
- **De outro dispositivo na rede:** em `strata-<modelo>.json`, defina `"host": "0.0.0.0"` e uma `"api_key"`, reinicie
  o Strata e use `http://<IP do seu Mac>:8080` com essa chave. O tráfego não é criptografado: use em uma rede confiável
  (ou uma VPN).

## Algo deu errado?

- **Está lento.** Use o modo de energia *Alto desempenho* se o seu Mac tiver (notebooks: *Ajustes do Sistema >
  Bateria > Modo de Energia*; desktops: *Ajustes do Sistema > Energia*) e mantenha o notebook na tomada. Para ver a frequência
  da GPU: `brew install macmon`, depois `macmon`.
- **Ele pensa por muito tempo.** Diminua o esforço de raciocínio ou defina um **Thinking budget** (4096-8192): quando
  ele acaba, o servidor encerra o raciocínio e o modelo escreve a resposta, se o max tokens deixar espaço.
- **A porta está em uso.** Outro programa escuta na 8080. Se for outro Strata, `make stop` o encerra, ou iniciar um
  `run-<modelo>.sh` oferece encerrá-lo; caso contrário, use outra porta (`make run PORT=8090`). Dois modelos ao mesmo
  tempo precisam de memória para os dois.
- **O modelo não cabe.** Feche aplicativos grandes, escolha um modelo ou um contexto menor (`make run CONTEXT=32768`).

Mais: [docs/MACOS.md](docs/MACOS.md#limits-and-warnings) (em inglês). Problemas com este fork para Mac:
[issues](https://github.com/dennis-akimov/Strata/issues). Um problema de segurança: informe em particular, veja
[SECURITY.md](SECURITY.md).

## Como funciona?

No Mac, o motor próprio do Strata (`strata-metal`, em [`metal/`](metal/)) roda o modelo na GPU pelo backend Metal do
[llama.cpp](https://github.com/ggml-org/llama.cpp), e o servidor do Strata fica por cima: o aplicativo web, as APIs
OpenAI / Anthropic / Responses / MCP, o reaproveitamento de conversas, a camada de rascunho do modelo que adivinha
alguns tokens à frente e os formatos de conversa do Qwen, do GPT-OSS (harmony) e do GLM. A CPU e a GPU compartilham a
memória do Mac, então os pesos do modelo ficam nela uma vez só, para as duas. Projeto e medições:
[docs/MACOS.md](docs/MACOS.md#how-it-fits-together) (em inglês). Os documentos do original são mantidos em
[`docs/`](docs/README.md), indicando quais valem no Mac.

## Créditos e licença

**O [Strata](https://github.com/Niko1221/Strata) é obra de Niko1221 e dos colaboradores do Strata**
([todos os créditos](docs/HOW_IT_WORKS.md#credits)); este fork adiciona o suporte ao Mac. Os modelos são da equipe do
Qwen ([Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), comprimido pela
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)), da OpenAI
([GPT-OSS](https://huggingface.co/openai/gpt-oss-120b)) e da Z.ai (GLM-5.3-Flash, comprimido pelo
[Project Maya](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)). O Strata usa partes do
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Código aberto sob a [licença MIT](LICENSE) (Copyright (c)
2026 Niko1221 and the Strata contributors); algumas partes e cada modelo têm suas próprias licenças
([quais](docs/HOW_IT_WORKS.md#license)).

Se este fork para Mac for útil para você, você pode apoiar o mantenedor:
[buymeacoffee.com/dennisakimov](https://buymeacoffee.com/dennisakimov).
