<h1 align="center">Strata für den Mac</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **Deutsch** · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>KI-Modelle mit 100 Milliarden Parametern auf dem eigenen Mac</b><br>
Apple Silicon · getestet auf einem M5 Max mit 128 GB · kostenlos und Open Source</p>

> **Dies ist ein unabhängig gepflegter Fork von [Strata](https://github.com/Niko1221/Strata), nur für Macs mit Apple
> Silicon**, gepflegt von [dennis-akimov](https://github.com/dennis-akimov). Er ergänzt eine Metal-Engine und die
> Einrichtung für den Mac. Die Windows- und Linux-Engines des Originals (NVIDIA CUDA, AMD HIP, Intel SYCL) liegen
> weiterhin in diesem Repository, so wie sie beim Abzweigen waren, werden hier aber weder gebaut, getestet noch
> aktualisiert. **Für einen Windows- oder Linux-PC nimm das Original: [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata).**

Strata führt große KI-Modelle auf deinem eigenen Computer aus: Sie chatten, schreiben Code, lesen Bilder und arbeiten
über dieselben APIs wie Cloud-Dienste mit deinen Apps und Coding-Agenten zusammen. Das Modell läuft auf deinem Mac;
was ihn verlässt, bestimmst du (die Modell-Downloads und jede App, jedes Tool und jeder MCP-Server, den du anbindest).

Es erzeugt auch Bilder aus einem Text-Prompt, mit einer eigenen Bild-Engine ([Nucleus-Image](docs/MACOS.md#pictures-from-a-prompt-nucleus-image) oder [HunyuanImage-3.0](docs/MACOS.md#pictures-from-a-prompt-hunyuanimage-30)). Und kurze Videos aus einem Prompt ([Wan 2.2](docs/MACOS.md#videos-from-a-prompt-wan-22-a14b)).

## Wie schnell ist es?

Gemessen auf einem MacBook Pro mit M5 Max (40-Kern-GPU, 128 GB), macOS 26.4, im Energiemodus *Hohe Leistung*,
2026-10-08, ohne Nachdenken. Ein Token ist etwa ¾ eines Wortes.

| Modell | Antworten mit 900 Wörtern | Kurze Antworten | Modelldatei im Speicher |
| --- | ---: | ---: | ---: |
| **[GPT-OSS 120B](docs/MACOS.md#gpt-oss-120b)** (MXFP4) | 93-100 Token/s | 93-103 Token/s | 63 GB |
| **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) Q2_0**, mit `--mtp on` | 60-69 Token/s | 92-102 Token/s | 38 GB |
| **Qwen3.8-Flash-Next IQ3_XXS**, mit `--mtp on` | 46-48 Token/s | 73-92 Token/s | 47 GB |
| **[GLM-5.3-Flash](docs/MACOS.md#glm-53-flash) Maya-S24**\* | 13-26 Token/s | 29-32 Token/s | 86 GB |

\* Gemessen am 2026-10-10, während Docker etwa 40 GB belegte und der Swap voll war (GPU mit niedrigem Takt), also eine Untergrenze; GLM liest außerdem etwa 3x so viele aktive Parameter pro Token wie die anderen.

Bilder und Videos auf demselben Mac (2026-10-11/12):

| Modell | Ausgabe | Zeit | Speicher |
| --- | --- | ---: | ---: |
| **[HunyuanImage-3.0](docs/MACOS.md#pictures-from-a-prompt-hunyuanimage-30)** (80B, Q4_0-Experten) | Bild 1024x1024, 8 Schritte | 88 s | 0,3 GB, die 54-GB-Datei eingeblendet |
| **[Nucleus-Image](docs/MACOS.md#pictures-from-a-prompt-nucleus-image)** (17B, Q4_0-Experten) | Bild 1024x1024, 20 Schritte | 45 s | 2 GB, 12 GB Dateien eingeblendet |
| **[Wan 2.2 A14B](docs/MACOS.md#videos-from-a-prompt-wan-22-a14b)** + Lightning-LoRAs | Video 832x480, 2 s (33 Frames), 4 Schritte | 430 s | 30 GB in der Spitze |
| Wan 2.2 A14B ohne die LoRAs | dasselbe Video, 18 Schritte | 2907 s | etwa 28 GB |

Die Videozeiten wurden gemessen, während Docker 76 GB belegte und macOS auslagerte (GPU bei 600-750 MHz): ein Mindestwert.

Die Qwen-Werte nutzen seine MTP-Entwurfsschicht, die optional ist (`./setup.sh --setup --mtp on`, etwa 12 GB mehr
Speicherplatz). Der Cache des Kontexts kommt zur Modelldatei hinzu (einige GB; mehr bei langen Kontexten). Im
Energiemodus *Automatisch* schrieb derselbe Mac lange Antworten 2-3x langsamer. Andere Macs wurden nicht gemessen.
Alle Werte und wie sie gemessen wurden: [docs/MACOS.md](docs/MACOS.md#measured) (Englisch).

## Was du brauchst

| | |
| --- | --- |
| **Mac** | Apple Silicon (M1 oder neuer). Getestet wurde nur ein M5 Max mit 128 GB; andere Chips und Größen sollten funktionieren, wurden aber nicht ausprobiert. |
| **Arbeitsspeicher** | Genug für die Modelldatei und ihren Cache innerhalb dessen, was macOS der GPU erlaubt (auf dem getesteten 128-GB-Mac: 107,5 GB). Die Prüfung von Setup setzt 64 GB als Minimum an. `make check` schätzt, was auf deinen Mac passt; siehe die Modelltabelle unten. |
| **Speicherplatz** | Der Download des Modells plus einige GB (70-100 GB pro Modell), auf der internen SSD. |
| **Software** | Apples Command Line Tools (`xcode-select --install`, warten bis es fertig ist) und Python 3.10 oder neuer (fehlt es, installiert Setup es mit Homebrew, wenn Homebrew vorhanden ist). Getestet unter macOS 26.4. |

## Installation

Im Terminal:

```sh
git clone https://github.com/dennis-akimov/Strata.git
cd Strata
make check                 # was dieser Mac ausführen kann; installiert nichts
make pull MODEL=Q2_0       # baut die Engine und lädt das Modell herunter (66 GB)
make run                   # lädt das Modell und öffnet http://127.0.0.1:8080, sobald es bereit ist
```

`make pull` stellt ein paar Fragen (Kontextgröße, Bilder); Enter übernimmt die empfohlene Antwort. Bricht der Download
ab, starte ihn erneut: Er macht dort weiter, wo er aufgehört hat. Das Laden dauert bei jedem Start etwa eine Minute.
`make run` belegt das Terminal, solange Strata läuft; Strg+C beendet es, oder `make start` / `make stop` lässt es im
Hintergrund laufen. Alle Optionen, Grenzen und Warnungen: **[docs/MACOS.md](docs/MACOS.md)** (Englisch).

**Aktualisieren:** `git pull`, dann `make run` (Setup baut die Engine neu, wenn sich die Quellen geändert haben; das
Modell bleibt). **Wo was liegt:** Modelle in `Strata-data/` neben dem Ordner `Strata`, Einstellungen in
`strata-<modell>.json`, Chats in deinem Browser.

## Welches Modell?

| Modell | Download | Im Speicher, 128K Kontext | Hinweise |
| --- | ---: | ---: | --- |
| **Qwen3.8-Flash-Next Q2_0** | 66 GB | etwa 60 GB (Schätzung) | Der Standard und am meisten auf dem Mac getestet. `make pull MODEL=Q2_0` |
| Qwen3.8-Flash-Next IQ3_XXS | 76 GB | 65 GB (gemessen) | Weniger stark komprimiert als Q2_0; in unseren Läufen 10-30 % langsamer. `make pull MODEL=IQ3_XXS` |
| Qwen3.8-Flash-Next IQ3_S | 84 GB | 74 GB (gemessen) | Noch weniger komprimiert; in einem A/B-Test etwa 10 % langsamer als Q2_0. `make pull MODEL=IQ3_S` |
| **GPT-OSS 120B** | 63 GB | etwa 70 GB (Schätzung) | Hier das schnellste (etwa 5 Mrd. aktive Parameter pro Token). Installieren: `make pull MODEL=GPT-OSS-120B`, dann [Schritte](docs/MACOS.md#gpt-oss-120b) |
| GLM-5.3-Flash (Maya-S24) | 95 GB | 89 GB bei 32K (gemessen) | Das größte hier (320 Mrd. Parameter, etwa 18 Mrd. aktiv pro Token) und das langsamste. Chat, Denken und Tool-Aufrufe funktionieren. Installieren: `make pull MODEL=GLM-5.3`, dann [Schritte](docs/MACOS.md#glm-53-flash) |
| HunyuanImage-3.0 (Bilder) | 169 GB, gestreamt (54 GB bleiben) | 54 GB eingeblendet + 0,3 GB | 80B; etwa 90 s pro Bild. Die Lizenz gilt nicht in der EU, im Vereinigten Königreich und in Südkorea. `make pull MODEL=HUNYUAN` (etwa 2 Stunden) |
| Nucleus-Image (Bilder) | etwa 50 GB (21 GB bleiben) | 12 GB eingeblendet + 2 GB | 17B, Apache-2.0; etwa 45 s pro Bild. `make pull MODEL=NUCLEUS` |
| Wan 2.2 T2V A14B (Videos) | 29 GB | 30 GB in der Spitze | Zwei 14B-Experten + die Lightning-LoRAs; ein 2-s-Video in etwa 7 Minuten. `make pull MODEL=WAN2.2` |

„Im Speicher“ ist die Engine mit der MTP-Entwurfsschicht der Qwen-Modelle und 2 Anfrage-Slots, auf dem getesteten
Mac; ein kleinerer Kontext braucht weniger (`make run CONTEXT=32768`). Es muss in das passen, was macOS der GPU
erlaubt; `make check` zeigt das für deinen Mac an (107,5 GB auf dem getesteten 128-GB-Mac; weniger auf kleineren
Macs). Die Qwen-Modelle lesen außerdem während des Laufs eine 28-GB-Tabelle von der SSD. Die übrigen Qwen-Größen
(IQ2_XS, die von Unsloth) wurden auf dem Mac nicht ausprobiert.

## Benutzung

- **Im Browser:** `http://127.0.0.1:8080` hat **Chat**, einen Live-**Monitor** von Modell und Mac (GPU-Last,
  Speicher, Leistung, Temperatur) und **About**. Der Denkaufwand (**Thinking**, *Off* bis *High*) bestimmt, wie lange das
  Modell nachdenkt; ein **Thinking budget** in den **Sampling**-Einstellungen des Chats begrenzt das in Token.
- **Deine Apps und Coding-Agenten:** ein „OpenAI-kompatibler“ Anbieter mit der Basis-URL `http://127.0.0.1:8080/v1`
  (beliebiger Modellname; beliebiger API-Schlüssel, solange du keinen festlegst), die Anthropic-API unter
  `http://127.0.0.1:8080/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`) und die Responses-API
  unter `/v1/responses`. Tools und MCP funktionieren. Beispiele für Clients: [docs/DETAILS.md](docs/DETAILS.md#using-it).
- **Bilder:** Im Setup Bilder bejahen (Qwen-Modelle), dann im Chat oder in deiner App anhängen.
- **Bilder erzeugen:** `/image <Prompt>` im Chat oder `POST /v1/images/generations` (die Bilder-API von OpenAI), mit
  [Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) (17B) auf Stratas eigener Bild-Engine. Ein Bild mit
  1024x1024 dauert etwa 45-70 s (20-30 Schritte) und braucht einige GB Speicher neben den 20 GB Modelldateien, die macOS
  von der SSD einblendet. Installieren: `make pull MODEL=NUCLEUS`, dann [Schritte](docs/MACOS.md#pictures-from-a-prompt-nucleus-image).
  Oder [HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil) (Tencent, 80B-Mixture-of-Experts; ein Modell für Prompt und Bild, also kein
  Text-Encoder): 8 Schritte, etwa 90 s bei 1024x1024, 0,3 GB privater Speicher neben der 54-GB-Datei, die macOS von der
  SSD einblendet. Installieren: `make pull MODEL=HUNYUAN` (etwa 2 Stunden), dann [Schritte](docs/MACOS.md#pictures-from-a-prompt-hunyuanimage-30). Seine Lizenz gilt
  nicht in der EU, im Vereinigten Königreich und in Südkorea.
- **Videos erzeugen:** `/video <Prompt>` im Chat oder `POST /v1/videos` (OpenAIs Video-API: ein Job, den man
  abfragt), mit [Wan 2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) (zwei 14B-Experten) auf dem Metal-Backend von [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
  und [den Lightning-LoRAs von lightx2v](https://huggingface.co/lightx2v/Wan2.2-Lightning) (4 statt 18 Schritte, ein Durchlauf pro Schritt). Ein 2-s-Video mit
  832x480 dauert etwa 7 Minuten (49 ohne die LoRAs) und braucht in der Spitze 30 GB Speicher. Installieren:
  `make pull MODEL=WAN2.2` (29 GB, mit den LoRAs), dann [Schritte](docs/MACOS.md#videos-from-a-prompt-wan-22-a14b).
- **Eine Anfrage nach der anderen** als Standard; weitere warten. Eine Folgenachricht im selben Gespräch liest nur das
  Neue und beginnt daher schnell; ein langer neuer Prompt braucht eine Weile, bis die Antwort beginnt.
- **Von einem anderen Gerät im Netzwerk:** in `strata-<modell>.json` `"host": "0.0.0.0"` und einen `"api_key"`
  setzen, Strata neu starten, dann `http://<IP deines Macs>:8080` mit diesem Schlüssel nutzen. Der Verkehr ist nicht
  verschlüsselt: nur in einem Netzwerk, dem du vertraust (oder über VPN).

## Etwas funktioniert nicht?

- **Es ist langsam.** Nutze den Energiemodus *Hohe Leistung*, wo dein Mac ihn hat (Laptops: *Systemeinstellungen >
  Batterie > Energiemodus*; Desktops: *Systemeinstellungen > Energie*), und lass ein Laptop am Netzteil. Den GPU-Takt
  siehst du mit `brew install macmon`, dann `macmon`.
- **Es denkt sehr lange nach.** Wähle einen geringeren Denkaufwand oder ein **Thinking budget** (4096-8192): Ist das Budget
  aufgebraucht, schließt der Server das Nachdenken und das Modell schreibt seine Antwort, sofern max tokens Platz dafür
  lässt.
- **Der Port ist belegt.** Ein anderes Programm lauscht auf 8080. Ist es ein anderes Strata, beendet `make stop` es,
  oder ein `run-<modell>.sh` bietet an, es zu beenden; sonst einen anderen Port nutzen (`make run PORT=8090`). Zwei
  Modelle gleichzeitig brauchen den Speicher für beide.
- **Das Modell passt nicht.** Große Apps schließen, ein kleineres Modell oder einen kleineren Kontext wählen
  (`make run CONTEXT=32768`).

Mehr: [docs/MACOS.md](docs/MACOS.md#limits-and-warnings) (Englisch). Probleme mit diesem Mac-Fork:
[Issues](https://github.com/dennis-akimov/Strata/issues). Ein Sicherheitsproblem: bitte vertraulich melden, siehe
[SECURITY.md](SECURITY.md).

## Wie funktioniert es?

Auf dem Mac führt Stratas eigene Engine (`strata-metal`, in [`metal/`](metal/)) das Modell über das Metal-Backend von
[llama.cpp](https://github.com/ggml-org/llama.cpp) auf der GPU aus, und der Strata-Server sitzt darüber: die Web-App,
die OpenAI-/Anthropic-/Responses-/MCP-APIs, die Wiederverwendung von Gesprächen, die Entwurfsschicht des Modells, die
einige Token vorausrät, und die Chatformate von Qwen, GPT-OSS (harmony) und GLM. CPU und GPU teilen sich den
Arbeitsspeicher des Macs, daher liegen die Gewichte des Modells einmal darin, für beide. Aufbau und Messungen:
[docs/MACOS.md](docs/MACOS.md#how-it-fits-together) (Englisch). Die Dokumente des Originals sind in
[`docs/`](docs/README.md) erhalten, mit dem Hinweis, welche auf dem Mac gelten.

Bilder aus einem Prompt erzeugt eine zweite Engine, `strata-image` (in [`metal/image/`](metal/image/)): der Text-Encoder
von Nucleus-Image, sein 17B-Mixture-of-Experts-Bildmodell und sein Decoder auf demselben Metal-Backend, ein Prozess je
Bild. Auch HunyuanImage-3.0 läuft dort: die Keys und Values des Prompts einmal, dann 8 Schritte der Bild-Tokens
durch den 80B-Transformer und sein 3D-VAE.

Videos laufen in `sd-cli` von stable-diffusion.cpp, gebaut mit einem Strata-Patch, der die 3D-Faltungen des Wan-VAE
auf Metal 13-mal schneller macht; der Server führt jeden Clip als Job aus.

## Danksagung und Lizenz

**[Strata](https://github.com/Niko1221/Strata) ist das Werk von Niko1221 und den Strata-Mitwirkenden**
([alle Danksagungen](docs/HOW_IT_WORKS.md#credits)); dieser Fork ergänzt die Mac-Unterstützung. Die Modelle stammen
vom Qwen-Team ([Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), komprimiert von
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)), von OpenAI
([GPT-OSS](https://huggingface.co/openai/gpt-oss-120b)) und von Z.ai (GLM-5.3-Flash, komprimiert von
[Project Maya](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)). Strata nutzt Teile von
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Open Source unter der [MIT-Lizenz](LICENSE) (Copyright (c)
2026 Niko1221 and the Strata contributors); einige Teile und jedes Modell haben eigene Lizenzen
([welche](docs/HOW_IT_WORKS.md#license)).

Bilder: [Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) von NucleusAI (Apache-2.0), konvertiert aus
der Portierung [mlx-nucleus-image](https://huggingface.co/treadon/mlx-nucleus-image).
[HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil) von Tencent ([Tencent Hunyuan Community License](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE): in der EU, im Vereinigten Königreich und in Südkorea nicht lizenziert).
Videos: [Wan 2.2](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) vom Wan-Team von Alibaba (Apache-2.0), ausgeführt mit [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (MIT).

Wenn dir dieser Mac-Fork nützt, kannst du seinen Maintainer unterstützen:
[buymeacoffee.com/dennisakimov](https://buymeacoffee.com/dennisakimov).
