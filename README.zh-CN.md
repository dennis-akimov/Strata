<h1 align="center">Strata for Mac</h1>

[English](README.md) · **简体中文** · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>在你自己的 Mac 上运行千亿参数级的 AI 模型</b><br>
Apple Silicon · 已在 128 GB 的 M5 Max 上测试 · 免费开源</p>

> **这是 [Strata](https://github.com/Niko1221/Strata) 的一个独立维护的分支（fork），只面向 Apple Silicon Mac**，
> 维护者为 [dennis-akimov](https://github.com/dennis-akimov)。它增加了 Metal 引擎和 Mac 上的安装流程。原项目的
> Windows 和 Linux 引擎（NVIDIA CUDA、AMD HIP、Intel SYCL）仍保留在本仓库中，保持分叉时的状态，但在这里不会编译、
> 测试或更新。**如果你用的是 Windows 或 Linux PC，请使用原项目：[github.com/Niko1221/Strata](https://github.com/Niko1221/Strata)。**

Strata 在你自己的电脑上运行大型 AI 模型：它们能聊天、写代码、看图片，并通过与云服务相同的 API 与你的应用和编程
助手协作。模型在你的 Mac 上运行；有哪些数据离开 Mac 由你决定（模型下载，以及你接入的应用、工具或 MCP 服务器）。

它还能根据文字提示生成图片，使用自己的图像引擎（[Nucleus-Image](docs/MACOS.md#pictures-from-a-prompt-nucleus-image) 或 [HunyuanImage-3.0](docs/MACOS.md#pictures-from-a-prompt-hunyuanimage-30)）。还能根据提示生成短视频（[Wan 2.2](docs/MACOS.md#videos-from-a-prompt-wan-22-a14b)）。

## 速度如何？

在一台 M5 Max（40 核 GPU，128 GB）的 MacBook Pro 上测得，macOS 26.4，能耗模式为 *高功率*，2026-10-08，关闭思考。
一个 token 约相当于 ¾ 个英文单词。

| 模型 | 900 词的回答 | 简短回答 | 内存中的模型文件 |
| --- | ---: | ---: | ---: |
| **[GPT-OSS 120B](docs/MACOS.md#gpt-oss-120b)**（MXFP4） | 93-100 token/秒 | 93-103 token/秒 | 63 GB |
| **[GPT-OSS 120B 加 DFlash 草稿模型](docs/MACOS.md#gpt-oss-120b-with-a-dflash-draft)**‡ | 数学 122-135 token/秒，聊天 71-77 | - | 65 GB |
| **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) Q2_0**，`--mtp on` | 60-69 token/秒 | 92-102 token/秒 | 38 GB |
| **Qwen3.8-Flash-Next IQ3_XXS**，`--mtp on` | 46-48 token/秒 | 73-92 token/秒 | 47 GB |
| **[GLM-5.3-Flash](docs/MACOS.md#glm-53-flash) Maya-S24**\* | 13-26 token/秒 | 29-32 token/秒 | 86 GB |
| **[GPT-OSS 20B](docs/MACOS.md#gpt-oss-20b-and-qwen36-35b-a3b)** (MXFP4)† | 128-139 token/秒 | - | 12 GB |
| **[Qwen3.6-35B-A3B](docs/MACOS.md#gpt-oss-20b-and-qwen36-35b-a3b)** (MXFP4_MOE)† | 89-97 token/秒 | - | 22 GB |

\* 于 2026-10-10 测得，当时 Docker 占用约 40 GB、交换空间已满（GPU 频率偏低），因此是下限；此外 GLM 每个 token 读取的激活参数约为其他模型的 3 倍。

† 于 2026-10-12 以 600 词（800 token）的回答测得，此前的运行使 GPU 降频到 1.2-1.5 GHz，因此是下限。

‡ 于 2026-10-13 以 300 token 的回答测得，同一批运行中不用草稿为 91-99 token/秒：草稿对数学有帮助，对聊天没有。

同一台 Mac 上的图片和视频（2026-10-11/12）：

| 模型 | 输出 | 时间 | 内存 |
| --- | --- | ---: | ---: |
| **[HunyuanImage-3.0](docs/MACOS.md#pictures-from-a-prompt-hunyuanimage-30)**（80B，Q4_0 专家） | 1024x1024 图片，8 步 | 88 秒 | 0.3 GB，映射其 54 GB 文件 |
| **[Nucleus-Image](docs/MACOS.md#pictures-from-a-prompt-nucleus-image)**（17B，Q4_0 专家） | 1024x1024 图片，20 步 | 45 秒 | 2 GB，映射其 12 GB 文件 |
| **[Wan 2.2 A14B](docs/MACOS.md#videos-from-a-prompt-wan-22-a14b)** + Lightning LoRA | 832x480 视频，2 秒（33 帧），4 步 | 430 秒 | 峰值 30 GB |
| Wan 2.2 A14B（不用 LoRA） | 同一视频，18 步 | 2907 秒 | 约 28 GB |

视频时间是在 Docker 占用 76 GB、macOS 正在使用交换空间（GPU 频率 600-750 MHz）时测得的，因此是下限。

Qwen 的数字使用了它的 MTP 草稿层，这是可选功能（`python3 bootstrap.py --setup --mtp on`，多占约 12 GB 磁盘）。上下文缓存
另外占用内存（几 GB；上下文越长越多）。在 *自动* 能耗模式下，同一台 Mac 写长回答要慢 2-3 倍。没有在其他 Mac 上
测量过。全部数字及测量方法：[docs/MACOS.md](docs/MACOS.md#measured)（英文）。

## 你需要什么

| | |
| --- | --- |
| **Mac** | Apple Silicon（M1 或更新）。只测试过一台 128 GB 的 M5 Max；其他芯片和内存容量应该也能运行，但没有试过。 |
| **内存** | 足够容纳模型文件及其缓存，且在 macOS 允许 GPU 使用的范围内（测试用的 128 GB Mac 上为 107.5 GB）。安装检查以 64 GB 为最低要求。`make check` 会估计你的 Mac 能放下哪些模型；另见下面的模型表。 |
| **磁盘** | 模型下载大小再加几 GB（每个模型 70-100 GB），放在内置 SSD 上。 |
| **软件** | Apple 的 Command Line Tools（`xcode-select --install`，等它装完）以及 Python 3.10 或更新版本（如果没有，且装有 Homebrew，安装程序会用 Homebrew 安装）。在 macOS 26.4 上测试。 |

## 安装

在终端中：

```sh
git clone https://github.com/dennis-akimov/Strata.git
cd Strata
make check                 # 这台 Mac 能运行什么；不安装任何东西
make pull MODEL=Q2_0       # 编译引擎并下载模型（66 GB）
make run                   # 加载模型，就绪后打开 http://127.0.0.1:8080
```

`make pull` 会问几个问题（上下文大小、图片）；按回车即选推荐答案。如果下载中断，再运行一次：它会从中断处继续。
每次启动加载大约需要一分钟。`make run` 在 Strata 运行期间会占用终端；按 Ctrl+C 停止，或用 `make start` /
`make stop` 在后台运行。全部选项、限制和注意事项：**[docs/MACOS.md](docs/MACOS.md)**（英文）。

**更新：** `git pull`，然后 `make run`（源码有变化时，安装程序会重新编译引擎；模型保留不动）。**文件位置：** 模型在
`Strata` 文件夹旁边的 `Strata-data/` 中，设置在 `strata-<模型>.json` 中，聊天记录保存在浏览器里。

## 选哪个模型？

| 模型 | 下载大小 | 内存占用（128K 上下文） | 说明 |
| --- | ---: | ---: | --- |
| **Qwen3.8-Flash-Next Q2_0** | 66 GB | 约 60 GB（估计） | 默认选项，在 Mac 上测试最多。`make pull MODEL=Q2_0` |
| Qwen3.8-Flash-Next IQ3_XXS | 76 GB | 65 GB（实测） | 压缩程度低于 Q2_0；在我们的测试中慢 10-30%。`make pull MODEL=IQ3_XXS` |
| Qwen3.8-Flash-Next IQ3_S | 84 GB | 74 GB（实测） | 压缩程度更低；在一次 A/B 测试中比 Q2_0 慢约 10%。`make pull MODEL=IQ3_S` |
| **GPT-OSS 120B** | 63 GB | 约 70 GB（估计） | 每个 token 约激活 50 亿参数。安装：`make pull MODEL=GPT-OSS-120B`，然后按[步骤](docs/MACOS.md#gpt-oss-120b) |
| GPT-OSS 120B + DFlash 草稿 | 多 1.6 GB | 约 72 GB（估计） | 数学和分步回答更快（约 +35%），聊天更慢（约 -20%）。`make pull MODEL=GPT-OSS-120B-DFLASH`，然后按[步骤](docs/MACOS.md#gpt-oss-120b-with-a-dflash-draft) |
| GPT-OSS 20B | 12 GB | 19 GB（实测） | 这里最快的模型（每个 token 约激活 36 亿参数），能力不如 120B。`make pull MODEL=GPT-OSS-20B`，然后按[步骤](docs/MACOS.md#gpt-oss-20b-and-qwen36-35b-a3b) |
| Qwen3.6-35B-A3B | 22 GB | 29 GB（实测） | 每个 token 约激活 30 亿参数，但在这里并不比 GPT-OSS 120B 快。`make pull MODEL=QWEN3.6-35B`，然后按[步骤](docs/MACOS.md#gpt-oss-20b-and-qwen36-35b-a3b) |
| GLM-5.3-Flash（Maya-S24） | 95 GB | 32K 时 89 GB（实测） | 这里最大（3200 亿参数，每个 token 约激活 180 亿）也最慢。聊天、思考和工具调用都可用。安装：`make pull MODEL=GLM-5.3`，然后按[步骤](docs/MACOS.md#glm-53-flash) |
| HunyuanImage-3.0（图片） | 169 GB，流式转换（保留 54 GB） | 映射 54 GB + 0.3 GB | 80B；每张图片约 90 秒。其许可在欧盟、英国和韩国不适用。`make pull MODEL=HUNYUAN`（约 2 小时） |
| Nucleus-Image（图片） | 约 50 GB（保留 21 GB） | 映射 12 GB + 2 GB | 17B，Apache-2.0；每张图片约 45 秒。`make pull MODEL=NUCLEUS` |
| Wan 2.2 T2V A14B（视频） | 29 GB | 峰值 30 GB | 两个 14B 专家 + Lightning LoRA；2 秒视频约 7 分钟。`make pull MODEL=WAN2.2` |

“内存占用”是在测试用的 Mac 上，引擎加上 Qwen 模型的 MTP 草稿层和 2 个请求槽位时的数值；上下文越小占用越少
（`make run CONTEXT=32768`）。它必须放得进 macOS 允许 GPU 使用的范围，`make check` 会显示你的 Mac 的这个值（测试用的
128 GB Mac 上为 107.5 GB；内存更小的 Mac 更少）。Qwen 模型运行时还会从 SSD 读取一个 28 GB 的表。其他 Qwen 尺寸
（IQ2_XS、Unsloth 的版本）没有在 Mac 上试过。

## 使用

- **在浏览器中：** `http://127.0.0.1:8080` 有 **Chat**、模型和 Mac 的实时 **Monitor**（GPU 负载、内存、功耗、温度）
  以及 **About**。思考强度（**Thinking**，*Off* 到 *High*）决定模型思考多久；聊天的 **Sampling** 设置中的
  **Thinking budget** 可以按 token 数限制思考。
- **你的应用和编程助手：** 一个“OpenAI 兼容”的提供方，基础 URL 为 `http://127.0.0.1:8080/v1`（模型名任意；在你
  没有设置 API 密钥前，密钥也任意）；Anthropic API 在 `http://127.0.0.1:8080/v1/messages`（Claude Code：
  `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`）；Responses API 在 `/v1/responses`。支持工具和 MCP。客户端示例：
  [docs/DETAILS.md](docs/DETAILS.md#using-it)。
- **图片：** 在安装时选择启用图片（Qwen 模型），然后在聊天或应用中附加图片。
- **生成图片：** 在聊天中输入 `/image <提示词>`，或调用 `POST /v1/images/generations`（OpenAI 的图像 API），使用
  [Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image)（17B）和 Strata 自己的图像引擎。一张 1024x1024 的图片大约需要 45-70 秒（20-30 步），除了 macOS
  从 SSD 映射的 20 GB 模型文件外，只占用几 GB 内存。安装：`make pull MODEL=NUCLEUS`，然后按[步骤](docs/MACOS.md#pictures-from-a-prompt-nucleus-image)。
  也可使用 [HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil)（腾讯，80B 混合专家模型；提示与图片由同一个模型处理，无需文本编码器）：
  8 步，1024x1024 约 90 秒，除从 SSD 映射的 54 GB 文件外只占 0.3 GB 私有内存。安装：`make pull MODEL=HUNYUAN`
  （约 2 小时），然后按[步骤](docs/MACOS.md#pictures-from-a-prompt-hunyuanimage-30)设置。其许可在欧盟、英国和韩国不适用。
- **生成视频：** 在聊天中输入 `/video <提示>`，或调用 `POST /v1/videos`（OpenAI 视频 API：创建任务后轮询状态），
  使用 [Wan 2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)（两个 14B 专家），运行在 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) 的 Metal 后端上，并加上
  [lightx2v 的 Lightning LoRA](https://huggingface.co/lightx2v/Wan2.2-Lightning)（4 步代替 18 步，每步一次推理）。832x480 的 2 秒视频约需 7 分钟（不用 LoRA 时
  49 分钟），内存峰值 30 GB。安装：`make pull MODEL=WAN2.2`（29 GB，含 LoRA），然后按[步骤](docs/MACOS.md#videos-from-a-prompt-wan-22-a14b)设置。
- **默认一次处理一个请求**；其他请求排队等待。同一对话中的后续消息只读取新增内容，所以开始得很快；很长的新提示词
  需要先读取一段时间，回答才会开始。
- **从网络中的其他设备访问：** 在 `strata-<模型>.json` 中设置 `"host": "0.0.0.0"` 和 `"api_key"`，重启 Strata，
  然后用该密钥访问 `http://<你的 Mac 的 IP>:8080`。通信未加密：请只在可信的网络（或 VPN）中使用。

## 出了问题？

- **速度慢。** 如果你的 Mac 有这个选项，请把能耗模式设为 *高功率*（笔记本：*系统设置 > 电池 > 能耗模式*；台式机：
  *系统设置 > 能耗*），并让笔记本接上电源。查看 GPU 频率：`brew install macmon`，然后运行 `macmon`。
- **思考时间很长。** 降低思考强度，或设置 **Thinking budget**（4096-8192）：预算用完后，服务器会结束思考，只要
  max tokens 还留有余量，模型就会写出回答。
- **提示端口被占用。** 有其他程序在监听 8080。如果是另一个 Strata，用 `make stop` 停止它，或者启动
  `run-<模型>.sh` 时会询问是否停止它；否则换一个端口（`make run PORT=8090`）。同时运行两个模型需要两份内存。
- **模型放不下。** 关闭大型应用，选择更小的模型或更小的上下文（`make run CONTEXT=32768`）。

更多：[docs/MACOS.md](docs/MACOS.md#limits-and-warnings)（英文）。这个 Mac 分支的问题：
[issues](https://github.com/dennis-akimov/Strata/issues)。安全问题请私下报告，见 [SECURITY.md](SECURITY.md)。

## 它是怎么工作的？

在 Mac 上，Strata 自己的引擎（`strata-metal`，位于 [`metal/`](metal/)）通过 [llama.cpp](https://github.com/ggml-org/llama.cpp)
的 Metal 后端在 GPU 上运行模型，Strata 服务器运行在它之上：网页应用、OpenAI / Anthropic / Responses / MCP API、
对话复用、提前猜测几个 token 的模型草稿层，以及 Qwen、GPT-OSS（harmony）和 GLM 的对话格式。Mac 的内存由 CPU 和
GPU 共享，所以模型权重只在内存中存放一份，供两者共用。设计与测量：[docs/MACOS.md](docs/MACOS.md#how-it-fits-together)
（英文）。原项目的文档保留在 [`docs/`](docs/README.md) 中，并注明了哪些适用于 Mac。

根据提示生成图片由第二个引擎 `strata-image`（位于 [`metal/image/`](metal/image/)）完成：Nucleus-Image 的文本编码器、
17B 混合专家图像模型和解码器在同一个 Metal 后端上运行，每张图片一个进程。HunyuanImage-3.0 也在其中运行：提示的键和值只算一次，然后图像 token 经过 80B Transformer 和 3D VAE 走 8 步。

视频由 stable-diffusion.cpp 的 `sd-cli` 生成，编译时加入 Strata 的补丁，使 Wan VAE 的 3D 卷积在 Metal 上快 13 倍；
服务器把每个片段作为一个任务运行。

## 致谢与许可

**[Strata](https://github.com/Niko1221/Strata) 是 Niko1221 及 Strata 贡献者的成果**
（[全部致谢](docs/HOW_IT_WORKS.md#credits)）；这个分支增加了对 Mac 的支持。模型来自 Qwen 团队
（[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)，由
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) 压缩）、OpenAI
（[GPT-OSS](https://huggingface.co/openai/gpt-oss-120b)）和 Z.ai（GLM-5.3-Flash，由
[Project Maya](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF) 压缩）。Strata 使用了
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) 的部分代码。以 [MIT 许可证](LICENSE) 开源（Copyright (c)
2026 Niko1221 and the Strata contributors）；部分组件和每个模型都有各自的许可证（[详见](docs/HOW_IT_WORKS.md#license)）。

图片：[Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image)，由 NucleusAI 发布（Apache-2.0），由
[mlx-nucleus-image](https://huggingface.co/treadon/mlx-nucleus-image) 移植版转换而来。
[HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil)，由腾讯发布（[Tencent Hunyuan Community License](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE)：在欧盟、英国和韩国不授予许可）。
视频：[Wan 2.2](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)，由阿里巴巴 Wan 团队发布（Apache-2.0），通过 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)（MIT）运行。

如果这个 Mac 分支对你有帮助，你可以支持它的维护者：[buymeacoffee.com/dennisakimov](https://buymeacoffee.com/dennisakimov)。
