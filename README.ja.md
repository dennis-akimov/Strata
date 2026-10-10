<h1 align="center">Strata for Mac</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · **日本語** · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>1000 億パラメータ級の AI モデルを自分の Mac で動かす</b><br>
Apple Silicon · M5 Max（128 GB）でテスト済み · 無料のオープンソース</p>

> **これは [Strata](https://github.com/Niko1221/Strata) を独自にメンテナンスしている、Apple Silicon 搭載 Mac 専用の
> フォークです**（メンテナー：[dennis-akimov](https://github.com/dennis-akimov)）。Metal エンジンと Mac 用のセットアップを
> 追加しています。オリジナルの Windows・Linux 用エンジン（NVIDIA CUDA、AMD HIP、Intel SYCL）はフォークした時点の
> まま このリポジトリに残っていますが、ここではビルド・テスト・更新を行っていません。
> **Windows または Linux の PC では、オリジナルを使ってください：[github.com/Niko1221/Strata](https://github.com/Niko1221/Strata)。**

Strata は大規模な AI モデルを自分のコンピューターで動かします。チャット、コードの作成、画像の読み取りができ、
クラウドサービスと同じ API を通じてアプリやコーディングエージェントと連携します。モデルは Mac 上で動きます。
Mac の外に出るものは、あなたが決めます（モデルのダウンロードと、接続したアプリ・ツール・MCP サーバー）。

## どのくらい速い？

M5 Max（40 コア GPU、128 GB）の MacBook Pro、macOS 26.4、省エネルギーモード *高出力* で測定（2026-10-08、思考オフ）。
1 トークンは英語で約 ¾ 語です。

| モデル | 900 語の回答 | 短い回答 | メモリ上のモデルファイル |
| --- | ---: | ---: | ---: |
| **[GPT-OSS 120B](docs/MACOS.md#gpt-oss-120b)**（MXFP4） | 93-100 トークン/秒 | 93-103 トークン/秒 | 63 GB |
| **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) Q2_0**（`--mtp on`） | 60-69 トークン/秒 | 92-102 トークン/秒 | 38 GB |
| **Qwen3.8-Flash-Next IQ3_XXS**（`--mtp on`） | 46-48 トークン/秒 | 73-92 トークン/秒 | 47 GB |

Qwen の数値は MTP ドラフト層を使っています。MTP は任意で、有効にするには `./setup.sh --setup --mtp on`（ディスクを
約 12 GB 追加で使用）。コンテキストのキャッシュはモデルファイルとは別に必要です（数 GB、長いコンテキストではさらに
多く）。省エネルギーモードが *自動* のときは、同じ Mac で長い回答が 2〜3 倍遅くなりました。ほかの Mac では測定して
いません。すべての数値と測定方法：[docs/MACOS.md](docs/MACOS.md#measured)（英語）。

## 必要なもの

| | |
| --- | --- |
| **Mac** | Apple Silicon（M1 以降）。テストしたのは 128 GB の M5 Max 1 台だけです。ほかのチップやメモリ容量でも動くはずですが、試していません。 |
| **メモリ** | モデルファイルとキャッシュが、macOS が GPU に使わせる範囲（テストした 128 GB の Mac では 107.5 GB）に収まること。セットアップのチェックは 64 GB を最低ラインとしています。`make check` があなたの Mac に収まるモデルを見積もります。下のモデル表も参照してください。 |
| **ディスク** | モデルのダウンロード分と数 GB（1 モデルあたり 70〜100 GB）、内蔵 SSD 上。 |
| **ソフトウェア** | Apple の Command Line Tools（`xcode-select --install`、完了するまで待つ）と Python 3.10 以降（ない場合、Homebrew があればセットアップが Homebrew でインストールします）。macOS 26.4 でテスト済み。 |

## インストール

ターミナルで：

```sh
git clone https://github.com/dennis-akimov/Strata.git
cd Strata
make check                 # この Mac で動かせるもの。何もインストールしない
make pull MODEL=Q2_0       # エンジンをビルドし、モデルをダウンロード（66 GB）
make run                   # モデルを読み込み、準備ができたら http://127.0.0.1:8080 を開く
```

`make pull` はいくつか質問します（コンテキストの大きさ、画像）。Enter で推奨の答えを選べます。ダウンロードが止まったら
もう一度実行してください。続きから再開します。読み込みには起動のたびに 1 分ほどかかります。`make run` は Strata が
動いている間ターミナルを使い続けます。Ctrl+C で止めるか、`make start` / `make stop` でバックグラウンド実行できます。
すべてのオプション、制限、注意点：**[docs/MACOS.md](docs/MACOS.md)**（英語）。

**更新：** `git pull` のあと `make run`（ソースが変わっていればセットアップがエンジンを再ビルドします。モデルは
そのまま）。**保存場所：** モデルは `Strata` フォルダーの隣の `Strata-data/`、設定は `strata-<モデル>.json`、
チャットはブラウザーの中です。

## どのモデルを選ぶ？

| モデル | ダウンロード | メモリ上（128K コンテキスト） | メモ |
| --- | ---: | ---: | --- |
| **Qwen3.8-Flash-Next Q2_0** | 66 GB | 約 60 GB（推定） | 標準で、Mac で最も多くテストしたもの。`make pull MODEL=Q2_0` |
| Qwen3.8-Flash-Next IQ3_XXS | 76 GB | 65 GB（実測） | Q2_0 より圧縮が少ない。今回の測定では 10〜30 % 遅い。`make pull MODEL=IQ3_XXS` |
| Qwen3.8-Flash-Next IQ3_S | 84 GB | 74 GB（実測） | さらに圧縮が少ない。A/B テスト 1 回で Q2_0 より約 10 % 遅い。`make pull MODEL=IQ3_S` |
| **GPT-OSS 120B** | 63 GB | 約 70 GB（推定） | ここで最速（1 トークンあたり約 50 億パラメータが動作）。手動でセットアップ：[手順](docs/MACOS.md#gpt-oss-120b) |
| GLM-5.3-Flash（Maya-S24） | 95 GB | 32K で約 100 GB（推定） | コードのみ：Strata はトークナイザー、チャット形式、ツール呼び出しに対応していますが、まだモデルを動かしていません。 |

「メモリ上」は、テストした Mac で Qwen モデルの MTP ドラフト層とリクエスト枠 2 つを含むエンジン全体の値です。
コンテキストを小さくすれば少なくて済みます（`make run CONTEXT=32768`）。これが macOS が GPU に使わせる範囲に収まる
必要があり、その値は `make check` が表示します（テストした 128 GB の Mac では 107.5 GB、小さい Mac ではもっと少ない）。
Qwen モデルは動作中に 28 GB のテーブルを SSD から読み込みます。ほかの Qwen のサイズ（IQ2_XS、Unsloth のもの）は Mac
では試していません。

## 使い方

- **ブラウザーで：** `http://127.0.0.1:8080` に **Chat**、モデルと Mac のライブ **Monitor**（GPU 負荷、メモリ、電力、
  温度）、**About** があります。思考の強さ（**Thinking**、*Off*〜*High*）でモデルが考える長さが決まり、チャットの
  **Sampling** 設定の **Thinking budget** でトークン数の上限を設定できます。
- **アプリやコーディングエージェントから：** ベース URL `http://127.0.0.1:8080/v1` の「OpenAI 互換」プロバイダー
  （モデル名は任意。API キーを設定していなければキーも任意）、`http://127.0.0.1:8080/v1/messages` の Anthropic API
  （Claude Code：`ANTHROPIC_BASE_URL=http://127.0.0.1:8080`）、`/v1/responses` の Responses API。ツールと MCP も
  使えます。クライアントの例：[docs/DETAILS.md](docs/DETAILS.md#using-it)。
- **画像：** セットアップで画像を有効にし（Qwen モデル）、チャットやアプリで添付します。
- **標準では 1 度に 1 リクエスト**で、ほかは待ちます。同じ会話の続きは新しい部分だけを読むのですぐ始まりますが、
  長い新しいプロンプトは読み込みに時間がかかってから回答が始まります。
- **ネットワーク上の別の機器から：** `strata-<モデル>.json` に `"host": "0.0.0.0"` と `"api_key"` を設定して
  Strata を再起動し、`http://<Mac の IP>:8080` にそのキーで接続します。通信は暗号化されません。信頼できる
  ネットワーク（または VPN）で使ってください。

## うまくいかないときは

- **遅い。** Mac にあれば省エネルギーモードを *高出力* にしてください（ノート：*システム設定 > バッテリー >
  省エネルギーモード*、デスクトップ：*システム設定 > エネルギー*）。ノートは電源につないでおきます。GPU の
  クロックを見るには `brew install macmon` のあと `macmon`。
- **考える時間がとても長い。** 思考の強さを下げるか、**Thinking budget**（4096〜8192）を設定します。上限に達すると
  サーバーが思考を閉じ、max tokens に余裕があればモデルが回答を書きます。
- **ポートが使用中と表示される。** 別のプログラムが 8080 で待ち受けています。別の Strata なら `make stop` で止める
  か、`run-<モデル>.sh` を起動すると止めるかどうか聞いてきます。そうでなければ別のポートを使います
  （`make run PORT=8090`）。2 つのモデルを同時に動かすには両方の分のメモリが必要です。
- **モデルが収まらない。** 大きなアプリを閉じるか、小さいモデルや小さいコンテキストを選びます
  （`make run CONTEXT=32768`）。

詳しくは：[docs/MACOS.md](docs/MACOS.md#limits-and-warnings)（英語）。この Mac 用フォークの問題：
[issues](https://github.com/dennis-akimov/Strata/issues)。セキュリティの問題は非公開で報告してください：
[SECURITY.md](SECURITY.md)。

## どう動いている？

Mac では、Strata 独自のエンジン（`strata-metal`、[`metal/`](metal/) 内）が [llama.cpp](https://github.com/ggml-org/llama.cpp)
の Metal バックエンドを通じて GPU でモデルを動かし、その上で Strata のサーバーが動きます。ウェブアプリ、OpenAI /
Anthropic / Responses / MCP の API、会話の再利用、数トークン先を予測するモデルのドラフト層、そして Qwen・GPT-OSS
（harmony）・GLM のチャット形式です。Mac のメモリは CPU と GPU で共有されるので、モデルの重みは両方のために 1 回だけ
メモリに置かれます。設計と測定：[docs/MACOS.md](docs/MACOS.md#how-it-fits-together)（英語）。オリジナルのドキュメントは
[`docs/`](docs/README.md) に残してあり、Mac で当てはまるものがわかるようにしてあります。

## クレジットとライセンス

**[Strata](https://github.com/Niko1221/Strata) は Niko1221 と Strata のコントリビューターによる成果です**
（[すべてのクレジット](docs/HOW_IT_WORKS.md#credits)）。このフォークは Mac 対応を追加しています。モデルは Qwen チーム
（[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、圧縮：
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)）、OpenAI
（[GPT-OSS](https://huggingface.co/openai/gpt-oss-120b)）、Z.ai（GLM-5.3-Flash、圧縮：
[Project Maya](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)）によるものです。Strata は
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) の一部を使っています。[MIT ライセンス](LICENSE)
（Copyright (c) 2026 Niko1221 and the Strata contributors）のオープンソースです。一部のコンポーネントと各モデルには
それぞれのライセンスがあります（[一覧](docs/HOW_IT_WORKS.md#license)）。

この Mac 用フォークが役に立ったら、メンテナーを支援できます：
[buymeacoffee.com/dennisakimov](https://buymeacoffee.com/dennisakimov)。
