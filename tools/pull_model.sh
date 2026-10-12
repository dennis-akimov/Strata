#!/bin/sh
# tools/pull_model.sh NAME [DATA_DIR] - the models setup does not install, each as docs/MACOS.md does it by hand
# (`make pull MODEL=NAME`; the chat quants Q2_0 ... UD-IQ4_XS go through setup instead).  Every file comes from a
# pinned Hugging Face revision with aria2c -c, so a stopped download continues when you run the same line again.
#
#   GLM-5.3        GLM-5.3-Flash Maya-S24 (94.7 GB) and its tokenizer pack
#   GPT-OSS-120B   gpt-oss-120b MXFP4 (63.4 GB, sha-256 checked) and its tokenizer pack
#   NUCLEUS        Nucleus-Image: ~50 GB downloaded, converted to 3 GGUFs (~21 GB), the downloads deleted
#   HUNYUAN        HunyuanImage-3.0-Instruct-Distil: 169 GB streamed shard by shard into one 54 GB GGUF
#   WAN2.2         Wan 2.2 T2V A14B for videos: both experts (Q4_K_M), UMT5 (Q8_0), the VAE, the Lightning LoRAs (29 GB), and
#                  stable-diffusion.cpp's sd-cli built for Metal with metal/video/sdcpp-metal-conv3d.patch
set -eu
cd "$(dirname "$0")/.."
NAME=$(printf %s "${1:-}" | tr '[:lower:]' '[:upper:]')
DATA=${2:-../Strata-data}
M=$DATA/models
PY=.venv/bin/python

get() {        # DIR URL [aria2c options]: one file into DIR, under the URL's own name (-o: Hugging Face redirects
    d=$1; u=$2; shift 2  # some files to a CDN link whose name is a hash)
    mkdir -p "$d"
    aria2c -x16 -s16 -k8M -c --file-allocation=none --console-log-level=warn --summary-interval=0 -d "$d" \
        -o "$(basename "$u")" "$@" "$u"
}
get_dir() {    # REPO REV PATH DEST: every file under PATH (its sub-folders kept) into DEST/PATH
    curl -fsS "https://huggingface.co/api/models/$1/tree/$2/$3?recursive=true" |
        "$PY" -c "import json,sys; [print(f['path']) for f in json.load(sys.stdin) if f['type'] == 'file']" |
        while read -r f; do get "$4/$(dirname "$f")" "https://huggingface.co/$1/resolve/$2/$f"; done
}

case "$NAME" in
GLM-5.3)
    D=$M/GLM-5.3-Flash-Maya-S24
    R=https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/resolve/f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da
    for i in 1 2 3; do get "$D" "$R/Maya-S24/GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-0000$i-of-00003.gguf"; done
    get "$D" "$R/vision/GLM-5.3-Flash-vocab.gguf"
    "$PY" tools/strata_tokenizer.py --gguf "$D/GLM-5.3-Flash-vocab.gguf" --out "$DATA/packs/glm-5.3-flash-maya-s24"
    ;;
GPT-OSS-120B)
    D=$M/gpt-oss-120b
    get "$D" "https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/238abdd290bb874b90a5da1b4549881b7d05c091/gpt-oss-120b-MXFP4.gguf" \
        --checksum=sha-256=582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d
    "$PY" tools/strata_tokenizer.py --gguf "$D/gpt-oss-120b-MXFP4.gguf" --out "$DATA/packs/gpt-oss-120b"
    ;;
NUCLEUS)
    D=$M/Nucleus-Image
    if [ -f "$D/nucleus-dit-q4_0.gguf" ] && [ -f "$D/nucleus-vae.gguf" ] && [ -f "$D/qwen3vl-8b-text-q8_0.gguf" ]; then
        echo "NUCLEUS: already in $D"; exit 0
    fi
    [ -x .venv-mtp/bin/python ] || { echo "NUCLEUS needs .venv-mtp (PyTorch) for its text encoder: setup makes it"; exit 1; }
    get_dir treadon/mlx-nucleus-image d5ac768cf398a25c4e157bffdcd193edf54f175b dit "$D/mlx"
    get_dir treadon/mlx-nucleus-image d5ac768cf398a25c4e157bffdcd193edf54f175b vae "$D/mlx"
    get_dir NucleusAI/Nucleus-Image 5e963db4fd0a65c7e4faf53ca2d4eca567c4dcfa text_encoder "$D/hf"
    L=$(ls -d build-metal/_deps/llama_cpp-src-* | head -1)
    "$PY" tools/image/nucleus_gguf.py --src "$D/mlx/dit" --out "$D/nucleus-dit-q4_0.gguf" --experts q4_0
    "$PY" tools/image/nucleus_gguf.py --vae --src "$D/mlx/vae" --out "$D/nucleus-vae.gguf"
    .venv-mtp/bin/python "$L/convert_hf_to_gguf.py" "$D/hf/text_encoder" --outtype q8_0 --outfile "$D/qwen3vl-8b-text-q8_0.gguf"
    rm -rf "${D:?}/mlx" "${D:?}/hf"            # only the 3 GGUFs are used
    ;;
HUNYUAN)
    D=$M/HunyuanImage-3
    [ -f "$D/hunyuan-image3-q4_0.gguf" ] ||           # its .tmp is not resumable: a stopped conversion starts over
        "$PY" tools/image/hunyuan_gguf.py --out "$D/hunyuan-image3-q4_0.gguf" --work "$D/shards"
    get "$D" "https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil/resolve/c8ffd07206f1b843697606968196e8f59f8ff38c/tokenizer.json"
    ;;
WAN2.2)
    D=$M/Wan2.2-T2V-A14B
    Q=https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/73eafba53a1a8f29254e4c77f92e74ea27d7cd6f
    get "$D" "$Q/HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf"
    get "$D" "$Q/LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf"
    get "$D" "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7/umt5-xxl-encoder-Q8_0.gguf"
    get "$D" "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/123acf1cc74bccbb9bfff8ac1ee72edc08c2341d/split_files/vae/wan_2.1_vae.safetensors"
    # lightx2v's Wan2.2-Lightning LoRAs (4 steps, no CFG; Apache-2.0): their integer alphas as float32 for sd-cli
    L=https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/18bccf8884ec0a078eed79785eb4ef13ea16ce1e/Wan2.2-T2V-A14B-4steps-lora-250928
    for x in high low; do
        [ -f "$D/loras/lightning-$x.safetensors" ] && continue   # the converter writes it only when complete
        get "$D/loras" "$L/${x}_noise_model.safetensors"
        "$PY" tools/video/lora_alpha_f32.py "$D/loras/${x}_noise_model.safetensors" "$D/loras/lightning-$x.safetensors"
        rm -f "$D/loras/${x}_noise_model.safetensors"
    done
    # sd-cli: stable-diffusion.cpp at a pinned commit, with Strata's patch (Metal has no IM2COL_3D: the Wan VAE's
    # Conv3d as 2D convolutions per temporal tap, 13x faster decoding)
    B=build-sdcpp
    SDC=d637cfb935feb2254898a8f1524046c958f3df80
    [ -d $B/src/.git ] || git clone -q https://github.com/leejet/stable-diffusion.cpp $B/src
    if [ "$(git -C $B/src rev-parse HEAD)" != $SDC ] || ! git -C $B/src apply --reverse --check ../../metal/video/sdcpp-metal-conv3d.patch 2>/dev/null; then
        git -C $B/src fetch -q origin && git -C $B/src checkout -q -f $SDC && git -C $B/src submodule update -q --init --recursive
        git -C $B/src apply ../../metal/video/sdcpp-metal-conv3d.patch
    fi
    NINJA=$(command -v ninja || echo "$PWD/.venv/bin/ninja")
    cmake -S $B/src -B $B/build -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" -DCMAKE_BUILD_TYPE=Release -DSD_METAL=ON > /dev/null
    cmake --build $B/build --target sd-cli
    echo "sd-cli: $PWD/$B/build/bin/sd-cli"
    ;;
*)
    echo "make pull MODEL=NAME: a setup quant (Q2_0 IQ2_XS IQ3_XXS IQ3_S IQ1_M UD-Q4_K_XL UD-IQ4_XS) or one of"
    echo "GLM-5.3 GPT-OSS-120B NUCLEUS HUNYUAN WAN2.2 (tools/pull_model.sh)"
    exit 2
    ;;
esac
echo "$NAME: done in $M"
