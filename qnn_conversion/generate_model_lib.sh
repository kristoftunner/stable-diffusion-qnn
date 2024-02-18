source setup_env.sh

workspace="_exports_"
models=("text_encoder" "unet" "vae_decoder")

generate_model_lib(){
    echo "------------------------------------"
    echo "Generating model library for $1"
    ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator \
        -c "${workspace}/$1/qnn/$1.cpp" \
        -b "${workspace}/$1/qnn/$1.bin" \
        -t "x86_64-linux-clang"                      \
        -o "${workspace}/$1/qnn/converted_$1"
}

