#!#/usr/bin/zsh

source setup_env.sh

workspace="_exports_"
models=("text_encoder" "unet" "vae_decoder")

echo "Converting the models"
for model in "${models[@]}"; do
  echo "Converting ${mode} model"
  ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter \
    -o "${workspace}/${model}/qnn/${model}.cpp" -i "${workspace}/${model}/onnx/${model}.onnx" \
    --input_list "${model}_input_list.txt" --act_bw "16" --bias_bw "32" \
    --quantization_overrides "${workspace}/${model}/onnx/${model}.encodings"
done
