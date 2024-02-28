source setup_env.sh

source qnn_conversion/convert_onnx_model.sh
source qnn_conversion/generate_model_lib.sh
source qnn_conversion/generate_context_binary.sh

echo "------------------------"
echo "Converting the models for QNN"

#models=("text_encoder" "unet" "vae_decoder")
models=("unet")

for model in "${models[@]}"; do
    convert_onnx_model $model
    generate_model_lib $model
    generate_context_binary $model
done

