source setup_env.sh

source convert_onnx_model.sh
source generate_model_lib.sh
source generate_context_binary.sh

echo "------------------------"
echo "Converting the models for QNN"

for model in "${models[@]}"; do
    convert_onnx_model $model
    generate_model_lib $model
    generate_context_binary $model
done

