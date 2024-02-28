source setup_env.sh

workspace="_exports_"

convert_onnx_model(){
    echo "------------------------"
    echo "Converting $1 model"
    mkdir "${workspace}/$1/qnn"
    
    ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter \
        -o "${workspace}/$1/qnn/$1.cpp" -i "${workspace}/$1/onnx/$1.onnx" \
        --input_list "input_lists/$1_input_list.txt" --act_bw "16" --bias_bw "32" \
        --quantization_overrides "${workspace}/$1/onnx/$1.encodings" \
        --input_layout "input3" "NONTRIVIAL"
}
