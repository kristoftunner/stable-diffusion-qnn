source setup_env.sh

workspace="_exports_"
platform="x86_64-linux-clang"

generate_context_binary(){
    echo "--------------------------------------"
    echo "Generating context binary for $1"
    ${QNN_SDK_ROOT}/bin/${platform}/qnn-context-binary-generator                                        \
        --model "${workspace}/$1/qnn/converted_$1/${platform}/lib$1.so"               \
        --backend "${QNN_SDK_ROOT}/lib/${platform}/libQnnHtp.so"                                         \
        --output_dir  "${workspace}/$1/qnn/converted_$1/${platform}/serialized_binaries"    \
        --binary_file "$1.serialized"                                                             \
        --log_level "verbose"
}

