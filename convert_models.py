import os

STABLE_DIFFUSION_MODELS = os.getcwd() + "/assets/stable_diffusion"

if __name__ == '__main__':
    fp32_pickle_path = STABLE_DIFFUSION_MODELS + '/fp32.npy'

    env = os.environ.copy()
    env["QNN_SDK_ROOT"] = QNN_SDK_ROOT
    env["PYTHONPATH"] = QNN_SDK_ROOT + "/benchmarks/QNN/:" + QNN_SDK_ROOT + "/lib/python"
    env["PATH"] = QNN_SDK_ROOT + "/bin/x86_64-linux-clang:" + env["PATH"]
    env["LD_LIBRARY_PATH"] = QNN_SDK_ROOT + "/lib/x86_64-linux-clang"
    env["HEXAGON_TOOLS_DIR"] = QNN_SDK_ROOT + "/bin/x86_64-linux-clang"
