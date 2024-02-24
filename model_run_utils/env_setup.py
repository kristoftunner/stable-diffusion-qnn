import numpy as np
import shutil
import os


def setup_env() -> os.path:
    SDK_dir = os.getenv('QNN_SDK_ROOT')

    libs_dir = os.path.join(SDK_dir, 'lib/aarch64-windows-msvc')
    net_run_binary = os.path.join(
        SDK_dir, 'bin/aarch64-windows-msvc/qnn-net-run.exe')
    skel_file = os.path.join(
        SDK_dir,  "lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so")

    bin_dir = os.path.join(os.getcwd(), 'bin')
    if not os.path.exists(bin_dir):
        os.mkdir(bin_dir)

    # Copy necessary libraries to a common location
    libs = ["QnnHtp.dll", "QnnHtpNetRunExtensions.dll",
            "QnnHtpPrepare.dll", "QnnHtpV68Stub.dll"]

    for lib in libs:
        shutil.copy(os.path.join(libs_dir, lib), bin_dir)

    shutil.copy(net_run_binary, bin_dir)
    shutil.copy(skel_file, bin_dir)

    return net_run_binary


def check_user_inputs(user_seed, user_step, user_text_guidance):
    if not isinstance(user_step, int):
        raise ValueError("user_step should be of int type")
    if (user_step != 20) and (user_step != 50):
        raise ValueError("user_step should be either 20 or 50")
    if not isinstance(user_seed, np.int64):
        raise ValueError("user_seed is not int64")
    if not isinstance(user_text_guidance, float):
        raise ValueError("user_text_guidance should be of float type")
    if user_text_guidance < 5.0 or user_text_guidance > 15.0:
        raise ValueError("user_text_guidance should be [5.0, 15.0]")
