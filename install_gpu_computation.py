import subprocess
import sys

def install_cuda(command: str) -> None:
    # Uninstall existing packages
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])
    # Install CUDA-enabled packages
    subprocess.check_call([sys.executable, "-m", "pip", "install", *command.split(" ")])

def main() -> None:
    cuda_version = input("Please enter your CUDA version (Supported versions: 11.8, 12.4, 12.6): ")
    cuda_version = cuda_version.strip()

    if not cuda_version:
        print("Cuda version not set in config.yaml! Please set the CUDA_VERSION (e.g. CUDA_VERSION: 12.4)")
        return

    cuda_version = cuda_version.strip() # Remove whitespaces

    match cuda_version:
        case "11.8":
            install_cuda("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        case "12.4":
            install_cuda("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        case "12.6":
            install_cuda("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
        case _:
            print(f"Version {cuda_version} not found! Supported versions: 11.8, 12.4, 12.6")



if __name__ == '__main__':
    main()