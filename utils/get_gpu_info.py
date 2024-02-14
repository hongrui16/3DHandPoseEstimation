import GPUtil
import torch

def print_gpu_utilization():
    # 获取所有可用的GPU信息
    gpus = GPUtil.getGPUs()
    
    # 遍历所有的GPU并打印它们的详细信息
    for gpu in gpus:
        print(f"GPU ID: {gpu.id}, Name: {gpu.name}")
        print(f"  GPU Load: {gpu.load*100}%")
        print(f"  GPU Memory Total: {gpu.memoryTotal}MB")
        print(f"  GPU Memory Used: {gpu.memoryUsed}MB")
        print(f"  GPU Memory Utilization: {gpu.memoryUtil*100}%")
        print("")

def get_gpu_utilization_as_string():
    gpu_strings = []
    gpus = GPUtil.getGPUs()

    for gpu in gpus:
        gpu_info = (
            f"GPU ID: {gpu.id}, Name: {gpu.name}\n"
            f"  GPU Load: {gpu.load * 100}%\n"
            f"  GPU Memory Total: {gpu.memoryTotal}MB\n"
            f"  GPU Memory Used: {gpu.memoryUsed}MB\n"
            f"  GPU Memory Utilization: {gpu.memoryUtil * 100}%\n"
        )
        gpu_strings.append(gpu_info)

    return '\n'.join(gpu_strings)


def check_gpu_available():
    flag = torch.cuda.is_available()
    if flag:
        print("CUDA is available")
        device = "cuda"
    else:
        print("CUDA is unavailable")
        device = "cpu"

def print_available_gpu():
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        # 打印 GPU 的数量
        print("Number of GPUs available:", torch.cuda.device_count())

        # 遍历并打印每个 GPU 的名称
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available.")


if __name__ == '__main__':
    print_available_gpu()