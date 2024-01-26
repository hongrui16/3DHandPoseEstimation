import GPUtil

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

