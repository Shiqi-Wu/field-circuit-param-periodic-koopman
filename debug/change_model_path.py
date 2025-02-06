import torch

# 模块路径映射：旧模块路径 -> 新模块路径
module_mapping = {
    "src.param_perodic_koopman": "src.param_periodic_koopman",
}

# 自定义模块映射方法
def map_module_name(storage, location):
    return storage

def remap_module_name(module):
    for old_name, new_name in module_mapping.items():
        if module.startswith(old_name):
            return module.replace(old_name, new_name)
    return module

# 自定义 `pickle` 加载器，修复模块路径
class CustomUnpickler(torch.serialization.Unpickler):
    def find_class(self, module, name):
        module = remap_module_name(module)
        return super().find_class(module, name)

# 加载模型
model_path = "/home/shiqi/code/Project2-sensor-case/field-circuit-param-periodic-koopman/results/experiment_8/model.pth"

with open(model_path, "rb") as f:
    unpickler = CustomUnpickler(f)
    model = unpickler.load()

# 提取 `state_dict`
if hasattr(model, "state_dict"):
    state_dict = model.state_dict()
    state_dict_path = "/home/shiqi/code/Project2-sensor-case/field-circuit-param-periodic-koopman/results/experiment_8/state_dict.pth"
    torch.save(state_dict, state_dict_path)
    print(f"state_dict 已保存到: {state_dict_path}")
else:
    raise ValueError("加载的模型不包含 state_dict，请检查模型文件是否完整。")