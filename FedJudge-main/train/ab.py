# import sys
# import os
# # Add the root directory (project) to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# # # 获取项目根目录
# # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# # # 将项目根目录添加到 sys.path 中
# # sys.path.append(project_root)
# # sys.path.append('/home/chen/pyh/FedJudge-main/utils')
# # sys.path.append('')
# print(sys.path)
# # import options
# import utils
# import utils.options
# # 现在可以从 utils 导入
# from utils.sampling import mnist_iid, mnist_noniid, cifar_iid


import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 然后你就可以使用 import component.dataset
import component.dataset