# import numpy as np
#
# # 读取.npy文件
# coordinates_3d = np.load('D:/duanxianlin/PoseFormerV21-main/PoseFormerV2-main/demo/output/video (6)/coordinates_3d.npy')
#
# # 查看数据
# print(coordinates_3d)



import numpy as np

# 加载数据
coordinates_3d = np.load('')

# 保存为txt文本文件
np.savetxt('coordinates_3d.txt', coordinates_3d.reshape(coordinates_3d.shape[0], -1),
           fmt='%1.8f', delimiter='\t')

print("数据已保存为coordinates_3d.txt文件。")
