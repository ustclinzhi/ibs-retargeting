import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# 示例数据：x 值为索引，y1 和 y2 分别为两个折线的纵坐标数据
x = list(range(10))
y1 = [3, 5, 2, 6, 7, 4, 8, 6, 7, 5]
y2 = [4, 4, 6, 3, 5, 9, 7, 8, 6, 4]

# 绘制折线
plt.plot(x, y1, marker='o', label='折线1')
plt.plot(x, y2, marker='o', label='折线2')

# 添加标题和坐标标签
plt.title('两个折线图')
plt.xlabel('X轴')
plt.ylabel('Y轴')

# 显示图例
plt.legend()

# 显示图形
plt.show()