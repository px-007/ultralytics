from ultralytics import YOLOE
# 1. 加载 YOLOE 模型
model = YOLOE("yoloe-26n-seg.pt")
# 2. 定义你要检测的类别
names = ["人抱在怀中的红色物体"]
# 3. 关键步骤：将文本类别转换为文本嵌入，并设置给模型
model.set_classes(names, model.get_text_pe(names))
# 4. 进行预测
results = model.predict(r"D:\workspace\ultralytics\ultralytics\models\yolo\yoloe\images\gufeng_0147.jpg",conf=0.15)
# 5. 保存结果
results[0].save()