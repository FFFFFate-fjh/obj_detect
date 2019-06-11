# obj_detect
python obj_detect.py source.jpg t1.jpg
## 方法优缺点
主要算法:<br/> 
1.大津法阈值分割<br/> 
2.轮廓检测<br/> 
3.hash,多通道直方图相似度<br/> 
4.sift特征检测与匹配<br/> 
5.模板匹配<br/> 
### 优点
1.物体快速检测得到boundingbox<br/> 
2.boundingbox有较高的IOU<br/> 
3.对不同尺度有一定适应性<br/> 
3.整体算法下有较高的召回率<br/> 
### 缺点
1.小物体特征点数量不足,导致漏识别<br/> 
2.物体不同的动作下识别准确率不足<br/> 
3.算法中设置了多个阈值,多于不同识别对象,统一阈值效果不足<br/> 
4.相互连通,有重叠的物体检测会将其合并<br/> 
## 其他方法
1.使用多个特征结合<br/> 
2.统一尺度下,结合模板匹配检测物体位置可以提升检测效果<br/> 
3.使用基于学习特征的方法,例如ferns<br/> 

