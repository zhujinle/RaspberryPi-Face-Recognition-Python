# RaspberryPi-Face-Recognition-Python
Raspberry Pi Face Recognition Code with Web Control Panel 公选课最终代码

---

​        在 `main.py` 里面做Flask的配置，使用`camera_opencv.py` 进行人脸的标记与框选，这一块性能要求不高所以可以做到实时。点击识别后进入 /recognized_feed网页，从另一个api地址传递base64后的图像数据。图像数据用框标记出人脸位置并且标注人脸名称，测试下来识别率还是蛮高的哈哈哈哈。

​        图像数据使用`facer.py` 进行处理，调用face_recognition库进行识别，识别时从\dataset文件夹中读入所有人脸照片并进行训练传参。原本准备上mysql或者redis的，迫于学校的垃圾网速放弃了。本地部署由于树莓派垃圾的性能不大可行。

---

### 展望

· 其实完全可以使用数据库保存，本地的.db写起来有点麻烦，而且这个内存卡性能也不大行，怕给他搞坏了要赔钱。学校网络太烂了，不然还能自己在寝室里放台3B专门做redis的部署

· 没能做成实时的，还是性能限制。。原本还准备上 `CNN` 训练一个模型出来用的，自带的那个库对于亚洲人的效果不大行，但是想想树莓派的性能，还是放弃了。倒是同组的同学做出来一个，在电脑上跑能做到实时，不过用的是自带的那个库最终fps只有个位数😅

· 界面可以美化一下，搞一个css上去，套一层模板还能做个login或者文件上传的端口，不过文件上传就要做文件判断来防马，还是麻烦懒得做了。

· 其实树莓派的GPIO也可以利用起来，拿来控制道闸或者三辊闸的蛮不错的

---

~~反正结课了，代码共享出来交给下一届的同学水学分去了~~ 
