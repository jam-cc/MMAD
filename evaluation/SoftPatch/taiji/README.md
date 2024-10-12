

### 1. Devcloud账号申请
打开该网页http://devcloud.woa.com/#!/gpu/gpu_man?page=1&size=10
申请GPU服务器。其中机房位置，镜像，产品请选择：
重庆，tlinux2.2，工业AI

### 2.git上下载太极示例
在https://git.woa.com/niceliu/taiji_beginner.git
下载该示例代码是一个简单的pytorch版cifar10训练案例；
cifar10下载脚本在cifar10目录中，需自行下载至该目录下；

### 3. 提交太极任务

每个任务提交需提供两个文件：config.json和start.sh，这俩个文件在示例代码的taiji目录下；
（1）配置config.json文件；
（2）编写任务启动脚本start.sh
（3）提交太极任务命令：jizhi_client start -scfg config.json

### 4. 查看太极任务

每个任务提交后，可以在https://taiji.oa.com/#/project-list
链接下查看提交的任务详情

