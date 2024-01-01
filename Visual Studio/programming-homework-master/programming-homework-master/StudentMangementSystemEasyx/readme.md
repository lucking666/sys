# 图形界面版本开发过程整理
**更新时间** 

2020-12-24

* 开发环境

2020-12-28

* 图形库安装
* 创建新项目

2021-1-4

* 界面设置


## 开发环境
系统：Win 10

IDE：visual studio 2019

图形库： EasyX
## 安装EaxyX库

### 什么是EasyX

EasyX 是针对 C++ 的图形库，可以帮助 C 语言初学者快速上手图形和游戏编程。

比如，可以用 VC + EasyX 很快的用几何图形画一个房子，或者一辆移动的小车，可以编写俄罗斯方块、贪吃蛇、黑白棋等小游戏，可以练习图形学的各种算法，等等。

### 安装

首先去[官网](https://easyx.cn/)下载界面下载图形库。

![image-20201228121703711](https://gitee.com/wyloving/picCloud/raw/master/image-20201228121703711.png)

双击下载下来的安装包

![image-20201228121836493](https://gitee.com/wyloving/picCloud/raw/master/image-20201228121836493.png)

点击相应版本进行安装，建议把帮助文档也安装上。我的电脑上是visual studio2019。

![image-20201228122038334](https://gitee.com/wyloving/picCloud/raw/master/image-20201228122038334.png)

![image-20201228122109774](https://gitee.com/wyloving/picCloud/raw/master/image-20201228122109774.png)

## 新建项目并简单测试

打开VS2019并新建空白项目。

![image-20201228122227521](https://gitee.com/wyloving/picCloud/raw/master/image-20201228122227521.png)

右击源文件并点击添加-新建项。

![image-20201228122326672](https://gitee.com/wyloving/picCloud/raw/master/image-20201228122326672.png)

选择C++源文件，并修改名字为main

![image-20201228122439920](https://gitee.com/wyloving/picCloud/raw/master/image-20201228122439920.png)

输入简单的测试文件，看能否运行图形库。

```cpp
#include <graphics.h>		// 引用图形库头文件
#include <conio.h>
int main()
{
	initgraph(640, 480);	// 创建绘图窗口，大小为 640x480 像素
	circle(200, 200, 100);	// 画圆，圆心(200, 200)，半径 100
	_getch();				// 按任意键继续
	closegraph();			// 关闭绘图窗口
	return 0;
}
```

出现下图说明能够正常使用。

![image-20201228122611184](https://gitee.com/wyloving/picCloud/raw/master/image-20201228122611184.png)

解释下个函数的作用。

`initgraph(width,height)`是初始化窗口并设定长宽大小。

`closegraph`是在结束的时候关闭窗口用的。

## 界面设置

整体分成两个部分，左侧按钮功能区，右侧是信息显示区。

```cpp
int main()
{
	initgraph(WIDTH, HEIGHT);	// 创建绘图窗口，大小为 640x480 像素
	drawLeft();
	drawRight();
	_getch();				// 按任意键继续
	closegraph();			// 关闭绘图窗口
	return 0;
}
```

定义按钮结构体

```cpp
//包含按钮放置的位置和按键信息
btNode leftMen[6] = {
	{5,5,L"    作品信息 "},
	{5,35,L"显示学生信息"},
	{5,65,L"增加学生信息"},
	{5,95,L"删除学生信息"},
	{5,125,L"修改学生信息"},
	{5,155,L"查询学生信息"}
};
```

绘制按钮

```cpp
//绘制按钮功能区
void drawBtn(btNode t) {
	setfillcolor(RGB(93, 107, 153));
	setbkmode(TRANSPARENT);
	fillroundrect(t.x, t.y, t.x + 120, t.y + 20, 10, 10);
	outtextxy(t.x + 15, t.y + 2, t.text);

}
```

`setfillcolor`是设定填充颜色。

`setbkmode`是设置背景模式，选择了透明模式

`fillroundrext`绘制圆角矩形，作为按钮的形状

`outtextxy`在对应位置输出文字，显示按钮信息

绘制左侧功能按钮区

``` cpp
//绘制左侧功能按钮区
void drawLeft() {
	
	SetWorkingImage(&leftImg);//设置绘制的图像
	setbkcolor(RGB(93, 107, 153));//设置背景色
	cleardevice();//清空图像

	//绘制按钮
	for (int i = 0;i < 6;i++) {
		drawBtn(leftMen[i]);
	}
	SetWorkingImage();//将图像绘制到窗口上
	putimage(0, 0, &leftImg);
}
```

绘制右侧显示区

``` cpp
//绘制右侧显示区
void drawRight() {
	SetWorkingImage(&rightImg);//设置绘制的图像
	
	setbkcolor(RGB(247, 249, 254));//设置背景色
	cleardevice();//清空图像
	SetWorkingImage();//将图像绘制到窗口上
	putimage(161, 0, &rightImg);
}
```

![image-20210104161410013](https://gitee.com/wyloving/picCloud/raw/master/image-20210104161410013.png)

## 增加按钮的点击状态

现在的按钮部分仅仅是个图像，要想产生点击的功能，还要配合对鼠标消息的处理。

想设置出的效果是这样的:按钮点击后，对应的颜色会发生改变。可以增加一个状态子属性，检测到点击信息后，修改对应状态值，在绘制按钮时，对状态值进行判断，不同的状态对应不同的颜色。

按钮结构体

```cpp
typedef struct btNode{
	int x, y;
	wchar_t text[20];//内容
	int status;//0-默认  1-按下
}btNode;
```

绘制按钮

```cpp
//绘制按钮
void drawBtn(btNode t) {
	if (t.status == 0)
		setfillcolor(RGB(93, 107, 153));//设置填充颜色
	else
		setfillcolor(RGB(204, 213, 240));

	setbkmode(TRANSPARENT);//设置背景模式为透明

	//绘制圆角矩形作为按钮形状

	fillroundrect(t.x, t.y, t.x + 120, t.y + 20, 10, 10);
	outtextxy(t.x + 15, t.y + 2, t.text);//输出按钮信息

}
```

接下来就是对鼠标行为的检测判断了，这部分可以查看一下帮助文档，

![image-20210106013959209](https://gitee.com/wyloving/picCloud/raw/master/image-20210106013959209.png)

![image-20210106014142387](https://gitee.com/wyloving/picCloud/raw/master/image-20210106014142387.png)

在主函数中，修改逻辑，不断循环获取鼠标信息看是否按下左键，在判断鼠标的位置是否在对应的按钮区域内，在对应的区域就修改哪个属于的按键状态。

``` cpp
int main()
{
	initgraph(WIDTH, HEIGHT);	// 创建绘图窗口，大小为 640x480 像素
	int i;
	MOUSEMSG m;//定义鼠标信息

	while (1) {

		m = GetMouseMsg();//获取鼠标信息
		if (m.uMsg == WM_LBUTTONDOWN) {
			if (m.x >= 0 && m.x <= 160) {//鼠标在左侧功能按键区
				for (i = 0;i < 6;i++) {
					//判断鼠标位置是否在按钮范围内
					if (m.x >= leftMen[i].x && m.x <= leftMen[i].x + 120 && m.y >= leftMen[i].y && m.y <= leftMen[i].y + 20) {
						setUpLeftBtn();//初始化按钮状态
						leftMen[i].status = 1;//修改对应按钮状态
					}


				}
				
			}
		}


		drawLeft();//绘制左侧区域
		drawRight();//绘制右侧区域
	}


	closegraph();			// 关闭绘图窗口
	return 0;
}
```

初始化左侧按钮状态

``` cpp
// 初始化左侧按钮状态
void setUpLeftBtn() {
	int i;
	for (i = 0;i < 6;i++) {
		leftMen[i].status = 0;
	}
}
```

![image-20210106014610053](https://gitee.com/wyloving/picCloud/raw/master/image-20210106014610053.png)

剩余内容待整理上传

