#include <easyx.h>			// 引用图形库头文件
#include <conio.h>

const int WIDTH = 1000;
const int HEIGHT = 618;

IMAGE leftImg(160, 618);
IMAGE rightImg(840, 618);


typedef struct btNode{
	int x, y;
	wchar_t text[20];//内容
	int status;//0-默认  1-按下
}btNode;

btNode leftMen[6] = {
	{5,5,L"    作品信息 ",0},
	{5,35,L"显示学生信息",0},
	{5,65,L"增加学生信息",0},
	{5,95,L"删除学生信息",0},
	{5,125,L"修改学生信息",0},
	{5,155,L"查询学生信息",0}
};



/*
按钮
主页   ---  显示作品信息 

显示学生信息
增加学生信息
删除学生信息
修改学生信息
查询学生信息

*/

//绘制按钮
void drawBtn(btNode t) {
	if (t.status == 0)
		setfillcolor(RGB(93, 107, 153));//设置填充颜色
	else
		setfillcolor(RGB(204, 213, 240));

	setbkmode(TRANSPARENT);//设置背景模式为透明

	//绘制圆角矩形作为按钮形状

	/*
	x,y
	    x+120,y+20
	
	*/
	fillroundrect(t.x, t.y, t.x + 120, t.y + 20, 10, 10);
	outtextxy(t.x + 15, t.y + 2, t.text);//输出按钮信息

}

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

//绘制右侧显示区
void drawRight() {
	SetWorkingImage(&rightImg);//设置绘制的图像

	setbkcolor(RGB(247, 249, 254));//设置背景色
	cleardevice();//清空图像
	SetWorkingImage();//将图像绘制到窗口上
	putimage(161, 0, &rightImg);
}


// 初始化左侧按钮状态
void setUpLeftBtn() {
	int i;
	for (i = 0;i < 6;i++) {
		leftMen[i].status = 0;
	}
}



int main()
{
	initgraph(WIDTH, HEIGHT);	// 创建绘图窗口，大小为 640x480 像素
	int i;
	ExMessage m;//定义鼠标信息

	while (1) {

		m = getmessage();//获取鼠标信息
		if (m.message == WM_LBUTTONDOWN) {
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