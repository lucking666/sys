#include <easyx.h>			// ����ͼ�ο�ͷ�ļ�
#include <conio.h>

const int WIDTH = 1000;
const int HEIGHT = 618;

IMAGE leftImg(160, 618);
IMAGE rightImg(840, 618);


typedef struct btNode{
	int x, y;
	wchar_t text[20];//����
	int status;//0-Ĭ��  1-����
}btNode;

btNode leftMen[6] = {
	{5,5,L"    ��Ʒ��Ϣ ",0},
	{5,35,L"��ʾѧ����Ϣ",0},
	{5,65,L"����ѧ����Ϣ",0},
	{5,95,L"ɾ��ѧ����Ϣ",0},
	{5,125,L"�޸�ѧ����Ϣ",0},
	{5,155,L"��ѯѧ����Ϣ",0}
};



/*
��ť
��ҳ   ---  ��ʾ��Ʒ��Ϣ 

��ʾѧ����Ϣ
����ѧ����Ϣ
ɾ��ѧ����Ϣ
�޸�ѧ����Ϣ
��ѯѧ����Ϣ

*/

//���ư�ť
void drawBtn(btNode t) {
	if (t.status == 0)
		setfillcolor(RGB(93, 107, 153));//���������ɫ
	else
		setfillcolor(RGB(204, 213, 240));

	setbkmode(TRANSPARENT);//���ñ���ģʽΪ͸��

	//����Բ�Ǿ�����Ϊ��ť��״

	/*
	x,y
	    x+120,y+20
	
	*/
	fillroundrect(t.x, t.y, t.x + 120, t.y + 20, 10, 10);
	outtextxy(t.x + 15, t.y + 2, t.text);//�����ť��Ϣ

}

//������๦�ܰ�ť��
void drawLeft() {

	SetWorkingImage(&leftImg);//���û��Ƶ�ͼ��
	setbkcolor(RGB(93, 107, 153));//���ñ���ɫ
	
	cleardevice();//���ͼ��

	//���ư�ť
	for (int i = 0;i < 6;i++) {
		drawBtn(leftMen[i]);
	}
	SetWorkingImage();//��ͼ����Ƶ�������
	putimage(0, 0, &leftImg);
}

//�����Ҳ���ʾ��
void drawRight() {
	SetWorkingImage(&rightImg);//���û��Ƶ�ͼ��

	setbkcolor(RGB(247, 249, 254));//���ñ���ɫ
	cleardevice();//���ͼ��
	SetWorkingImage();//��ͼ����Ƶ�������
	putimage(161, 0, &rightImg);
}


// ��ʼ����ఴť״̬
void setUpLeftBtn() {
	int i;
	for (i = 0;i < 6;i++) {
		leftMen[i].status = 0;
	}
}



int main()
{
	initgraph(WIDTH, HEIGHT);	// ������ͼ���ڣ���СΪ 640x480 ����
	int i;
	ExMessage m;//���������Ϣ

	while (1) {

		m = getmessage();//��ȡ�����Ϣ
		if (m.message == WM_LBUTTONDOWN) {
			if (m.x >= 0 && m.x <= 160) {//�������๦�ܰ�����
				for (i = 0;i < 6;i++) {
					//�ж����λ���Ƿ��ڰ�ť��Χ��
					if (m.x >= leftMen[i].x && m.x <= leftMen[i].x + 120 && m.y >= leftMen[i].y && m.y <= leftMen[i].y + 20) {
						setUpLeftBtn();//��ʼ����ť״̬
						leftMen[i].status = 1;//�޸Ķ�Ӧ��ť״̬
					}


				}
				
			}
		}


		drawLeft();//�����������
		drawRight();//�����Ҳ�����
	}


	closegraph();			// �رջ�ͼ����
	return 0;
}