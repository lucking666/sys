# 前言
学生成绩管理系统可以说是C语言程序设计的结课的必备大作业了。花了些时间，费了些头发肝了下，完成了两个系统，一个是控制台版本的，另一个用easyx图形库进行了优化。
先放出完成后的演示图片占个坑。具体的实现过程，等我再梳理下，再慢慢更新整理到论坛上来。

# 演示DEMO
## 基础控制台版本
![image.png](https://bbs.leyuz.net/uploads/202012/24/1608813206162.png)
![输出所有学生信息](https://bbs.leyuz.net/uploads/202012/24/16088044491057.png)
## EasyX带界面版本


![image.png](https://bbs.leyuz.net/uploads/202012/24/16088133328121.png)
![image.png](https://bbs.leyuz.net/uploads/202012/24/16088045388889.png)
![image.png](https://bbs.leyuz.net/uploads/202012/24/16088045476637.png)
![image.png](https://bbs.leyuz.net/uploads/202012/24/16088045547435.png)
![image.png](https://bbs.leyuz.net/uploads/202012/24/16088045584286.png)
![image.png](https://bbs.leyuz.net/uploads/202012/24/16088045631788.png)
# 控制台版本开发过程整理
**更新时间：** 
12-24 
* 初始界面的处理
* 功能分析及框架搭建

12-25

* 文件输入
* 学生信息输出
* 学生信息查询
* 文件信息保存

12-26

* 学生信息

## 开发环境

系统： win10 

IDE: Dev Cpp

## 前置知识
需要掌握基础的C语言知识
* 顺序结构
* 分支结构
* 循环结构
* 数组、字符串
* 函数
* 结构体、指针
* 链表
* 文件操作

## 功能分析
工欲善其事必先利其器，先分析好整体功能和大体的布局再慢慢动手进行代码的实现。
基础设想是，先显示主菜单，通过输入数字选择对应的功能，包括有增加学生信息，删除学生信息，修改学生信息，查询学生信息以及退出程序功能。
![image.png](https://bbs.leyuz.net/uploads/202012/24/16088053255334.png)

## 主菜单界面实现
使用输出语句来实现界面。计划使用数字键来代表各自的功能。
```cpp
//主菜单界面 
void welcome(){
	printf("************************\n");
	printf("**  学生成绩管理系统  **\n");
	printf("**      作者：咸鱼君  **\n");
	printf("**                    **\n");
	printf("**  增加学生信息 ---1 **\n");
	printf("**  删除学生信息 ---2 **\n");
	printf("**  修改学生信息 ---3 **\n");
	printf("**  查询学生信息 ---4 **\n");
	printf("**  输出学生信息 ---5 **\n");
	printf("**  退出管理系统 ---0 **\n");
	
	printf("请输入对应的功能键（数字）: ");
}
```
![image.png](C:\Users\w-yao\Documents\ShareX\Screenshots\2020-12\160881314526.png)

## 功能框架搭建
先处理好整个功能框架，通过输入数字，进行分支判断，不同的数字代表不同的功能。先将要实现的功能做个简易的版本出来，之后再慢慢填充细节。
想的是执行完某些功能后还能继续进行操作，所以将程序放入循环中，并在操作执行完成后，询问继续操作，再根据选择进行处理。

``` c++
int main(){
	int choice=0;
	while(true){
		welcome();
			scanf("%d",&choice);
			switch(choice){
				case 1://增加学生信息 
					addStuInfo(); 
					break;
				case 2://删除学生信息
					deleteStuInfo();
					break;
				case 3://修改学生信息 
					fixStuInfo();
					break;
				case 4://查询学生信息
					searchStuInfo();
					break;
				case 5://输出学生信息
					printStuInfo();
					break;
				case 0://退出程序 
					goodBye();
					break;						
			}
		printf("是否需要继续操作？(yes:1 / no:0 )：");
		scanf("%d",&choice);
		if(choice==0){
			break;
		}
	}
	
	return 0;
}
```
![tLRiEed7FC.gif](https://bbs.leyuz.net/uploads/202012/24/16088158486977.gif)



## 数据结构定义
围绕学生信息进行处理，那么思考学生具有哪些信息。包含，学号，姓名，性别，语文，数学，英语成绩，还有个总分。将对应属性集合在一块，采用结构体方式进行数据操作。并使用链表的方式将数据进行串联。
```c++
typedef struct Node{
	int id;//学号
	char name[30];//姓名 
	char sex[10];//性别
	int ch;//语文
	int ma;//数学
	int en;//英语
	int sum;//总分
	
	struct Node *next;//指向下一个结点 
}node;
```






## 文件数据的读取
这此程序的数据都要以文件的形式进行信息的保存，如果想要在屏幕上输出数据，那么得先读取文件中的信息才行。
C语言中文件读取操作要使用文件指针和相关函数,格式如下。

```c++
FILE *fpr=fopen("文件名","操作方式");
fscanf(fpr,"%d",&intValue);
```
文件名需要加上后缀名，操作方式因为是要从文件中读取信息，所以写r。如果是进行信息的写入则是w。
之后需要将读取的信息以链表的方式组织起来，打算采用尾插法的方式插入数据。

``` c++
// 尾插法 
t->next=s;//链表尾指针 的后一个结点指向新结点 
t=s;//更新尾结点 
t->next=NULL;//链表尾指针 的后一个结点指向NULL 
```
读取函数

```c++
// 文件输入
int readFile(Node *L){
	FILE *fpr=fopen("studentInfo.txt","r");
	node *t=L;
	node st;
	node *s;
	if(fpr==NULL){
		return 0;
	}else{
		
		//fscanf()
		while(fscanf(fpr,"%d %s %s %d %d %d %d",&st.id,st.name,st.sex,&st.ch,&st.ma,&st.en,&st.sum)!=EOF){
			
			s=(node *)malloc(sizeof(node));
			*s=st;
			
			// 尾插法 
			t->next=s;//链表尾指针 的后一个结点指向新结点 
			t=s;//更新尾结点 
			t->next=NULL;//链表尾指针 的后一个结点指向NULL 
			
		}
	}
	fclose(fpr);//关闭文件指针
	return 1;
}
```

## 输出所有学生信息
接下来完成所有学生信息的输出。此处需要考察链表的遍历。

``` cpp
void printStuInfo(node *L){
	 system("cls");
	 node *p=L->next;
	 printf("________________________________________________________\n");
	 printf("|学号\t|姓名\t|性别\t|语文\t|数学\t|英语\t|总分\t|\n");
	 printf("________________________________________________________\n");
	 if(p!=NULL){
	 	
	 	while(p!=NULL){
			printf("%d|%s\t|%s\t|%d\t|%d\t|%d\t|%d\t|\n",p->id,p->name,p->sex,p->ch,p->ma,p->en,p->sum);
			printf("________________________________________________________\n");
			p=p->next;
		}
	 }
}
```

![image-20201225155709522](https://gitee.com/wyloving/picCloud/raw/master/image-20201225155709522.png)

## 增加学生信息

接下来是增加学生的信息，此处采用头插法将链表结点进行插入。将学生信息的增加分成了两部分，一部分是界面的打印，一部分是底层数据的处理。

![image-20201225180440775](https://gitee.com/wyloving/picCloud/raw/master/image-20201225180440775.png)

界面实现：

``` cpp
//增加学生信息
void printAddStuInfo(){
	// 
	system("cls");
	node st;
	printf("请输入新增学生相关信息\n");
	printf("学号:");
	scanf("%d",&st.id);
	printf("姓名：");
	scanf("%s",st.name);
	printf("性别:");
	scanf("%s",st.sex);
	printf("语文:");
	scanf("%d",&st.ch);
	printf("数学:");
	scanf("%d",&st.ma);
	printf("英语:");
	scanf("%d",&st.en);
	st.sum=st.ch+st.ma+st.en;
	
	insertStuInfo(&List,st);
	 
}
```

功能实现:

```cpp
void insertStuInfo(node *L,node e){
	//头插法
	node *h=L;
	node *s=(node *)malloc(sizeof(node));
	*s=e;
	
	s->next=h->next;
	h->next=s;
}
```



![image-20201225161643690](https://gitee.com/wyloving/picCloud/raw/master/image-20201225161643690.png)



![image-20201225161603787](https://gitee.com/wyloving/picCloud/raw/master/image-20201225161603787.png)

## 文件数据的写入

这部分和文件的读取部分相似，思路是将整个链表内容存储到文件中。

使用fprintf()将文件信息进行存储。

```cpp
//保存文件
int saveFile(node *L){
	FILE *fpw=fopen("studentInfo.txt","w");
	if(fpw==NULL) return 0;
	node *p=L->next;	
	while(p!=NULL){
		fprintf(fpw,"%d %s %s %d %d %d %d\n",p->id,p->name,p->sex,p->ch,p->ma,p->en,p->sum);
		p=p->next;
	}
    fclose(fpw);//关闭文件指针
	return 1; 
}
```

再在学生信息的增加过程中添加文件数据的保存操作。

```cpp
void insertStuInfo(node *L,node e){
	//头插法
	node *h=L;
	node *s=(node *)malloc(sizeof(node));
	*s=e;
	
	s->next=h->next;
	h->next=s;
	
	//保存文件 
	saveFile(L);
}
```

![6nW6K36jsj](https://gitee.com/wyloving/picCloud/raw/master/6nW6K36jsj.png)

![Ne41ND5ZLg](https://gitee.com/wyloving/picCloud/raw/master/Ne41ND5ZLg.png)

## 学生信息查询

接下来是实现学生信息查询功能，计划也是页面输出部分与逻辑实现部分进行分离。打算，可以通过学号与姓名两个关键值进行信息的查找。因为是链表结构，为了方便之后的操作，逻辑函数会返回查找到的学生信息的前一个结点位置，这样的话也能在删除学生信息与修改学生信息中进行函数的复用了。

界面实现：

```cpp
//查询学生信息
void printSearchStuInfo(node *L){
	system("cls");
	int choice=0;
	int id;
	char name[50];
	node *st;
	printf("按学号查询----- 1\n");
	printf("按姓名查询----- 2\n");
	printf("请输入查询方式：");
	scanf("%d",&choice);
	
	if(choice == 1){
		printf("请输入要查询的学号：");
		scanf("%d",&id);
		st=searchStuInfoById(id,L);
		
		if(st==NULL){
			printf("查无此人！\n");
		}else{
			st=st->next;
			printf("________________________________________________________\n");
			printf("|学号\t|姓名\t|性别\t|语文\t|数学\t|英语\t|总分\t|\n");
			printf("________________________________________________________\n");
			printf("%d|%s\t|%s\t|%d\t|%d\t|%d\t|%d\t|\n",st->id,st->name,st->sex,st->ch,st->ma,st->en,st->sum);
			printf("________________________________________________________\n");
		}
	}else if(choice ==2){
		printf("请输入要查询的姓名：");
			scanf("%s",name);
			st=searchStuInfoByName(name,L);
			
			if(st==NULL){
				printf("查无此人！\n");
			}else{
				st=st->next;
				printf("________________________________________________________\n");
				printf("|学号\t|姓名\t|性别\t|语文\t|数学\t|英语\t|总分\t|\n");
				printf("________________________________________________________\n");
				printf("%d|%s\t|%s\t|%d\t|%d\t|%d\t|%d\t|\n",st->id,st->name,st->sex,st->ch,st->ma,st->en,st->sum);
				printf("________________________________________________________\n");
			}
	}
	
}
```

逻辑实现：

思路是遍历整个链表，逐一对关键信息进行比较。

按学号进行查找，找不到返回NULL，找到了返回前一个结点位置

```cpp
//按学号进行查找 
node * searchStuInfoById(int id,node *L){
	
	node *p=L;
	
	while(p->next!=NULL){
		
		if(p->next->id==id){
			return p;
		}
		
		p=p->next;
	}
	
	return NULL;
}
```

按姓名进行查找，找不到返回NULL，找到了返回前一个结点位置

```cpp
//按姓名进行查找 
node * searchStuInfoByName(char name[],node *L){
	node *p=L;
	
	while(p->next!=NULL){
		
		if(strcmp(name,p->next->name)==0){
			return p;
		}
		
		p=p->next;
	}
	
	return NULL;
}
```

![4aoMYPA7rj](https://gitee.com/wyloving/picCloud/raw/master/4aoMYPA7rj.png)

![Mk9h8hz0gF](https://gitee.com/wyloving/picCloud/raw/master/Mk9h8hz0gF.png)

## 学生信息修改

依旧是分成两部分，先输出界面，过程逻辑的话就沿用学生信息查询的部分。实现逻辑是这样的：先查到要查询的学生信息，在对信息修改，改完了再保存到文件中。

页面和实现部分：

```cpp
//修改学生信息
void printFixStuInfo(node *L){
	system("cls");
	int id;
	int choice=-1;
	
	printf("请输入要查找的学生学号");
	scanf("%d",&id);
	node *st=searchStuInfoById(id,L);
	
	if(st==NULL){
		printf("查无此人！");
		return;
	}
    
	st=st->next; 
	
	while(1){
		system("cls"); 
		printf("________________________________________________________\n");
		printf("|学号\t|姓名\t|性别\t|语文\t|数学\t|英语\t|总分\t|\n");
		printf("________________________________________________________\n");
		printf("%d|%s\t|%s\t|%d\t|%d\t|%d\t|%d\t|\n",st->id,st->name,st->sex,st->ch,st->ma,st->en,st->sum);
		printf("________________________________________________________\n");
		printf("修改姓名---- 1\n");
		printf("修改性别---- 2\n");
		printf("修改语文---- 3\n");
		printf("修改数学---- 4\n");
		printf("修改英语---- 5\n");
		
		printf("请输入要修改的信息: ");
		scanf("%d",&choice);
		
		switch(choice){
			case 1:
				printf("请输入姓名：");
				scanf("%s",st->name);
				break;
			case 2:
				printf("请输入性别：");
				scanf("%s",st->sex);
				break;
			case 3:
				printf("请输入语文：");
				scanf("%d",&st->ch);
				break;
			case 4:
				printf("请输入数学：");
				scanf("%d",&st->ma);				
				break;
			case 5:
				printf("请输入英语：");
				scanf("%d",&st->en);				
				break;
		}
		st->sum=st->ch+st->ma+st->en; 
		printf("是否继续修改学生信息?（y-1 / n-0）\n");
		scanf("%d",&choice);
		if(choice == 0){
			break;
		}
	}
	
	printf("________________________________________________________\n");
	printf("|学号\t|姓名\t|性别\t|语文\t|数学\t|英语\t|总分\t|\n");
	printf("________________________________________________________\n");
	printf("%d|%s\t|%s\t|%d\t|%d\t|%d\t|%d\t|\n",st->id,st->name,st->sex,st->ch,st->ma,st->en,st->sum);
	printf("________________________________________________________\n");
	
	//保存文件信息
	saveFile(L);
}

```

![image-20201226175035839](https://gitee.com/wyloving/picCloud/raw/master/image-20201226175035839.png)

![image-20201226175102218](https://gitee.com/wyloving/picCloud/raw/master/image-20201226175102218.png)

## 学生信息删除

接下来实现学生信息删除部分。页面部分输出提示，之后输入学号查询要删除的学生信息。利用之前实现的查询信息的函数得到结点位置，之后再根据位置删除对应的结点，再将修改后的信息保存至文件中去。

页面部分

``` cpp
//删除学生信息
void printDeleteStuInfo(node *L){
	system("cls");
	int id;
	
	node *p;
	
	printf("请输入要查找的学生学号");
	scanf("%d",&id);
	node *st=searchStuInfoById(id,L);
	p=st;
	
	if(st==NULL){
		printf("查无此人！");
		return;
	}
	
	st=st->next; 
	printf("________________________________________________________\n");
	printf("|学号\t|姓名\t|性别\t|语文\t|数学\t|英语\t|总分\t|\n");
	printf("________________________________________________________\n");
	printf("%d|%s\t|%s\t|%d\t|%d\t|%d\t|%d\t|\n",st->id,st->name,st->sex,st->ch,st->ma,st->en,st->sum);
	printf("________________________________________________________\n");
	
	deleteStuInfo(p);
	saveFile(L);	
 	
}
```

结点删除部分

``` cpp
//删除学生信息
void deleteStuInfo(node *pr){
	node *s=pr->next;
	
	pr->next=s->next;
	s->next=NULL;
	
	free(s);//释放结点空间 
	
}
```

![image-20201226210348062](https://gitee.com/wyloving/picCloud/raw/master/image-20201226210348062.png)

![70euw6DARL](https://gitee.com/wyloving/picCloud/raw/master/70euw6DARL.png)

![image-20201226210102154](https://gitee.com/wyloving/picCloud/raw/master/image-20201226210102154.png)

## 结束界面

到此为止，这个小系统的基础功能都已经完成了。接下来把结束界面处理下。

```cpp
//退出程序
void goodBye(){
	system("cls");
	printf("欢迎下次使用~\n");
	exit(0);//结束程序 
}
```

![ConsolePauser_F0B8oTbtcU](https://gitee.com/wyloving/picCloud/raw/master/ConsolePauser_F0B8oTbtcU.png)

学生成绩管理系统V1.0版本到此也就完结了。之后也可以在此基础上进行其他功能的开发，比如，程序排序，最高分，平均分等等的处理。大体都是围绕链表的遍历，插入和删除操作来执行的。再结合自己对界面的设计即可。



