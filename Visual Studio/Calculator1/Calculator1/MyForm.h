#pragma once

namespace Calculator1 {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// MyForm 摘要
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		MyForm(void)
		{
			InitializeComponent();
			//
			//TODO:  在此处添加构造函数代码
			//
		}

	protected:
		/// <summary>
		/// 清理所有正在使用的资源。
		/// </summary>
		~MyForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Button^ button1;
	private: System::Windows::Forms::TextBox^ textBox1;
	private: System::Windows::Forms::Button^ button2;
	private: System::Windows::Forms::Button^ button3;
	private: System::Windows::Forms::Button^ button4;
	private: System::Windows::Forms::Button^ button5;
	private: System::Windows::Forms::Button^ button6;
	private: System::Windows::Forms::Button^ button7;
	private: System::Windows::Forms::Button^ button8;
	private: System::Windows::Forms::Button^ button9;
	private: System::Windows::Forms::Button^ button10;
	private: System::Windows::Forms::Button^ button11;
	private: System::Windows::Forms::Button^ button12;
	private: System::Windows::Forms::Button^ button13;
	private: System::Windows::Forms::Button^ button14;
	private: System::Windows::Forms::Button^ button15;
	private: System::Windows::Forms::Button^ button16;
	private: System::Windows::Forms::Button^ button17;
	private: System::Windows::Forms::Button^ button18;
	private: System::Windows::Forms::Button^ button19;
	private: System::Windows::Forms::Button^ button20;
	private: System::Windows::Forms::MenuStrip^ menuStrip3;
	private: System::Windows::Forms::ToolStripMenuItem^ fILEToolStripMenuItem1;
	private: System::Windows::Forms::ToolStripMenuItem^ standardToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ scienceToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ exitToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ hELPToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ hISTORYToolStripMenuItem;
	private: System::Windows::Forms::Button^ button21;
	private: System::Windows::Forms::Button^ button22;
	private: System::Windows::Forms::Button^ button23;
	private: System::Windows::Forms::Button^ button24;
	private: System::Windows::Forms::Button^ button25;
	private: System::Windows::Forms::Button^ button26;
	private: System::Windows::Forms::Button^ button27;
	private: System::Windows::Forms::Button^ button28;
	private: System::Windows::Forms::Button^ button29;
	private: System::Windows::Forms::Button^ button30;
	private: System::Windows::Forms::Button^ button31;
	private: System::Windows::Forms::Button^ button32;
	private: System::Windows::Forms::Button^ button33;
	private: System::Windows::Forms::Button^ button34;
	private: System::Windows::Forms::Button^ button35;
	private: System::Windows::Forms::Button^ button36;
	private: System::Windows::Forms::Button^ button37;
	private: System::Windows::Forms::Button^ button38;
	private: System::Windows::Forms::Button^ button39;
	private: System::Windows::Forms::Button^ button40;

	protected:

	private:
		/// <summary>
		/// 必需的设计器变量。
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// 设计器支持所需的方法 - 不要修改
		/// 使用代码编辑器修改此方法的内容。
		/// </summary>
		void InitializeComponent(void)
		{
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->button3 = (gcnew System::Windows::Forms::Button());
			this->button4 = (gcnew System::Windows::Forms::Button());
			this->button5 = (gcnew System::Windows::Forms::Button());
			this->button6 = (gcnew System::Windows::Forms::Button());
			this->button7 = (gcnew System::Windows::Forms::Button());
			this->button8 = (gcnew System::Windows::Forms::Button());
			this->button9 = (gcnew System::Windows::Forms::Button());
			this->button10 = (gcnew System::Windows::Forms::Button());
			this->button11 = (gcnew System::Windows::Forms::Button());
			this->button12 = (gcnew System::Windows::Forms::Button());
			this->button13 = (gcnew System::Windows::Forms::Button());
			this->button14 = (gcnew System::Windows::Forms::Button());
			this->button15 = (gcnew System::Windows::Forms::Button());
			this->button16 = (gcnew System::Windows::Forms::Button());
			this->button17 = (gcnew System::Windows::Forms::Button());
			this->button18 = (gcnew System::Windows::Forms::Button());
			this->button19 = (gcnew System::Windows::Forms::Button());
			this->button20 = (gcnew System::Windows::Forms::Button());
			this->menuStrip3 = (gcnew System::Windows::Forms::MenuStrip());
			this->fILEToolStripMenuItem1 = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->standardToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->scienceToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->exitToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->hELPToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->hISTORYToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->button21 = (gcnew System::Windows::Forms::Button());
			this->button22 = (gcnew System::Windows::Forms::Button());
			this->button23 = (gcnew System::Windows::Forms::Button());
			this->button24 = (gcnew System::Windows::Forms::Button());
			this->button25 = (gcnew System::Windows::Forms::Button());
			this->button26 = (gcnew System::Windows::Forms::Button());
			this->button27 = (gcnew System::Windows::Forms::Button());
			this->button28 = (gcnew System::Windows::Forms::Button());
			this->button29 = (gcnew System::Windows::Forms::Button());
			this->button30 = (gcnew System::Windows::Forms::Button());
			this->button31 = (gcnew System::Windows::Forms::Button());
			this->button32 = (gcnew System::Windows::Forms::Button());
			this->button33 = (gcnew System::Windows::Forms::Button());
			this->button34 = (gcnew System::Windows::Forms::Button());
			this->button35 = (gcnew System::Windows::Forms::Button());
			this->button36 = (gcnew System::Windows::Forms::Button());
			this->button37 = (gcnew System::Windows::Forms::Button());
			this->button38 = (gcnew System::Windows::Forms::Button());
			this->button39 = (gcnew System::Windows::Forms::Button());
			this->button40 = (gcnew System::Windows::Forms::Button());
			this->menuStrip3->SuspendLayout();
			this->SuspendLayout();
			// 
			// button1
			// 
			this->button1->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button1->Location = System::Drawing::Point(15, 176);
			this->button1->Margin = System::Windows::Forms::Padding(6);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(142, 130);
			this->button1->TabIndex = 0;
			this->button1->Text = L"×";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &MyForm::button1_Click);
			// 
			// textBox1
			// 
			this->textBox1->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->textBox1->Location = System::Drawing::Point(15, 62);
			this->textBox1->Margin = System::Windows::Forms::Padding(6);
			this->textBox1->Multiline = true;
			this->textBox1->Name = L"textBox1";
			this->textBox1->RightToLeft = System::Windows::Forms::RightToLeft::Yes;
			this->textBox1->Size = System::Drawing::Size(600, 98);
			this->textBox1->TabIndex = 1;
			this->textBox1->TabStop = false;
			this->textBox1->Text = L"0";
			this->textBox1->TextChanged += gcnew System::EventHandler(this, &MyForm::textBox1_TextChanged);
			// 
			// button2
			// 
			this->button2->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button2->Location = System::Drawing::Point(169, 176);
			this->button2->Margin = System::Windows::Forms::Padding(6);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(142, 130);
			this->button2->TabIndex = 0;
			this->button2->Text = L"C";
			this->button2->UseVisualStyleBackColor = true;
			this->button2->Click += gcnew System::EventHandler(this, &MyForm::button2_Click);
			// 
			// button3
			// 
			this->button3->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button3->Location = System::Drawing::Point(323, 176);
			this->button3->Margin = System::Windows::Forms::Padding(6);
			this->button3->Name = L"button3";
			this->button3->Size = System::Drawing::Size(142, 130);
			this->button3->TabIndex = 0;
			this->button3->Text = L"CE";
			this->button3->UseVisualStyleBackColor = true;
			this->button3->Click += gcnew System::EventHandler(this, &MyForm::button3_Click);
			// 
			// button4
			// 
			this->button4->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button4->Location = System::Drawing::Point(477, 176);
			this->button4->Margin = System::Windows::Forms::Padding(6);
			this->button4->Name = L"button4";
			this->button4->Size = System::Drawing::Size(142, 130);
			this->button4->TabIndex = 0;
			this->button4->Text = L"±";
			this->button4->UseVisualStyleBackColor = true;
			this->button4->Click += gcnew System::EventHandler(this, &MyForm::button4_Click);
			// 
			// button5
			// 
			this->button5->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button5->Location = System::Drawing::Point(15, 318);
			this->button5->Margin = System::Windows::Forms::Padding(6);
			this->button5->Name = L"button5";
			this->button5->Size = System::Drawing::Size(142, 130);
			this->button5->TabIndex = 0;
			this->button5->Text = L"7";
			this->button5->UseVisualStyleBackColor = true;
			this->button5->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button6
			// 
			this->button6->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button6->Location = System::Drawing::Point(169, 318);
			this->button6->Margin = System::Windows::Forms::Padding(6);
			this->button6->Name = L"button6";
			this->button6->Size = System::Drawing::Size(142, 130);
			this->button6->TabIndex = 0;
			this->button6->Text = L"8";
			this->button6->UseVisualStyleBackColor = true;
			this->button6->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button7
			// 
			this->button7->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button7->Location = System::Drawing::Point(323, 318);
			this->button7->Margin = System::Windows::Forms::Padding(6);
			this->button7->Name = L"button7";
			this->button7->Size = System::Drawing::Size(142, 130);
			this->button7->TabIndex = 0;
			this->button7->Text = L"9";
			this->button7->UseVisualStyleBackColor = true;
			this->button7->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button8
			// 
			this->button8->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button8->Location = System::Drawing::Point(477, 318);
			this->button8->Margin = System::Windows::Forms::Padding(6);
			this->button8->Name = L"button8";
			this->button8->Size = System::Drawing::Size(142, 130);
			this->button8->TabIndex = 0;
			this->button8->Text = L"+";
			this->button8->UseVisualStyleBackColor = true;
			this->button8->Click += gcnew System::EventHandler(this, &MyForm::EnterOperator);
			// 
			// button9
			// 
			this->button9->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button9->Location = System::Drawing::Point(15, 460);
			this->button9->Margin = System::Windows::Forms::Padding(6);
			this->button9->Name = L"button9";
			this->button9->Size = System::Drawing::Size(142, 130);
			this->button9->TabIndex = 0;
			this->button9->Text = L"4";
			this->button9->UseVisualStyleBackColor = true;
			this->button9->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button10
			// 
			this->button10->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button10->Location = System::Drawing::Point(169, 460);
			this->button10->Margin = System::Windows::Forms::Padding(6);
			this->button10->Name = L"button10";
			this->button10->Size = System::Drawing::Size(142, 130);
			this->button10->TabIndex = 0;
			this->button10->TabStop = false;
			this->button10->Text = L"5";
			this->button10->UseVisualStyleBackColor = true;
			this->button10->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button11
			// 
			this->button11->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button11->Location = System::Drawing::Point(323, 460);
			this->button11->Margin = System::Windows::Forms::Padding(6);
			this->button11->Name = L"button11";
			this->button11->Size = System::Drawing::Size(142, 130);
			this->button11->TabIndex = 0;
			this->button11->Text = L"6";
			this->button11->UseVisualStyleBackColor = true;
			this->button11->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button12
			// 
			this->button12->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button12->Location = System::Drawing::Point(477, 460);
			this->button12->Margin = System::Windows::Forms::Padding(6);
			this->button12->Name = L"button12";
			this->button12->Size = System::Drawing::Size(142, 130);
			this->button12->TabIndex = 0;
			this->button12->Text = L"-";
			this->button12->UseVisualStyleBackColor = true;
			this->button12->Click += gcnew System::EventHandler(this, &MyForm::EnterOperator);
			// 
			// button13
			// 
			this->button13->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button13->Location = System::Drawing::Point(15, 602);
			this->button13->Margin = System::Windows::Forms::Padding(6);
			this->button13->Name = L"button13";
			this->button13->Size = System::Drawing::Size(142, 130);
			this->button13->TabIndex = 0;
			this->button13->Text = L"1";
			this->button13->UseVisualStyleBackColor = true;
			this->button13->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button14
			// 
			this->button14->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button14->Location = System::Drawing::Point(169, 602);
			this->button14->Margin = System::Windows::Forms::Padding(6);
			this->button14->Name = L"button14";
			this->button14->Size = System::Drawing::Size(142, 130);
			this->button14->TabIndex = 0;
			this->button14->Text = L"2";
			this->button14->UseVisualStyleBackColor = true;
			this->button14->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button15
			// 
			this->button15->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button15->Location = System::Drawing::Point(323, 602);
			this->button15->Margin = System::Windows::Forms::Padding(6);
			this->button15->Name = L"button15";
			this->button15->Size = System::Drawing::Size(142, 130);
			this->button15->TabIndex = 0;
			this->button15->Text = L"3";
			this->button15->UseVisualStyleBackColor = true;
			this->button15->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button16
			// 
			this->button16->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button16->Location = System::Drawing::Point(477, 602);
			this->button16->Margin = System::Windows::Forms::Padding(6);
			this->button16->Name = L"button16";
			this->button16->Size = System::Drawing::Size(142, 130);
			this->button16->TabIndex = 0;
			this->button16->Text = L"*";
			this->button16->UseVisualStyleBackColor = true;
			this->button16->Click += gcnew System::EventHandler(this, &MyForm::EnterOperator);
			// 
			// button17
			// 
			this->button17->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button17->Location = System::Drawing::Point(15, 744);
			this->button17->Margin = System::Windows::Forms::Padding(6);
			this->button17->Name = L"button17";
			this->button17->Size = System::Drawing::Size(142, 130);
			this->button17->TabIndex = 0;
			this->button17->Text = L"0";
			this->button17->UseVisualStyleBackColor = true;
			this->button17->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button18
			// 
			this->button18->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button18->Location = System::Drawing::Point(169, 744);
			this->button18->Margin = System::Windows::Forms::Padding(6);
			this->button18->Name = L"button18";
			this->button18->Size = System::Drawing::Size(142, 130);
			this->button18->TabIndex = 0;
			this->button18->Text = L".";
			this->button18->UseVisualStyleBackColor = true;
			this->button18->Click += gcnew System::EventHandler(this, &MyForm::button18_Click);
			// 
			// button19
			// 
			this->button19->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button19->Location = System::Drawing::Point(323, 744);
			this->button19->Margin = System::Windows::Forms::Padding(6);
			this->button19->Name = L"button19";
			this->button19->Size = System::Drawing::Size(142, 130);
			this->button19->TabIndex = 0;
			this->button19->Text = L"=";
			this->button19->UseVisualStyleBackColor = true;
			this->button19->Click += gcnew System::EventHandler(this, &MyForm::button19_Click);
			// 
			// button20
			// 
			this->button20->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button20->Location = System::Drawing::Point(477, 744);
			this->button20->Margin = System::Windows::Forms::Padding(6);
			this->button20->Name = L"button20";
			this->button20->Size = System::Drawing::Size(142, 130);
			this->button20->TabIndex = 0;
			this->button20->Text = L"/";
			this->button20->UseVisualStyleBackColor = true;
			this->button20->Click += gcnew System::EventHandler(this, &MyForm::EnterOperator);
			// 
			// menuStrip3
			// 
			this->menuStrip3->GripMargin = System::Windows::Forms::Padding(2, 2, 0, 2);
			this->menuStrip3->ImageScalingSize = System::Drawing::Size(32, 32);
			this->menuStrip3->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {
				this->fILEToolStripMenuItem1,
					this->hELPToolStripMenuItem, this->hISTORYToolStripMenuItem
			});
			this->menuStrip3->Location = System::Drawing::Point(0, 0);
			this->menuStrip3->Name = L"menuStrip3";
			this->menuStrip3->Size = System::Drawing::Size(2869, 64);
			this->menuStrip3->TabIndex = 4;
			this->menuStrip3->Text = L"menuStrip3";
			// 
			// fILEToolStripMenuItem1
			// 
			this->fILEToolStripMenuItem1->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {
				this->standardToolStripMenuItem,
					this->scienceToolStripMenuItem, this->exitToolStripMenuItem
			});
			this->fILEToolStripMenuItem1->Name = L"fILEToolStripMenuItem1";
			this->fILEToolStripMenuItem1->Size = System::Drawing::Size(79, 35);
			this->fILEToolStripMenuItem1->Text = L"FILE";
			// 
			// standardToolStripMenuItem
			// 
			this->standardToolStripMenuItem->Name = L"standardToolStripMenuItem";
			this->standardToolStripMenuItem->Size = System::Drawing::Size(359, 44);
			this->standardToolStripMenuItem->Text = L"standard";
			// 
			// scienceToolStripMenuItem
			// 
			this->scienceToolStripMenuItem->Name = L"scienceToolStripMenuItem";
			this->scienceToolStripMenuItem->Size = System::Drawing::Size(359, 44);
			this->scienceToolStripMenuItem->Text = L"science";
			// 
			// exitToolStripMenuItem
			// 
			this->exitToolStripMenuItem->Name = L"exitToolStripMenuItem";
			this->exitToolStripMenuItem->Size = System::Drawing::Size(359, 44);
			this->exitToolStripMenuItem->Text = L"exit";
			// 
			// hELPToolStripMenuItem
			// 
			this->hELPToolStripMenuItem->Name = L"hELPToolStripMenuItem";
			this->hELPToolStripMenuItem->Size = System::Drawing::Size(93, 35);
			this->hELPToolStripMenuItem->Text = L"HELP";
			// 
			// hISTORYToolStripMenuItem
			// 
			this->hISTORYToolStripMenuItem->Name = L"hISTORYToolStripMenuItem";
			this->hISTORYToolStripMenuItem->Size = System::Drawing::Size(138, 35);
			this->hISTORYToolStripMenuItem->Text = L"HISTORY";
			// 
			// button21
			// 
			this->button21->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button21->Location = System::Drawing::Point(670, 176);
			this->button21->Margin = System::Windows::Forms::Padding(6);
			this->button21->Name = L"button21";
			this->button21->Size = System::Drawing::Size(142, 130);
			this->button21->TabIndex = 0;
			this->button21->Text = L"×";
			this->button21->UseVisualStyleBackColor = true;
			this->button21->Click += gcnew System::EventHandler(this, &MyForm::button1_Click);
			// 
			// button22
			// 
			this->button22->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button22->Location = System::Drawing::Point(670, 318);
			this->button22->Margin = System::Windows::Forms::Padding(6);
			this->button22->Name = L"button22";
			this->button22->Size = System::Drawing::Size(142, 130);
			this->button22->TabIndex = 0;
			this->button22->Text = L"7";
			this->button22->UseVisualStyleBackColor = true;
			this->button22->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button23
			// 
			this->button23->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button23->Location = System::Drawing::Point(670, 460);
			this->button23->Margin = System::Windows::Forms::Padding(6);
			this->button23->Name = L"button23";
			this->button23->Size = System::Drawing::Size(142, 130);
			this->button23->TabIndex = 0;
			this->button23->Text = L"4";
			this->button23->UseVisualStyleBackColor = true;
			this->button23->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button24
			// 
			this->button24->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button24->Location = System::Drawing::Point(670, 602);
			this->button24->Margin = System::Windows::Forms::Padding(6);
			this->button24->Name = L"button24";
			this->button24->Size = System::Drawing::Size(142, 130);
			this->button24->TabIndex = 0;
			this->button24->Text = L"1";
			this->button24->UseVisualStyleBackColor = true;
			this->button24->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button25
			// 
			this->button25->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button25->Location = System::Drawing::Point(670, 744);
			this->button25->Margin = System::Windows::Forms::Padding(6);
			this->button25->Name = L"button25";
			this->button25->Size = System::Drawing::Size(142, 130);
			this->button25->TabIndex = 0;
			this->button25->Text = L"0";
			this->button25->UseVisualStyleBackColor = true;
			this->button25->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button26
			// 
			this->button26->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button26->Location = System::Drawing::Point(824, 176);
			this->button26->Margin = System::Windows::Forms::Padding(6);
			this->button26->Name = L"button26";
			this->button26->Size = System::Drawing::Size(142, 130);
			this->button26->TabIndex = 0;
			this->button26->Text = L"C";
			this->button26->UseVisualStyleBackColor = true;
			this->button26->Click += gcnew System::EventHandler(this, &MyForm::button2_Click);
			// 
			// button27
			// 
			this->button27->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button27->Location = System::Drawing::Point(824, 318);
			this->button27->Margin = System::Windows::Forms::Padding(6);
			this->button27->Name = L"button27";
			this->button27->Size = System::Drawing::Size(142, 130);
			this->button27->TabIndex = 0;
			this->button27->Text = L"8";
			this->button27->UseVisualStyleBackColor = true;
			this->button27->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button28
			// 
			this->button28->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button28->Location = System::Drawing::Point(824, 460);
			this->button28->Margin = System::Windows::Forms::Padding(6);
			this->button28->Name = L"button28";
			this->button28->Size = System::Drawing::Size(142, 130);
			this->button28->TabIndex = 0;
			this->button28->TabStop = false;
			this->button28->Text = L"5";
			this->button28->UseVisualStyleBackColor = true;
			this->button28->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button29
			// 
			this->button29->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button29->Location = System::Drawing::Point(824, 602);
			this->button29->Margin = System::Windows::Forms::Padding(6);
			this->button29->Name = L"button29";
			this->button29->Size = System::Drawing::Size(142, 130);
			this->button29->TabIndex = 0;
			this->button29->Text = L"2";
			this->button29->UseVisualStyleBackColor = true;
			this->button29->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button30
			// 
			this->button30->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button30->Location = System::Drawing::Point(824, 744);
			this->button30->Margin = System::Windows::Forms::Padding(6);
			this->button30->Name = L"button30";
			this->button30->Size = System::Drawing::Size(142, 130);
			this->button30->TabIndex = 0;
			this->button30->Text = L".";
			this->button30->UseVisualStyleBackColor = true;
			this->button30->Click += gcnew System::EventHandler(this, &MyForm::button18_Click);
			// 
			// button31
			// 
			this->button31->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button31->Location = System::Drawing::Point(978, 176);
			this->button31->Margin = System::Windows::Forms::Padding(6);
			this->button31->Name = L"button31";
			this->button31->Size = System::Drawing::Size(142, 130);
			this->button31->TabIndex = 0;
			this->button31->Text = L"CE";
			this->button31->UseVisualStyleBackColor = true;
			this->button31->Click += gcnew System::EventHandler(this, &MyForm::button3_Click);
			// 
			// button32
			// 
			this->button32->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button32->Location = System::Drawing::Point(978, 318);
			this->button32->Margin = System::Windows::Forms::Padding(6);
			this->button32->Name = L"button32";
			this->button32->Size = System::Drawing::Size(142, 130);
			this->button32->TabIndex = 0;
			this->button32->Text = L"9";
			this->button32->UseVisualStyleBackColor = true;
			this->button32->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button33
			// 
			this->button33->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button33->Location = System::Drawing::Point(978, 460);
			this->button33->Margin = System::Windows::Forms::Padding(6);
			this->button33->Name = L"button33";
			this->button33->Size = System::Drawing::Size(142, 130);
			this->button33->TabIndex = 0;
			this->button33->Text = L"6";
			this->button33->UseVisualStyleBackColor = true;
			this->button33->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button34
			// 
			this->button34->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button34->Location = System::Drawing::Point(978, 602);
			this->button34->Margin = System::Windows::Forms::Padding(6);
			this->button34->Name = L"button34";
			this->button34->Size = System::Drawing::Size(142, 130);
			this->button34->TabIndex = 0;
			this->button34->Text = L"3";
			this->button34->UseVisualStyleBackColor = true;
			this->button34->Click += gcnew System::EventHandler(this, &MyForm::EnterNumber);
			// 
			// button35
			// 
			this->button35->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button35->Location = System::Drawing::Point(978, 744);
			this->button35->Margin = System::Windows::Forms::Padding(6);
			this->button35->Name = L"button35";
			this->button35->Size = System::Drawing::Size(142, 130);
			this->button35->TabIndex = 0;
			this->button35->Text = L"=";
			this->button35->UseVisualStyleBackColor = true;
			this->button35->Click += gcnew System::EventHandler(this, &MyForm::button19_Click);
			// 
			// button36
			// 
			this->button36->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button36->Location = System::Drawing::Point(1132, 176);
			this->button36->Margin = System::Windows::Forms::Padding(6);
			this->button36->Name = L"button36";
			this->button36->Size = System::Drawing::Size(142, 130);
			this->button36->TabIndex = 0;
			this->button36->Text = L"±";
			this->button36->UseVisualStyleBackColor = true;
			this->button36->Click += gcnew System::EventHandler(this, &MyForm::button4_Click);
			// 
			// button37
			// 
			this->button37->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button37->Location = System::Drawing::Point(1132, 318);
			this->button37->Margin = System::Windows::Forms::Padding(6);
			this->button37->Name = L"button37";
			this->button37->Size = System::Drawing::Size(142, 130);
			this->button37->TabIndex = 0;
			this->button37->Text = L"+";
			this->button37->UseVisualStyleBackColor = true;
			this->button37->Click += gcnew System::EventHandler(this, &MyForm::EnterOperator);
			// 
			// button38
			// 
			this->button38->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button38->Location = System::Drawing::Point(1132, 460);
			this->button38->Margin = System::Windows::Forms::Padding(6);
			this->button38->Name = L"button38";
			this->button38->Size = System::Drawing::Size(142, 130);
			this->button38->TabIndex = 0;
			this->button38->Text = L"-";
			this->button38->UseVisualStyleBackColor = true;
			this->button38->Click += gcnew System::EventHandler(this, &MyForm::EnterOperator);
			// 
			// button39
			// 
			this->button39->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button39->Location = System::Drawing::Point(1132, 602);
			this->button39->Margin = System::Windows::Forms::Padding(6);
			this->button39->Name = L"button39";
			this->button39->Size = System::Drawing::Size(142, 130);
			this->button39->TabIndex = 0;
			this->button39->Text = L"*";
			this->button39->UseVisualStyleBackColor = true;
			this->button39->Click += gcnew System::EventHandler(this, &MyForm::EnterOperator);
			// 
			// button40
			// 
			this->button40->Font = (gcnew System::Drawing::Font(L"隶书", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(134)));
			this->button40->Location = System::Drawing::Point(1132, 744);
			this->button40->Margin = System::Windows::Forms::Padding(6);
			this->button40->Name = L"button40";
			this->button40->Size = System::Drawing::Size(142, 130);
			this->button40->TabIndex = 0;
			this->button40->Text = L"/";
			this->button40->UseVisualStyleBackColor = true;
			this->button40->Click += gcnew System::EventHandler(this, &MyForm::EnterOperator);
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(12, 24);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(2152, 891);
			this->Controls->Add(this->textBox1);
			this->Controls->Add(this->button40);
			this->Controls->Add(this->button20);
			this->Controls->Add(this->button39);
			this->Controls->Add(this->button16);
			this->Controls->Add(this->button38);
			this->Controls->Add(this->button12);
			this->Controls->Add(this->button37);
			this->Controls->Add(this->button36);
			this->Controls->Add(this->button8);
			this->Controls->Add(this->button35);
			this->Controls->Add(this->button4);
			this->Controls->Add(this->button34);
			this->Controls->Add(this->button19);
			this->Controls->Add(this->button33);
			this->Controls->Add(this->button15);
			this->Controls->Add(this->button32);
			this->Controls->Add(this->button11);
			this->Controls->Add(this->button31);
			this->Controls->Add(this->button7);
			this->Controls->Add(this->button30);
			this->Controls->Add(this->button3);
			this->Controls->Add(this->button29);
			this->Controls->Add(this->button18);
			this->Controls->Add(this->button28);
			this->Controls->Add(this->button14);
			this->Controls->Add(this->button27);
			this->Controls->Add(this->button10);
			this->Controls->Add(this->button26);
			this->Controls->Add(this->button6);
			this->Controls->Add(this->button25);
			this->Controls->Add(this->button2);
			this->Controls->Add(this->button24);
			this->Controls->Add(this->button17);
			this->Controls->Add(this->button23);
			this->Controls->Add(this->button13);
			this->Controls->Add(this->button22);
			this->Controls->Add(this->button9);
			this->Controls->Add(this->button21);
			this->Controls->Add(this->button5);
			this->Controls->Add(this->button1);
			this->Controls->Add(this->menuStrip3);
			this->Margin = System::Windows::Forms::Padding(6);
			this->Name = L"MyForm";
			this->Text = L"MyForm";
			this->menuStrip3->ResumeLayout(false);
			this->menuStrip3->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

		double firstDigit, secondDigit, result;
		String^ operators;

private: System::Void EnterNumber(System::Object^ sender, System::EventArgs^ e) {
	Button^ Numbers = safe_cast<Button^>(sender);

	if (textBox1->Text == "0") {
		textBox1->Text = Numbers->Text;
	}else{
		textBox1->Text = textBox1->Text + Numbers->Text;
	}
}

private: System::Void EnterOperator(System::Object^ sender, System::EventArgs^ e) {
	Button^ NumbersOp = safe_cast<Button^>(sender);
	firstDigit = Double::Parse(textBox1->Text);
	textBox1->Text = "";
	operators =NumbersOp->Text;
}
private: System::Void button18_Click(System::Object^ sender, System::EventArgs^ e) {
	if (!textBox1->Text->Contains(".")) {
		textBox1->Text = textBox1->Text + ".";
	}
}
private: System::Void textBox1_TextChanged(System::Object^ sender, System::EventArgs^ e) {


}
	   
private: System::Void button19_Click(System::Object^ sender, System::EventArgs^ e) {
	secondDigit= Double::Parse(textBox1->Text);

	if (operators == "+") {
		result = firstDigit + secondDigit;
		textBox1->Text = System::Convert::ToString(result);
		
	}
	else if (operators == "-") {
		result = firstDigit - secondDigit;
		textBox1->Text = System::Convert::ToString(result);
		result = firstDigit + secondDigit;

	}
	else if (operators == "*") {
		result = firstDigit * secondDigit;
		textBox1->Text = System::Convert::ToString(result);
	}
	else if (operators == "/") {
		result = firstDigit / secondDigit;
		textBox1->Text = System::Convert::ToString(result);
	}

}
private: System::Void button2_Click(System::Object^ sender, System::EventArgs^ e) {
	textBox1->Text = "0";
}
private: System::Void button3_Click(System::Object^ sender, System::EventArgs^ e) {
	textBox1->Text = "0";
}
private: System::Void button4_Click(System::Object^ sender, System::EventArgs^ e) {
	if (textBox1->Text->Contains("-")) {
		textBox1->Text = textBox1->Text->Remove(0,1);
	}
	else {
		textBox1->Text = "-"+textBox1->Text;
	}
}
private: System::Void button1_Click(System::Object^ sender, System::EventArgs^ e) {
	if (textBox1->Text->Length>0) {
		textBox1->Text = textBox1->Text->Remove(textBox1->Text->Length-1,1);
	}
}
};
}
