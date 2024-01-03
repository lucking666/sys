#include "tcpclient.h"
#include "ui_tcpclient.h"
#include<QByteArray>
#include<QDebug>
#include<QHostAddress>
#include<QMessageBox>
TcpClient::TcpClient(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::TcpClient)
{
    ui->setupUi(this);
    loadconfig();
    connect(&m_tcpSocket,SIGNAL(connected()),this,SLOT(showConnect()));
    m_tcpSocket.connectToHost(QHostAddress(m_strIP),m_usPort);
}

TcpClient::~TcpClient()
{
    delete ui;
}

void TcpClient::loadconfig()
{
    QFile file(":/client.config");
    if(file.open(QIODevice::ReadOnly)){
        QByteArray baData=file.readAll();
        QString strData=baData.toStdString().c_str();
        qDebug()<<strData;
        strData.replace("\r\n"," ");
        qDebug()<<strData;
        file.close();
        QStringList strList=strData.split(" ");
        for(int i=0;i<strList.size();i++){
            qDebug()<<"--->"<<strList[i];
        }
        m_strIP=strList.at(0);
        m_usPort=strList.at(1).toUShort();
        qDebug()<<"ip:"<<m_strIP<<"port:"<<m_usPort;
    }
}

void TcpClient::showConnect()
{
    QMessageBox::information(this,"连接服务器","链接服务器成功");
}
