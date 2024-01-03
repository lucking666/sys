#ifndef TCPCLIENT_H
#define TCPCLIENT_H

#include <QWidget>
#include<qfile.h>
#include<QTcpSocket>

QT_BEGIN_NAMESPACE
namespace Ui {
class TcpClient;
}
QT_END_NAMESPACE

class TcpClient : public QWidget
{
    Q_OBJECT

public:
    TcpClient(QWidget *parent = nullptr);
    ~TcpClient();
    void loadconfig();
public slots:
    void showConnect();

private:
    Ui::TcpClient *ui;
    QString m_strIP;
    quint16 m_usPort;
    QTcpSocket m_tcpSocket;
};
#endif // TCPCLIENT_H
