
#include <cpp_space_init.h>

void qin_fun()
{
    QTextStream qin(stdin);
    QString qin_str;
    while(1)
    {
        //qin >> qin_str;
        //qDebug() << qin_str;
        qDebug() << "1111111111111111111~";
    }
}

void regist_callback(Fun pCallback)
{
    p = pCallback;
}

int cpp_space_init()
{
    int argc = 1;
    char* argv[] = {"./SophonOS"};
    QApplication a(argc, argv);
    MainWindow w;
    //w.show();

    (*p)(32);
    //ffmpeg_encode();
    //opencv_encode();
    testrtmp();
    qDebug() << "ffmpeg_encode end~";
    //QThread * qin_thread = QThread::create(qin_fun);
    //qin_thread->start();
    //qDebug() << "argc: " << argc;
    //for (int i = 1; i < argc; ++i) {
      //qDebug() << "argv: " << argv[i];
    //}
    return a.exec();
}