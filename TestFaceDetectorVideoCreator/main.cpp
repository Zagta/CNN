#include "mainwindow.h"
#include <QApplication>

#include <SuperVisor.h>
#include <OpenCVStreamer.h>
#include <StreamDisplay.h>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    SuperVisor superVisor(a.applicationDirPath());
    MainWindow w(a.applicationDirPath(), &superVisor);
    w.show();

    return a.exec();
}
