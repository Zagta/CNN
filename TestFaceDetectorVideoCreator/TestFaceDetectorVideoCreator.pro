#-------------------------------------------------
#
# Project created by QtCreator 2017-04-26T18:16:28
#
#-------------------------------------------------

QT       += core gui
QMAKE_CXXFLAGS += -std=c++11

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = TestFaceDetectorVideoCreator
TEMPLATE = app

LIBS += -L/usr/local/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect

SOURCES += main.cpp\
        mainwindow.cpp \
    OpenCVStreamer.cpp \
    SuperVisor.cpp \
    CpuFaceDetector.cpp \
    StreamDisplay.cpp \
    FaceMetaDataProcessor.cpp

HEADERS  += mainwindow.h \
    SuperVisor.h \
    OpenCVStreamer.h \
    CpuFaceDetector.h \
    StreamDisplay.h \
    FaceMetaDataProcessor.h

FORMS    += mainwindow.ui
