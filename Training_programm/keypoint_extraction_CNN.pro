QT += core
QT -= gui

CONFIG += c++11

TARGET = keypoint_extraction_CNN
CONFIG += console
CONFIG -= app_bundle

QMAKE_CXXFLAGS += -DDLIB_USE_CUDA

TEMPLATE = app

SOURCES += main.cpp \
    LMDBwrapper/LMDB.cpp \
    LMDBwrapper/LMDBCursor.cpp \
    LMDBwrapper/LMDBTransaction.cpp \
    img_proc.cpp \
    boost_program_options.cpp \
    LMDBwrapper/LMDB_handler.cpp

SOURCES += /home/ginseng/lib/dlib-19.4/dlib/all/source.cpp

INCLUDEPATH += /home/ginseng/lib/dlib-19.4

LIBS += /home/ginseng/lib/dlib-19.4/buildGPU/libdlib.a
LIBS += -L/usr/local/cuda/lib64 -lcuda -lcurand -lcublas -lcudart -lcudnn
LIBS += -L/usr/local/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect
LIBS += -L/usr/lib/x86_64-linux-gnu -lboost_program_options
LIBS += -L/usr/lib/x86_64-linux-gnu -llmdb


LIBS += -lX11

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


HEADERS += \
    LMDBwrapper/LMDB.h \
    LMDBwrapper/LMDBCursor.h \
    LMDBwrapper/LMDBTransaction.h \
    img_proc.h \
    boost_program_options.h \
    LMDBwrapper/LMDB_handler.h
