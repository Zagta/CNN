#ifndef STREAMDISPLAY_H
#define STREAMDISPLAY_H

#include <QLabel>
#include <QVector>
#include <QRect>
#include <QPainter>
#include <QMouseEvent>
#include <QDebug>

enum class BorderCode {LEFT, RIGHT, TOP, BOTTOM, NO}; // коды выбраных сторон
enum class CornerCode {LT, RT, LB, RB, NO}; //коды выбраных углов

// Объект для отображения потока видео и найденых лиц
class StreamDisplay : public QLabel
{
    Q_OBJECT

public:
    StreamDisplay(QWidget *parent=0);
    ~StreamDisplay();

public slots:
    void setVideoFrame(const QImage &qImg, const QSize &videoFrameSize);
    void setFaceRects(const QVector<QRect> &qFaceRects);

    void getFaceRects(QVector<QRect> &qFaceRects);
    void getStreamDisplaySize(QSize &streamDisplaySize);
    void getPixmapSize(QSize &pixmapSize);

protected:
    void resizeEvent(QResizeEvent *event); // изменение размера окна

    void mousePressEvent( QMouseEvent *event );
    void mouseDoubleClickEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent( QMouseEvent *event );

private:
    void drawFaceRects();
    int getPressedRect(const QPoint &pressedPoint, BorderCode &selectRectBorder, CornerCode &selectedRectCorner, int &dx, int &dy);

    QVector<QRect> _qFaceRects;

    QPixmap _streamDisplayPixmap;
    QPixmap _scaledStreamDisplayPixmap;
    QSize _videoFrameSize;

    bool _isMousePressed;

    int _selectedRectId; // номер выбранного прямоугольника
    int _dx; // расстояние по x от точки нажатия на сторону прямоугольника до левого верхнего угла
    int _dy; // расстояние по y от точки нажатия на сторону прямоугольника до левого верхнего угла
    BorderCode _selectedRectBorder;
    CornerCode _selectedRectCorner;

};

#endif // STREAMDISPLAY_H
