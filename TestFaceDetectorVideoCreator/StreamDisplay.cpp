#include "StreamDisplay.h"

StreamDisplay::StreamDisplay(QWidget *parent) : QLabel(parent)
{
    _isMousePressed = false;
    _selectedRectId = -1;

    setMouseTracking(true);
}

StreamDisplay::~StreamDisplay()
{
}

void StreamDisplay::setVideoFrame(const QImage &qImg, const QSize &videoFrameSize)
{
    _videoFrameSize = videoFrameSize;
    _streamDisplayPixmap.convertFromImage(qImg);
    _scaledStreamDisplayPixmap = _streamDisplayPixmap;

    setPixmap(_streamDisplayPixmap);
    drawFaceRects();
}

void StreamDisplay::drawFaceRects()
{
    QPixmap curDisplayState = _scaledStreamDisplayPixmap;

    // рассчитываем разницу между размером окна и размером кадра
    float widthRatio = float(_videoFrameSize.width()) / float(curDisplayState.size().width());
    float heightRatio = float(_videoFrameSize.height()) / float(curDisplayState.size().height());

    // отрисовка прямоугольников и их размеров
    QPainter painter(&curDisplayState);
    painter.setPen(QPen(QBrush(Qt::red), 5));
    painter.drawRects(_qFaceRects);
    painter.setPen(QPen(QBrush(Qt::green), 1));

    for(int i = 0; i < _qFaceRects.size(); ++i)
    {
        QString txtRectSize = QString((std::to_string(int(_qFaceRects[i].height() * heightRatio)) + std::string("x") + std::to_string(int(_qFaceRects[i].height() * heightRatio))).c_str());
        painter.drawText(QPoint(_qFaceRects[i].x() + 5, _qFaceRects[i].y() + 15), txtRectSize);
    }

    painter.end();

    setPixmap(curDisplayState);
}

void StreamDisplay::setFaceRects(const QVector<QRect> &qFaceRects)
{
    _qFaceRects = qFaceRects;

    drawFaceRects();
}

int StreamDisplay::getPressedRect(const QPoint &pressedPoint, BorderCode &selectedRectBorder, CornerCode &selectedRectCorner, int &dx, int &dy)
{
    int pressedRectId = -1;

    // определяем, выбран ли какой либо прямоугольник и если да, то какой его элемент
    for(int i = 0; i < _qFaceRects.size(); ++i)
    {
        if(pressedPoint.y() > _qFaceRects[i].y() + 5 && pressedPoint.y() < (_qFaceRects[i].y() + _qFaceRects[i].height() - 5))
        {
            if(abs(pressedPoint.x() - _qFaceRects[i].x()) <= 5)
            {
                selectedRectBorder = BorderCode::LEFT;
                selectedRectCorner = CornerCode::NO;
                dx = pressedPoint.x() - _qFaceRects[i].x();
                dy = pressedPoint.y() - _qFaceRects[i].y();
                pressedRectId = i;
            }
            else if(abs(pressedPoint.x() - (_qFaceRects[i].x() + _qFaceRects[i].width())) <= 5)
            {
                selectedRectBorder = BorderCode::RIGHT;
                selectedRectCorner = CornerCode::NO;
                dx = pressedPoint.x() - _qFaceRects[i].x();
                dy = pressedPoint.y() - _qFaceRects[i].y();
                pressedRectId = i;
            }
        }

        if(pressedPoint.x() > _qFaceRects[i].x() + 5 && pressedPoint.x() < (_qFaceRects[i].x() + _qFaceRects[i].width() - 5))
        {
            if(abs(pressedPoint.y() - _qFaceRects[i].y()) <= 5)
            {
                selectedRectBorder = BorderCode::TOP;
                selectedRectCorner = CornerCode::NO;
                dx = pressedPoint.x() - _qFaceRects[i].x();
                dy = pressedPoint.y() - _qFaceRects[i].y();
                pressedRectId = i;
            }
            else if(abs(pressedPoint.y() - (_qFaceRects[i].y() + _qFaceRects[i].height())) <= 5)
            {
                selectedRectBorder = BorderCode::BOTTOM;
                selectedRectCorner = CornerCode::NO;
                dx = pressedPoint.x() - _qFaceRects[i].x();
                dy = pressedPoint.y() - _qFaceRects[i].y();
                pressedRectId = i;
            }
        }

        if(pressedPoint.y() > _qFaceRects[i].y() - 5 && pressedPoint.y() < _qFaceRects[i].y() + 5)
        {
            if(abs(pressedPoint.x() - _qFaceRects[i].x()) <= 5)
            {
                selectedRectBorder = BorderCode::NO;
                selectedRectCorner = CornerCode::LT;
                dx = 0;
                dy = 0;
                pressedRectId = i;
            }
        }

        if(pressedPoint.y() > (_qFaceRects[i].bottomLeft().y() - 5) && pressedPoint.y() < (_qFaceRects[i].bottomLeft().y() + 5))
        {
            if(abs(pressedPoint.x() - _qFaceRects[i].bottomLeft().x()) <= 5)
            {
                selectedRectBorder = BorderCode::NO;
                selectedRectCorner = CornerCode::LB;
                dx = 0;
                dy = 0;
                pressedRectId = i;
            }
        }

        if(pressedPoint.x() > (_qFaceRects[i].topRight().x() - 5) && pressedPoint.x() < (_qFaceRects[i].topRight().x() + 5))
        {
            if(abs(pressedPoint.y() - (_qFaceRects[i].topRight().y())) <= 5)
            {
                selectedRectBorder = BorderCode::NO;
                selectedRectCorner = CornerCode::RT;
                dx = 0;
                dy = 0;
                pressedRectId = i;
            }
        }

        if(pressedPoint.x() > (_qFaceRects[i].bottomRight().x() - 5) && pressedPoint.x() < (_qFaceRects[i].bottomRight().x() + 5))
        {
            if(abs(pressedPoint.y() - (_qFaceRects[i].bottomRight().y())) <= 5)
            {
                selectedRectBorder = BorderCode::NO;
                selectedRectCorner = CornerCode::RB;
                dx = 0;
                dy = 0;
                pressedRectId = i;
            }
        }
    }

    return pressedRectId;
}

void StreamDisplay::resizeEvent(QResizeEvent *event)
{
    QSize oldPixmapSize = _scaledStreamDisplayPixmap.size();

    // при изменении размера окна, изменяем размер отрисовки кадра и прямоугольников
    _scaledStreamDisplayPixmap = _streamDisplayPixmap.scaled(size().width(), size().height(), Qt::KeepAspectRatio);

    QSize curPixmapSize = _scaledStreamDisplayPixmap.size();

    // рассчитываем разницу между размером окна и размером кадра
    float widthRatio = float(curPixmapSize.width()) / float(oldPixmapSize.width());
    float heightRatio = float(curPixmapSize.height()) / float(oldPixmapSize.height());

    QVector<QRect> qFaceRectsScaled;
    for(int i = 0; i < _qFaceRects.size(); ++i)
    {
        QRect qFaceRect(int(_qFaceRects[i].x() * widthRatio + 0.5),
                        int(_qFaceRects[i].y() * heightRatio + 0.5),
                        int(_qFaceRects[i].width() * widthRatio + 0.5),
                        int(_qFaceRects[i].height() * heightRatio + 0.5));

        qFaceRect.setWidth(qFaceRect.height());

        if(qFaceRect.width() > 5 && qFaceRect.height() > 5)
            qFaceRectsScaled.push_back(qFaceRect);
    }

    _qFaceRects = qFaceRectsScaled;

    drawFaceRects();
}

void StreamDisplay::mouseDoubleClickEvent(QMouseEvent *event)
{
    QPoint pressedPoint = event->pos();

    float dx = std::abs(size().width() - _scaledStreamDisplayPixmap.size().width()) / 2.0;
    float dy = std::abs(size().height() - _scaledStreamDisplayPixmap.size().height()) / 2.0;

    pressedPoint.setX(pressedPoint.x() - dx);
    pressedPoint.setY(pressedPoint.y() - dy);

    // по двойному нажатию левой кнопки мыши добавляем новый прямоугольник с центром в указанной точке
    if(event->button() == Qt::MouseButton::LeftButton)
    {
        QPoint rectCenterPoint(pressedPoint.x() - 50, pressedPoint.y() - 50);

        if(rectCenterPoint.x() < 0)
            rectCenterPoint.setX(0);

        if(rectCenterPoint.x() > (size().width() - 100))
            rectCenterPoint.setX(size().width() - 100);

        if(rectCenterPoint.y() < 0)
            rectCenterPoint.setY(0);

        if(rectCenterPoint.y() > (size().height() - 100))
            rectCenterPoint.setY(size().height() - 100);

        QRect qFaceRect(rectCenterPoint, QSize(100, 100));
        _qFaceRects.push_back(qFaceRect);
    }
    //при двойном нажатии правой кнопки мыши удаляем выбранный прямоугольник
    else if(event->button() == Qt::MouseButton::RightButton)
    {
        BorderCode bc;
        CornerCode cc;
        int pressedRectId = getPressedRect(pressedPoint, bc, cc, _dx, _dy);

        if(pressedRectId != -1)
            _qFaceRects.remove(pressedRectId);
    }

    drawFaceRects();
}

void StreamDisplay::mousePressEvent( QMouseEvent *event )
{
    _isMousePressed = true;

    QPoint pressedPoint = event->pos();

    float dx = std::abs(size().width() - _scaledStreamDisplayPixmap.size().width()) / 2.0;
    float dy = std::abs(size().height() - _scaledStreamDisplayPixmap.size().height()) / 2.0;

    pressedPoint.setX(pressedPoint.x() - dx);
    pressedPoint.setY(pressedPoint.y() - dy);

    _selectedRectId = getPressedRect(pressedPoint, _selectedRectBorder, _selectedRectCorner, _dx, _dy);
}

void StreamDisplay::mouseReleaseEvent( QMouseEvent *event )
{
    _isMousePressed = false;
    _selectedRectId = -1;
    _dx = 0;
    _dy = 0;
    _selectedRectBorder = BorderCode::NO;
    _selectedRectCorner = CornerCode::NO;
}

void StreamDisplay::mouseMoveEvent( QMouseEvent *event )
{
    QPoint pressedPoint = event->pos();

    float dx = std::abs(size().width() - _scaledStreamDisplayPixmap.size().width()) / 2.0;
    float dy = std::abs(size().height() - _scaledStreamDisplayPixmap.size().height()) / 2.0;

    pressedPoint.setX(pressedPoint.x() - dx);
    pressedPoint.setY(pressedPoint.y() - dy);

    if(_isMousePressed && _selectedRectId != -1)
    {
        // при зажатии мышью стороны прямоугольника, перетаскиваем его
        if(_selectedRectBorder != BorderCode::NO)
        {
            QRect tmpRect = _qFaceRects[_selectedRectId];

            _qFaceRects[_selectedRectId].setTopLeft(QPoint(pressedPoint.x() - _dx,
                                                           pressedPoint.y() - _dy));

            _qFaceRects[_selectedRectId].setWidth(tmpRect.width());
            _qFaceRects[_selectedRectId].setHeight(tmpRect.height());
            drawFaceRects();
        }
        // при зажатии мышью угла прямоугольника, изменяем его размер
        else
        {
            QPoint oldTopLeftPoint = _qFaceRects[_selectedRectId].topLeft();

            switch (_selectedRectCorner)
            {
                case CornerCode::LT:
                {
                    _qFaceRects[_selectedRectId].setTopLeft(QPoint(pressedPoint.x(), pressedPoint.y()));
                    if(_qFaceRects[_selectedRectId].height() != _qFaceRects[_selectedRectId].width())
                        _qFaceRects[_selectedRectId].setY(_qFaceRects[_selectedRectId].y() + (_qFaceRects[_selectedRectId].height() - _qFaceRects[_selectedRectId].width()));
                    break;
                }
                case CornerCode::RT:
                {
                    _qFaceRects[_selectedRectId].setTopRight(QPoint(pressedPoint.x(), pressedPoint.y()));
                    _qFaceRects[_selectedRectId].setWidth(_qFaceRects[_selectedRectId].height());
                    break;
                }
                case CornerCode::LB:
                {
                    _qFaceRects[_selectedRectId].setBottomLeft(QPoint(pressedPoint.x(), pressedPoint.y()));
                    _qFaceRects[_selectedRectId].setHeight(_qFaceRects[_selectedRectId].width());
                    break;
                }
                case CornerCode::RB:
                {
                    _qFaceRects[_selectedRectId].setBottomRight(QPoint(pressedPoint.x(), pressedPoint.y()));
                    _qFaceRects[_selectedRectId].setWidth(_qFaceRects[_selectedRectId].height());
                    break;
                }
            }

            if(_qFaceRects[_selectedRectId].width() < 20)
            {
                _qFaceRects[_selectedRectId].setTopLeft(oldTopLeftPoint);
                _qFaceRects[_selectedRectId].setWidth(20);
            }

            if(_qFaceRects[_selectedRectId].height() < 20)
            {
                _qFaceRects[_selectedRectId].setTopLeft(oldTopLeftPoint);
                _qFaceRects[_selectedRectId].setHeight(20);
            }

            drawFaceRects();
        }
    }
}

void StreamDisplay::getFaceRects(QVector<QRect> &qFaceRects)
{
    qFaceRects = _qFaceRects;
}

void StreamDisplay::getStreamDisplaySize(QSize &streamDisplaySize)
{
    streamDisplaySize = this->size();
}

void StreamDisplay::getPixmapSize(QSize &pixmapSize)
{
    pixmapSize = _scaledStreamDisplayPixmap.size();
}
