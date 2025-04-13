#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <vtkSmartPointer.h>
#include <vtkVolumeProperty.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(QWidget *parent = nullptr);
	~MainWindow();

private slots:
	void on_loadFileButton_clicked();

	void on_midOpacitySlider_sliderMoved(int position);

	void on_midAtSlider_sliderMoved(int position);

private:
	Ui::MainWindow *ui;

	vtkSmartPointer<vtkVolumeProperty> volumeProperty;

	void opacityChanged();
};
#endif // MAINWINDOW_H
