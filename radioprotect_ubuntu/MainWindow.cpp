#include "MainWindow.h"
#include "./ui_MainWindow.h"
#include <QFileDialog>
#include <vtkImageData.h>
#include <vtkVolume.h>
#include <vtkRenderer.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkOpenGLGPUVolumeRayCastMapper.h>
#include <vtkPiecewiseFunction.h>
#include <QVTKOpenGLNativeWidget.h>

#include <highfive/highfive.hpp>
#include <boost/multi_array.hpp>

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::MainWindow)
{
	ui->setupUi(this);
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::on_loadFileButton_clicked()
{
	try {
		QString name = QFileDialog::getOpenFileName(this,"Open As","", "HDF5 (*.h5)\n All Files (*.*)");
		HighFive::File file(name.toStdString(), HighFive::File::ReadOnly);
		HighFive::DataSet dataset = file.getDataSet("volume");
		std::vector<size_t> dimensions = dataset.getDimensions();
		std::cout << "Dimensions: " << dimensions[0] << " " << dimensions[1] << " " << dimensions[2] << std::endl;
		std::vector<uint16_t> values(dimensions[0] * dimensions[1] * dimensions[2]);
		dataset.read_raw(values.data());

		vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
		imageData->SetDimensions(dimensions[0], dimensions[1], dimensions[2]);
		imageData->AllocateScalars(VTK_UNSIGNED_SHORT, 1);
		imageData->SetSpacing(3.0, 1.0, 1.0);
		imageData->SetOrigin(0.0, 0.0, 0.0);
		uint16_t* vtkData = static_cast<uint16_t*>(imageData->GetScalarPointer());
		for (size_t z = 0; z < dimensions[2]; z++) {
			for (size_t y = 0; y < dimensions[1]; y++) {
				for (size_t x = 0; x < dimensions[0]; x++) {
					size_t vtkIndex = z * dimensions[0] * dimensions[1] + y * dimensions[0] + x;
					size_t hdf5Index = x * dimensions[2] * dimensions[1] + y * dimensions[2] + z;
					vtkData[vtkIndex] = values[hdf5Index];
				}
			}
		}

		vtkSmartPointer<vtkOpenGLGPUVolumeRayCastMapper> volumeMapper = vtkSmartPointer<vtkOpenGLGPUVolumeRayCastMapper>::New();
		volumeMapper->SetInputData(imageData);
		volumeMapper->SetBlendModeToComposite();

		volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();
		volumeProperty->ShadeOn();
		volumeProperty->SetAmbient(0.1);
		volumeProperty->SetDiffuse(0.9);
		volumeProperty->SetSpecular(0.2);
		volumeProperty->SetSpecularPower(10.0);
		opacityChanged();

		vtkSmartPointer<vtkVolume> volume = vtkSmartPointer<vtkVolume>::New();
		volume->SetMapper(volumeMapper);
		volume->SetProperty(volumeProperty);

		vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
		renderer->AddVolume(volume);
		renderer->SetBackground(0.1, 0.2, 0.3);

		vtkSmartPointer<vtkGenericOpenGLRenderWindow> renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
		renderWindow->AddRenderer(renderer);

		// 5. Qt Integration
		ui->visualisation->setRenderWindow(renderWindow);
	} catch (std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
}


void MainWindow::on_midOpacitySlider_sliderMoved(int position)
{
	opacityChanged();
}


void MainWindow::on_midAtSlider_sliderMoved(int position)
{
	opacityChanged();
}

void MainWindow::opacityChanged()
{
	vtkSmartPointer<vtkPiecewiseFunction> opacityFunction = vtkSmartPointer<vtkPiecewiseFunction>::New();
	// Add points (data value, opacity)
	opacityFunction->AddPoint(0, 0.0);
	opacityFunction->AddPoint(ui->midAtSlider->value(), 0.0001 * pow(1.01, ui->midOpacitySlider->value()));
	opacityFunction->AddPoint(3000, 1);
	volumeProperty->SetScalarOpacity(opacityFunction);
	ui->visualisation->renderWindow()->Render();
}

