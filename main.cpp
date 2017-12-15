#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>
#include<iostream>
#include<sstream>
#include <String.h>

using namespace std;
using namespace cv;


// Variáveis Globais ///////////////////////////////////////////////////////////////////////////////
const int MinAreaContorno = 800;

const int redimensionaLargura = 20;
const int redimensionaAltura = 30;

///////////////////////////////////////////////////////////////////////////////////////////////////
class DadosContorno {
public:

    vector<Point> ptContour;
    Rect areaLimitadora;
    float fltArea;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    bool verificaValidadeContorno() {
        if (fltArea < MinAreaContorno) return false;
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool ordenaPosicaoRecorte(const DadosContorno& cwdesqueda, const DadosContorno& cwdDireita) {
        return(cwdesqueda.areaLimitadora.x < cwdDireita.areaLimitadora.x);
    }

};


///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {
    Mat src = imread("placa 3C.jpg");
    cvtColor (src, src, CV_BGR2GRAY);
    threshold(src,src,250,250,CV_THRESH_OTSU);
    imshow("in",src);
    imwrite("placosa.jpg",src);
    Mat input = imread("placosa.jpg", 1 );


    string placa;

    vector<DadosContorno> todosContornos;
    vector<DadosContorno> contornosValidos;

            // Leitura do Treino das classificações  ///////////////////////////////////////////////////

    Mat matClassificacoes;

    FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);

    if (fsClassifications.isOpened() == false) {
        cout << "error, falha em abrir classificacoes \n\n";
        return(0);
    }

    fsClassifications["classifications"] >> matClassificacoes;
    fsClassifications.release();

            // Leitura do Treino das Imagens ////////////////////////////////////////////////////////////

    Mat matImagensClassificadas;

    FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);

    if (fsTrainingImages.isOpened() == false) {
        cout << "erro, falha em abrir imagens classificadas\n\n";
        return(0);
    }

    fsTrainingImages["images"] >> matImagensClassificadas;
    fsTrainingImages.release();

            // Treino //////////////////////////////////////////////////////////////////////////////

    Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());

    kNearest->train(matImagensClassificadas, cv::ml::ROW_SAMPLE, matClassificacoes);

            // Teste ///////////////////////////////////////////////////////////////////////////////

    Mat matTeste = input;

    if (matTeste.empty()) {
        std::cout << "erro: imagem nao foi lida do arquivo\n\n";
        return(0);
    }

    Mat matGrayscale;
    Mat matBlurred;
    Mat matThresh;
    Mat matThreshCopy;

    cvtColor(matTeste, matGrayscale, CV_BGR2GRAY);

    GaussianBlur(matGrayscale, matBlurred, Size(5, 5),0);

    adaptiveThreshold(matBlurred,matThresh,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY_INV,11,2);

    matThreshCopy = matThresh.clone();
    imshow("Plac", matThreshCopy);

    vector<vector<Point> > ptContours;
    vector<Vec4i> v4iHierarchy;


    //convertScaleAbs(input,input,256);
    findContours(matThreshCopy/*input*/, ptContours, v4iHierarchy,RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // For para pegar dados de cada contorno
    for (int i = 0; i < ptContours.size(); i++) {
        DadosContorno contorno;
        contorno.ptContour = ptContours[i];
        contorno.areaLimitadora = boundingRect(contorno.ptContour);
        contorno.fltArea = contourArea(contorno.ptContour);
        todosContornos.push_back(contorno);
    }


    // For para vaidar contornos
    for (int i = 0; i < todosContornos.size(); i++) {
        if (todosContornos[i].verificaValidadeContorno()) {
            contornosValidos.push_back(todosContornos[i]);
        }
    }

    // Ordena os contornos da esquerda para direita

    sort(contornosValidos.begin(), contornosValidos.end(), DadosContorno::ordenaPosicaoRecorte);

    string strFinalString;

    for (int i = 0; i < contornosValidos.size(); i++) {            // Este for trabalha cada contorno válido separadamente

        // Desenha retângulo sobre a letra
        rectangle(matTeste, contornosValidos[i].areaLimitadora, cv::Scalar(0, 255, 0), 2);

        Mat matROI = matThresh(contornosValidos[i].areaLimitadora);
        // Redimensiona ROI
        Mat matROIRedimensionada;
        resize(matROI, matROIRedimensionada, cv::Size(redimensionaLargura, redimensionaAltura));
        Mat matROIFloat;

        matROIRedimensionada.convertTo(matROIFloat, CV_32FC1);

        //Remodela ROI para dimensão 1x1 transformando em Mat float para trabalhar no KNN
        Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);
        Mat matCharAtual(0, 0, CV_32F);

        // Chamada do KNN
        kNearest->findNearest(matROIFlattenedFloat, 1, matCharAtual);

        float fltCurrentChar = (float)matCharAtual.at<float>(0, 0);

        strFinalString = strFinalString + char(int(fltCurrentChar));
        placa = strFinalString;
    }

    imshow("Placa", input);
    cout<<"Placa: ";
    for (int i=0;i<3;i++) cout<< placa[i];
    cout<<"-";
    for (int i=3;i<7;i++) cout<< placa[i];

    waitKey(0);

    return(0);
}
