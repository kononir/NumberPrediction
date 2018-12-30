// NumberPrediction.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "NumberPrediction.h"
#include <cmath>
#include <ctime>
#include <iostream>

using namespace std;

/*
* author: Novitskiy Vladislav
* group: 621701
* description: Главная функция; в ней происходит ввод входных параметров сети и вызываются остальные функции
*/
int main()
{
	int sequenceSize;
	double* sequence;
	NeuralNetwork neuralNetwork;

	try {
		int sequenceCode;

		cout << "Choose sequence: " << endl;
		cout << "1 - x+1\n2 - 1, 3, 5, 7, ... (periodic)\n3 - Fibonacci series\n4 - 2^(x+2)\n5 - x!\n6 - x^2\n";
		cin >> sequenceCode;

		switch (sequenceCode) {
		case 1: {
			sequenceSize = 10;
			double temp[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
			sequence = temp;

			break;
		}
		case 2: {
			sequenceSize = 10;
			double temp[10] = { 1, 3, 5, 7, 1, 3, 5, 7, 1, 3 };
			sequence = temp;

			break;
		}
		case 3: {
			sequenceSize = 10;
			double temp[10] = { 0, 1, 1, 2, 3, 5, 8, 13, 21, 34 };
			sequence = temp;

			break;
		}
		case 4: {
			sequenceSize = 8;
			double temp[8] = { 4, 8, 16, 32, 64, 128, 256, 512 };
			sequence = temp;

			break;
		}
		case 5: {
			sequenceSize = 5;
			double temp[5] = { 1, 2, 6, 24, 120 };
			sequence = temp;

			break;
		}
		case 6: {
			sequenceSize = 10;
			double temp[10] = { 1, 4, 9, 16, 25, 36, 49, 64, 81, 100 };
			sequence = temp;

			break;
		}
		default:
			throw "Error - Invalid parameter: sequence code";
		}

		cout << "Enter max error (0 < e <= 0.1): ";
		cin >> neuralNetwork.maximumAllowableError;

		if (neuralNetwork.maximumAllowableError <= 0) {
			throw "Error - Invalid parameter: maximum allowable error";
		}

		cout << "Enter number of neurons (m >= 1): ";
		cin >> neuralNetwork.neuronsNumber;

		if (neuralNetwork.neuronsNumber <= 0) {
			throw "Error - Invalid parameter: neurons number";
		}

		cout << "Enter window size (p >= 1): ";
		cin >> neuralNetwork.windowSize;

		if (neuralNetwork.windowSize <= 0) {
			throw "Error - Invalid parameter: window size";
		}

		cout << "Input coefficient of training (0 < a <= 0.1, a <= e): ";
		cin >> neuralNetwork.trainingCoefficient;

		if (neuralNetwork.trainingCoefficient <= 0 || neuralNetwork.trainingCoefficient > 1) {
			throw "Error - Invalid parameter: training coefficient";
		}

		cout << "Enter maximum allowable number of training steps (N >= 1, N = 1000000): ";
		cin >> neuralNetwork.maximumAllowableNumberOfTrainingSteps;

		if (neuralNetwork.maximumAllowableNumberOfTrainingSteps <= 0) {
			throw "Error - Invalid parameter: maximum allowable number of trainingSteps";
		}
	}
	catch (const char* mesage) {
		cerr << mesage << endl;

		system("pause");
		return 1;
	}

	double max = findMax(sequence, sequenceSize);
	neuralNetwork.scale = 0;

	if (max >= 1) {
		if (max < 10) {
			neuralNetwork.scale = 10;
		}
		else if (max < 100) {
			neuralNetwork.scale = 100;
		}
		else if (max < 1000) {
			neuralNetwork.scale = 1000;
		}

		scaleSequence(sequence, sequenceSize, neuralNetwork.scale);
	}

	initializeNeuralNetwork(neuralNetwork, sequence, sequenceSize);

	trainNeuralNetwork(neuralNetwork);

	double* lastWindow = new double[neuralNetwork.windowSize];

	for (int windowNumberIndex = 0; windowNumberIndex < neuralNetwork.windowSize; windowNumberIndex++) {
		lastWindow[windowNumberIndex] = sequence[windowNumberIndex + neuralNetwork.samplesNumber];
	}

	double predictedNumber = predictNextNumber(lastWindow, neuralNetwork);
	
	if (max >= 1) {
		predictedNumber = predictedNumber * neuralNetwork.scale;
	}
	
	cout << endl << "Reached error: " << neuralNetwork.reachedError << endl;
	cout << "Reached number of training steps: " << neuralNetwork.reachedNumberOfTrainingSteps << endl;

	cout << predictedNumber << endl;

	system("pause");
	return 0;
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция инициализации весовых матриц, порогов и обучающей выборки сети
*/
void initializeNeuralNetwork(NeuralNetwork &neuralNetwork, double* sequence, int sequenceSize) {
	double minWeight = -0.1;
	double maxWeight = 0.1;

	srand((unsigned int)time(0));

	neuralNetwork.currFirstLayerWeightMatrix = new double*[neuralNetwork.windowSize];

	for (int currRowNumber = 0; currRowNumber < neuralNetwork.windowSize; currRowNumber++) {
		neuralNetwork.currFirstLayerWeightMatrix[currRowNumber] = new double[neuralNetwork.neuronsNumber];

		for (int currColNumber = 0; currColNumber < neuralNetwork.neuronsNumber; currColNumber++) {
			neuralNetwork.currFirstLayerWeightMatrix[currRowNumber][currColNumber] 
				= (((double)rand() / RAND_MAX) * (maxWeight - minWeight)) + minWeight;
		}
	}

	neuralNetwork.currContextNeuronsWeightMatrix = new double*[neuralNetwork.neuronsNumber];

	for (int currRowNumber = 0; currRowNumber < neuralNetwork.neuronsNumber; currRowNumber++) {
		neuralNetwork.currContextNeuronsWeightMatrix[currRowNumber] = new double[neuralNetwork.neuronsNumber];

		for (int currColNumber = 0; currColNumber < neuralNetwork.neuronsNumber; currColNumber++) {
			neuralNetwork.currContextNeuronsWeightMatrix[currRowNumber][currColNumber]
				= (((double)rand() / RAND_MAX) * (maxWeight - minWeight)) + minWeight;
		}
	}

	neuralNetwork.currSecondLayerWeightMatrix = new double[neuralNetwork.neuronsNumber];

	for (int currRowNumber = 0; currRowNumber < neuralNetwork.neuronsNumber; currRowNumber++) {
		neuralNetwork.currSecondLayerWeightMatrix[currRowNumber] 
			= (((double)rand() / RAND_MAX) * (maxWeight - minWeight)) + minWeight;
	}

	neuralNetwork.currContextValues = new double[neuralNetwork.neuronsNumber];

	neuralNetwork.currFirstLayerThresholds = new double[neuralNetwork.neuronsNumber];
	memset(neuralNetwork.currFirstLayerThresholds, 0, neuralNetwork.neuronsNumber * sizeof(double));

	neuralNetwork.currSecondLayerThreshold = 0;

	neuralNetwork.samplesNumber = sequenceSize - neuralNetwork.windowSize;

	neuralNetwork.trainingSample = new double*[neuralNetwork.samplesNumber];
	neuralNetwork.rezultSample = new double[neuralNetwork.samplesNumber];

	for (int currRowNumber = 0; currRowNumber < neuralNetwork.samplesNumber; currRowNumber++) {
		neuralNetwork.trainingSample[currRowNumber] = new double[neuralNetwork.windowSize];

		for (int currColNumber = 0; currColNumber < neuralNetwork.windowSize; currColNumber++) {
			neuralNetwork.trainingSample[currRowNumber][currColNumber] = sequence[currColNumber + currRowNumber];
		}

		neuralNetwork.rezultSample[currRowNumber] = sequence[currRowNumber + neuralNetwork.windowSize];
	}
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция обучения рекурентной нейронной сети
*/
void trainNeuralNetwork(NeuralNetwork &neuralNetwork) {
	double** X = neuralNetwork.trainingSample;

	int &L = neuralNetwork.samplesNumber;
	int &m = neuralNetwork.neuronsNumber;
	int maxNumOfSteps = neuralNetwork.maximumAllowableNumberOfTrainingSteps;
	int numOfSteps = 0;

	double* XRez = neuralNetwork.rezultSample;
	double* contVal = neuralNetwork.currContextValues;
	double* S1 = new double[m];
	double* Y1 = new double[m];
    double* generalParts2 = new double[m];

	double &e = neuralNetwork.maximumAllowableError;
	double &scale = neuralNetwork.scale;
	double S2;
	double Y2;
    double generalPart1;
	double currError;
	double delta;

	do {
		memset(contVal, 0, m * sizeof(double));	// зануление контекстных нейронов

		for (int currImageryIndex = 0; currImageryIndex < L; currImageryIndex++) {
			calculateS1(S1, X[currImageryIndex], neuralNetwork);
			calculateY1(Y1, S1, neuralNetwork);

			calculateS2(S2, Y1, neuralNetwork);
			calculateY2(Y2, S2);
            
            calculateGeneralPart1(generalPart1, XRez[currImageryIndex], Y2, S2, neuralNetwork);

			modifyW2(generalPart1, Y1, neuralNetwork);
			modifyT2(generalPart1, neuralNetwork);
            
            calculateGeneralParts2(generalParts2, generalPart1, S1, neuralNetwork);

			modifyWCont(generalParts2, neuralNetwork);
			modifyW1(generalParts2, X[currImageryIndex], neuralNetwork);
			modifyT1(generalParts2, neuralNetwork);

			memcpy(contVal, Y1, m * sizeof(double));
		}

		memset(contVal, 0, m * sizeof(double));	// зануление контекстных нейронов

		currError = 0;

		for (int currImageryIndex = 0; currImageryIndex < L; currImageryIndex++) {
			calculateS1(S1, X[currImageryIndex], neuralNetwork);
			calculateY1(Y1, S1, neuralNetwork);

			calculateS2(S2, Y1, neuralNetwork);
			calculateY2(Y2, S2);

			delta = (Y2 - XRez[currImageryIndex]) * scale;
			currError += (delta * delta) / 2;

			memcpy(contVal, Y1, m * sizeof(double));
		}

		numOfSteps++;

		cout << currError << endl;
		//printf("%10f\r", currError);
	} while (currError > e && numOfSteps < maxNumOfSteps);

	neuralNetwork.reachedError = currError;
	neuralNetwork.reachedNumberOfTrainingSteps = numOfSteps;

	delete S1;
	delete Y1;
	delete generalParts2;
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция предсказания следующего элемента последовательности с помощью нейронной сети
*/
double predictNextNumber(double* X, NeuralNetwork &neuralNetwork) {
	double predictedNumber;
	int &m = neuralNetwork.neuronsNumber;

	double* S1 = new double[m];
	double* Y1 = new double[m];

	double S2;
	double Y2;

	calculateS1(S1, X, neuralNetwork);
	calculateY1(Y1, S1, neuralNetwork);

	calculateS2(S2, Y1, neuralNetwork);
	calculateY2(Y2, S2);

	predictedNumber = Y2;
	
	return predictedNumber;
}


double findMax(double* sequence, int length) {
	double max = 0;
	for (int i = 0; i < length; i++) {
		if (abs(sequence[i]) > max) {
			max = abs(sequence[i]);
		}
	}
	return max;
}


void scaleSequence(double* sequence, int length, double scale) {
	for (int i = 0; i < length; i++) {
		sequence[i] = (sequence[i] / scale);
	}
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция вычисления выхода функции синаптических преобразований нейронов скрытого слоя
*/
void calculateS1(double* S1, double* Xi, NeuralNetwork &neuralNetwork) {
	double** W1 = neuralNetwork.currFirstLayerWeightMatrix;
	double** WCont = neuralNetwork.currContextNeuronsWeightMatrix;

	double* contVal = neuralNetwork.currContextValues;
	double* T1 = neuralNetwork.currFirstLayerThresholds;

	int &m = neuralNetwork.neuronsNumber;
	int &p = neuralNetwork.windowSize;

	for (int currColNumber = 0; currColNumber < m; currColNumber++) {
		S1[currColNumber] = 0;

		for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
			S1[currColNumber] += Xi[currRowNumber] * W1[currRowNumber][currColNumber];
		}

		for (int currRowNumber = 0; currRowNumber < m; currRowNumber++) {
			S1[currColNumber] += contVal[currRowNumber] * WCont[currRowNumber][currColNumber];
		}

		S1[currColNumber] -= T1[currColNumber];
	}
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция вычисления выхода функции активации нейронов скрытого слоя
*/
void calculateY1(double* Y1, double* S1, NeuralNetwork &neuralNetwork) {
	int &m = neuralNetwork.neuronsNumber;

	for (int currRowNumber = 0; currRowNumber < m; currRowNumber++) {
		Y1[currRowNumber] = activateFunction(S1[currRowNumber]);
	}
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция вычисления выхода функции синаптических преобразований нейрона выходного слоя
*/
void calculateS2(double &S2, double* Y1, NeuralNetwork &neuralNetwork) {
	int &m = neuralNetwork.neuronsNumber;
	double* W2 = neuralNetwork.currSecondLayerWeightMatrix;
	double &T2 = neuralNetwork.currSecondLayerThreshold;

	S2 = 0;

	for (int currRowNumber = 0; currRowNumber < m; currRowNumber++) {
		S2 += Y1[currRowNumber] * W2[currRowNumber];
	}

	S2 -= T2;
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция вычисления выхода функции активации нейрона выходного слоя
*/
void calculateY2(double &Y2, double &S2) {
	Y2 = activateFunction(S2);
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция вычисления общей части для формул изменения весовых коэффициентов и порога выходного слоя
*/
void calculateGeneralPart1(double &generalPart1, double XRezi, double Y2, double S2, NeuralNetwork &neuralNetwork) {
	double &a = neuralNetwork.trainingCoefficient;
	double *XRez = neuralNetwork.rezultSample;

	generalPart1 = a * (Y2 - XRezi) * activateFunctionDerivative(S2);
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция изменения весовых коэффициентов матрицы выходного слоя
*/
void modifyW2(double generalPart1, double* Y1, NeuralNetwork &neuralNetwork) {
	int &m = neuralNetwork.neuronsNumber;
	double* W2 = neuralNetwork.currSecondLayerWeightMatrix;

	for (int i = 0; i < m; i++) {
		W2[i] -= generalPart1 * Y1[i];
	} 
}

/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция изменения порога выходного слоя
*/
void modifyT2(double generalPart1, NeuralNetwork &neuralNetwork) {
	double &T2 = neuralNetwork.currSecondLayerThreshold;

    T2 += generalPart1;
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция вычисления общих частей для формул изменения весовых коэффициентов и порогов
*/
void calculateGeneralParts2(double* generalParts2, double generalPart1, double* S1, NeuralNetwork &neuralNetwork) {
	int &m = neuralNetwork.neuronsNumber;
	double* W2 = neuralNetwork.currSecondLayerWeightMatrix;
	
	for (int j = 0; j < m; j++) {
        generalParts2[j] = generalPart1 * W2[j] 
            * activateFunctionDerivative(S1[j]);
    }
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция изменения весовых коэффициентов матрицы контекстных нейронов
*/
void modifyWCont(double* generalParts2, NeuralNetwork &neuralNetwork) {
	int &m = neuralNetwork.neuronsNumber;
	double* contVal = neuralNetwork.currContextValues;
	double** WCont = neuralNetwork.currContextNeuronsWeightMatrix;
	
	for (int k = 0; k < m; k++) {
		for (int j = 0; j < m; j++) {
			WCont[k][j] -= generalParts2[j] * contVal[k];
		}
	}
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция изменения весовых коэффициентов матрицы скрытого слоя
*/
void modifyW1(double* generalParts2, double* Xi, NeuralNetwork &neuralNetwork) {
	int &p = neuralNetwork.windowSize;
	int &m = neuralNetwork.neuronsNumber;
	double** W1 = neuralNetwork.currFirstLayerWeightMatrix;

	for (int i = 0; i < p; i++) {
		for (int j = 0; j < m; j++) {
			W1[i][j] -= generalParts2[j] * Xi[i];
			//W1[i][j] -= generalParts2[j] * Xi[p - 1 - i];
		}
	}
}

/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция изменения порогов скрытого слоя
*/
void modifyT1(double* generalParts2, NeuralNetwork &neuralNetwork) {
	int &m = neuralNetwork.neuronsNumber;
	double* T1 = neuralNetwork.currFirstLayerThresholds;

	for (int j = 0; j < m; j++) {
		T1[j] += generalParts2[j];
	}
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция активации
*/
double activateFunction(double x) {
	return mySin(myAtan(x));
	//return 0.1 * x;
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Производная функции активации
*/
double activateFunctionDerivative(double x) {
	return myCos(myAtan(x)) / (1 + (x * x));
	//return 0.1;
}


double linearActivateFunction(double x) {
	return 0.1 * x;
}


double linearActivateFunctionDerivative(double x) {
	return 0.1;
}


double mySin(double x) {
	return x - (x * x * x) / 6 + (x * x * x * x * x) / 120;
}


double myCos(double x) {
	return 1 - (x * x) / 2 + (x * x * x * x) / 24;
}


double myAtan(double x) {
	return x - (x * x * x) / 3 + (x * x * x * x * x) / 5 + (x * x * x * x * x * x * x) / 7;
}