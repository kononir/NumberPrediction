// NumberPrediction.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

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

	initializeNeuralNetwork(neuralNetwork, sequence, sequenceSize);

	trainNeuralNetwork(neuralNetwork);

	double* lastWindow = new double[neuralNetwork.windowSize];

	for (int windowNumberIndex = 0; windowNumberIndex < neuralNetwork.windowSize; windowNumberIndex++) {
		lastWindow[windowNumberIndex] = sequence[windowNumberIndex + neuralNetwork.samplesNumber];
	}

	int predictedNumber = predictNextNumber(lastWindow, neuralNetwork);

	cout << endl << predictedNumber << endl;

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
	memset(neuralNetwork.currContextValues, 0, neuralNetwork.neuronsNumber * sizeof(double));

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
	int &p = neuralNetwork.neuronsNumber;
	int maxNumOfSteps = neuralNetwork.maximumAllowableNumberOfTrainingSteps;
	int numOfSteps = 0;

	double* XRez = neuralNetwork.rezultSample;
	double* contVal = neuralNetwork.currContextValues;
	double* S1 = new double[p];
	double* Y1 = new double[p];
    double* generalParts2 = new double[p];

	double &e = neuralNetwork.maximumAllowableError;
	double S2;
	double Y2;
    double generalPart1;
	double currError;
	double delta;

	do {
		for (int currImageryIndex = 0; currImageryIndex < L; currImageryIndex++) {
			calculateS1(S1, X[currImageryIndex], neuralNetwork);
			calculateY1(Y1, S1, neuralNetwork);

			calculateS2(S2, Y1, neuralNetwork);
			calculateY2(Y2, S2);
            
            calculateGeneralPart1(generalPart1, currImageryIndex, Y2, S2, neuralNetwork);

			modifyW2(generalPart1, Y1, neuralNetwork);
			modifyT2(generalPart1, neuralNetwork);
            
            calculateGeneralParts2(generalParts2, generalPart1, S1, neuralNetwork);

			modifyWCont(generalParts2, neuralNetwork);
			modifyW1(generalParts2, X[currImageryIndex], neuralNetwork);
			modifyT1(generalParts2, neuralNetwork);

			memcpy(contVal, Y1, p * sizeof(double));
		}

		currError = 0;

		for (int currImageryIndex = 0; currImageryIndex < L; currImageryIndex++) {
			calculateS1(S1, X[currImageryIndex], neuralNetwork);
			calculateY1(Y1, S1, neuralNetwork);

			calculateS2(S2, Y1, neuralNetwork);
			calculateY2(Y2, S2);

			delta = (Y2 - XRez[currImageryIndex]);
			currError += (delta * delta) / 2;

			memcpy(contVal, Y1, p * sizeof(double));
		}

		numOfSteps++;

		cout << currError << /*"\r"*/ endl;
	} while (currError > e && numOfSteps < maxNumOfSteps);

	delete S1;
	delete Y1;
	delete generalParts2;
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция предсказания следующего элемента последовательности с помощью нейронной сети
*/
int predictNextNumber(double* X, NeuralNetwork &neuralNetwork) {
	int predictedNumber;
	int &p = neuralNetwork.neuronsNumber;

	double* S1 = new double[p];
	double* Y1 = new double[p];

	double S2;
	double Y2;

	calculateS1(S1, X, neuralNetwork);
	calculateY1(Y1, S1, neuralNetwork);

	calculateS2(S2, Y1, neuralNetwork);
	calculateY2(Y2, S2);

	predictedNumber = (int)Y2;
	
	return predictedNumber;
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

	int &p = neuralNetwork.neuronsNumber;
	int &windS = neuralNetwork.windowSize;

	for (int currColNumber = 0; currColNumber < p; currColNumber++) {
		S1[currColNumber] = 0;

		for (int currRowNumber = 0; currRowNumber < windS; currRowNumber++) {
			S1[currColNumber] += Xi[currRowNumber] * W1[currRowNumber][currColNumber];
		}

		for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
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
	int &p = neuralNetwork.neuronsNumber;

	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
		Y1[currRowNumber] = activateFunction(S1[currRowNumber]);
	}
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция вычисления выхода функции синаптических преобразований нейрона выходного слоя
*/
void calculateS2(double &S2, double* Y1, NeuralNetwork &neuralNetwork) {
	int &p = neuralNetwork.neuronsNumber;
	double* W2 = neuralNetwork.currSecondLayerWeightMatrix;
	double &T2 = neuralNetwork.currSecondLayerThreshold;

	S2 = 0;

	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
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
void calculateGeneralPart1(double &generalPart1, int currImageryIndex, double Y2, double S2, NeuralNetwork &neuralNetwork) {
	double &a = neuralNetwork.trainingCoefficient;
	double *XRez = neuralNetwork.rezultSample;

	generalPart1 = a * (Y2 - XRez[currImageryIndex]) * activateFunctionDerivative(S2);
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция изменения весовых коэффициентов матрицы выходного слоя
*/
void modifyW2(double generalPart1, double* Y1, NeuralNetwork &neuralNetwork) {
	int &p = neuralNetwork.neuronsNumber;
	double* W2 = neuralNetwork.currSecondLayerWeightMatrix;

	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
		W2[currRowNumber] -= generalPart1 * Y1[currRowNumber];
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
	int &p = neuralNetwork.neuronsNumber;
	double* W2 = neuralNetwork.currSecondLayerWeightMatrix;
	
	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
        generalParts2[currRowNumber] = generalPart1 * W2[currRowNumber] 
            * activateFunctionDerivative(S1[currRowNumber]);
    }
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция изменения весовых коэффициентов матрицы контекстных нейронов
*/
void modifyWCont(double* generalParts2, NeuralNetwork &neuralNetwork) {
	int &p = neuralNetwork.neuronsNumber;
	double* contVal = neuralNetwork.currContextValues;
	double** WCont = neuralNetwork.currContextNeuronsWeightMatrix;
	
	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
		for (int currColNumber = 0; currColNumber < p; currColNumber++) {
			WCont[currRowNumber][currColNumber] -= generalParts2[currColNumber] * contVal[currRowNumber];
		}
	}
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция изменения весовых коэффициентов матрицы скрытого слоя
*/
void modifyW1(double* generalParts2, double* Xi, NeuralNetwork &neuralNetwork) {
	int &windS = neuralNetwork.windowSize;
	int &p = neuralNetwork.neuronsNumber;
	double** W1 = neuralNetwork.currFirstLayerWeightMatrix;

	for (int currRowNumber = 0; currRowNumber < windS; currRowNumber++) {
		for (int currColNumber = 0; currColNumber < p; currColNumber++) {
			W1[currRowNumber][currColNumber] -= generalParts2[currColNumber] * Xi[currRowNumber];
			//W1[currRowNumber][currColNumber] -= generalParts2[currColNumber] * Xi[windS - 1 - currRowNumber];
		}
	}
}

/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция изменения порогов скрытого слоя
*/
void modifyT1(double* generalParts2, NeuralNetwork &neuralNetwork) {
	int &p = neuralNetwork.neuronsNumber;
	double* T1 = neuralNetwork.currFirstLayerThresholds;

	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
		T1[currRowNumber] += generalParts2[currRowNumber];
	}
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Функция активации
*/
double activateFunction(double x) {
	//return sin(atan(x));
	return 0.1 * x;
}


/*
* author: Novitskiy Vladislav
* group: 621701
* description: Производная функции активации
*/
double activateFunctionDerivative(double x) {
	//return cos(atan(x)) / (1 + (x * x));
	return 0.1;
}