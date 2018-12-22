// NumberPrediction.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "NumberPrediction.h"
#include <cmath>
#include <ctime>
#include <iostream>

using namespace std;

int main()
{
	int sequenceSize;
	double* sequence;
	NeuralNetwork neuralNetwork;

	try {
		int sequenceCode;

		cout << "Choose sequence: " << endl;
		cout << "1 - 1, 2, 3, ...\n2 - 1, 3, 5, 7, ... (periodic)\n3 - Fibonacci series\n4 - 3^(x+2)\n5 - x!\n6 - x^2\n";
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
			double temp[8] = { 9, 27, 81, 243, 729, 2187, 6561, 19683 };
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

		cout << "Enter number of neurons (L >= 1): ";
		cin >> neuralNetwork.neuronsNumber;

		cout << "Enter window size (p >= 1): ";
		cin >> neuralNetwork.windowSize;

		cout << "Input coefficient of training (0 < a <= 0.1, a <= e): ";
		cin >> neuralNetwork.trainingCoefficient;

		cout << "Enter maximum allowable number of training steps (N >= 1, N = 1000000): ";
		cin >> neuralNetwork.maximumAllowableNumberOfTrainingSteps;

		/*if (rectWidth <= 0 || rectWidth > imageWidth) {
			throw "Error - Invalid parameter: rectangle width";
		}*/
	}
	catch (const char* mesage) {
		cerr << mesage << endl;

		system("pause");
		return 1;
	}

	initializeNeuralNetwork(neuralNetwork, sequence, sequenceSize);

	trainNeuralNetwork(neuralNetwork);

	double* lastWindow = new double[neuralNetwork.windowSize];

	for (int windowNumberIndex = 0, sequenceNumberIndex = neuralNetwork.samplesNumber; windowNumberIndex < neuralNetwork.windowSize; windowNumberIndex++, sequenceNumberIndex++) {
		lastWindow[windowNumberIndex] = sequence[sequenceNumberIndex];
	}

	int predictedNumber = predictNextNumber(lastWindow, neuralNetwork);

	cout << endl << predictedNumber << endl;

	system("pause");
	return 0;
}


void initializeNeuralNetwork(NeuralNetwork &neuralNetwork, double* sequence, int sequenceSize) {
	double minWeight = -0.1;
	double maxWeight = 0.1;

	srand((unsigned int)time(0));

	neuralNetwork.currFirstLayerWeightMatrix = new double*[neuralNetwork.neuronsNumber];

	for (int currRowNumber = 0; currRowNumber < neuralNetwork.neuronsNumber; currRowNumber++) {
		neuralNetwork.currFirstLayerWeightMatrix[currRowNumber] = new double[neuralNetwork.windowSize];

		for (int currColNumber = 0; currColNumber < neuralNetwork.windowSize; currColNumber++) {
			neuralNetwork.currFirstLayerWeightMatrix[currRowNumber][currColNumber] 
				= (((double)rand() / RAND_MAX) * (maxWeight - minWeight)) + minWeight;
		}
	}

	neuralNetwork.currSecondLayerWeightMatrix = new double[neuralNetwork.neuronsNumber];

	for (int currRowNumber = 0; currRowNumber < neuralNetwork.neuronsNumber; currRowNumber++) {
		neuralNetwork.currSecondLayerWeightMatrix[currRowNumber] 
			= (((double)rand() / RAND_MAX) * (maxWeight - minWeight)) + minWeight;
	}

	neuralNetwork.currContextNeuronsWeightMatrix = new double*[neuralNetwork.neuronsNumber];

	for (int currRowNumber = 0; currRowNumber < neuralNetwork.neuronsNumber; currRowNumber++) {
		neuralNetwork.currContextNeuronsWeightMatrix[currRowNumber] = new double[neuralNetwork.neuronsNumber];
		memset(neuralNetwork.currContextNeuronsWeightMatrix[currRowNumber], 0, neuralNetwork.neuronsNumber * sizeof(double));
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

		for (int currColNumber = 0, sequenceNumberIndex = currRowNumber; currColNumber < neuralNetwork.windowSize; currColNumber++, sequenceNumberIndex++) {
			neuralNetwork.trainingSample[currRowNumber][currColNumber] = sequence[sequenceNumberIndex];
		}

		neuralNetwork.rezultSample[currRowNumber] = sequence[currRowNumber + neuralNetwork.windowSize];
	}
}


void trainNeuralNetwork(NeuralNetwork &neuralNetwork) {
	double** X = neuralNetwork.trainingSample;
	double** W1 = neuralNetwork.currFirstLayerWeightMatrix;
	double** WCont = neuralNetwork.currContextNeuronsWeightMatrix;

	int &L = neuralNetwork.samplesNumber;
	int &p = neuralNetwork.neuronsNumber;
	int &windS = neuralNetwork.windowSize;
	int maxNumOfSteps = neuralNetwork.maximumAllowableNumberOfTrainingSteps;
	int numOfSteps = 0;

	double* XRez = neuralNetwork.rezultSample;
	double* T1 = neuralNetwork.currFirstLayerThresholds;
	double* contVal = neuralNetwork.currContextValues;
	double* W2 = neuralNetwork.currSecondLayerWeightMatrix;
	double* rezult = neuralNetwork.rezultSample;
	double* S1 = new double[p];
	double* Y1 = new double[p];
    double* generalParts2 = new double[p];

	double &T2 = neuralNetwork.currSecondLayerThreshold;
	double &e = neuralNetwork.maximumAllowableError;
	double &a = neuralNetwork.trainingCoefficient;
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

			modifyW2(W2, generalPart1, Y1, neuralNetwork);
			modifyT2(T2, generalPart1);
            
            calculateGeneralParts2(generalParts2, generalPart1, S1, neuralNetwork);

			modifyWCont(WCont, generalParts2, neuralNetwork);
			modifyW1(W1, currImageryIndex, generalParts2, neuralNetwork);
			modifyT1(T1, generalParts2, neuralNetwork);

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

		cout /*<< "            " << "\r"*/ << currError << "\r";
	} while (currError > e && numOfSteps < maxNumOfSteps);

	delete S1;
	delete Y1;
	delete generalParts2;
}


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


void calculateS1(double* S1, double* X, NeuralNetwork &neuralNetwork) {
	double** W1 = neuralNetwork.currFirstLayerWeightMatrix;
	double** WCont = neuralNetwork.currContextNeuronsWeightMatrix;

	double* contVal = neuralNetwork.currContextValues;
	double* T1 = neuralNetwork.currFirstLayerThresholds;

	int &p = neuralNetwork.neuronsNumber;
	int &windS = neuralNetwork.windowSize;

	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
		S1[currRowNumber] = 0;

		for (int currColNumber = 0; currColNumber < windS; currColNumber++) {
			S1[currRowNumber] += X[currColNumber] * W1[currRowNumber][currColNumber];
		}

		for (int currColNumber = 0; currColNumber < p; currColNumber++) {
			S1[currRowNumber] += contVal[currColNumber] * WCont[currRowNumber][currColNumber];
		}

		S1[currRowNumber] -= T1[currRowNumber];
	}
}


void calculateY1(double* Y1, double* S1, NeuralNetwork &neuralNetwork) {
	int &p = neuralNetwork.neuronsNumber;

	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
		Y1[currRowNumber] = activateFunction(S1[currRowNumber]);
	}
}


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


void calculateY2(double &Y2, double &S2) {
	Y2 = activateFunction(S2);
}


void calculateGeneralPart1(double &generalPart1, int currImageryIndex, double Y2, double S2, NeuralNetwork &neuralNetwork) {
	double &a = neuralNetwork.trainingCoefficient;
	double *XRez = neuralNetwork.rezultSample;

	generalPart1 = a * (Y2 - XRez[currImageryIndex]) * activateFunctionDerivative(S2);
}


void modifyW2(double* W2, double generalPart1, double* Y1, NeuralNetwork &neuralNetwork) {
	int &p = neuralNetwork.neuronsNumber;

	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
		W2[currRowNumber] -= generalPart1 * Y1[currRowNumber];
	} 
}


void modifyT2(double &T2, double generalPart1) {
    T2 += generalPart1;
}


void calculateGeneralParts2(double* generalParts2, double generalPart1, double* S1, NeuralNetwork &neuralNetwork) {
	int &p = neuralNetwork.neuronsNumber;
	double* W2 = neuralNetwork.currSecondLayerWeightMatrix;
	
	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
        generalParts2[currRowNumber] = generalPart1 * W2[currRowNumber] 
            * activateFunctionDerivative(S1[currRowNumber]);
    }
}


void modifyWCont(double** WCont, double* generalParts2, NeuralNetwork &neuralNetwork) {
	int &p = neuralNetwork.neuronsNumber;
	double *contVal = neuralNetwork.currContextValues;
	
	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
		for (int currColNumber = 0; currColNumber < p; currColNumber++) {
			WCont[currRowNumber][currColNumber] -= generalParts2[currColNumber] * contVal[currRowNumber];
		}
	}
}


void modifyW1(double** W1, int currImageryIndex, double* generalParts2, NeuralNetwork &neuralNetwork) {
	int &windS = neuralNetwork.windowSize;
	int &p = neuralNetwork.neuronsNumber;

	double** X1 = neuralNetwork.trainingSample;

	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
		for (int currColNumber = 0; currColNumber < windS; currColNumber++) {
			W1[currRowNumber][currColNumber] -= generalParts2[currRowNumber] * X1[currImageryIndex][currColNumber];
		}
	}
}


void modifyT1(double* T1, double* generalParts2, NeuralNetwork &neuralNetwork) {
	int &p = neuralNetwork.neuronsNumber;

	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
		T1[currRowNumber] += generalParts2[currRowNumber];
	}
}


double activateFunction(double x) {
	//return sin(atan(x));
	return 0.1 * x;
}


double activateFunctionDerivative(double x) {
	//return cos(atan(x)) / (1 + (x * x));
	return 0.1;
}