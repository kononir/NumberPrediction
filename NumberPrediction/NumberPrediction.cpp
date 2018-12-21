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

	system("pause");
	return 0;
}


void initializeNeuralNetwork(NeuralNetwork &neuralNetwork, double* sequence, int sequenceSize) {
	double minWeight = -1.0;
	double maxWeight = 1.0;

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

	neuralNetwork.samplesNumber = sequenceSize - neuralNetwork.windowSize + 1;

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

	double* T1 = neuralNetwork.currFirstLayerThresholds;
	double* contVal = neuralNetwork.currContextValues;
	double* W2 = neuralNetwork.currSecondLayerWeightMatrix;
	double* rezult = neuralNetwork.rezultSample;
	double* S1;
	double* Y1;

	int L = neuralNetwork.samplesNumber;
	int p = neuralNetwork.neuronsNumber;
	int windS = neuralNetwork.windowSize;
	int numOfSteps = 0;

	double T2 = neuralNetwork.currSecondLayerThreshold;
	double e = neuralNetwork.maximumAllowableError;
	double a = neuralNetwork.trainingCoefficient;
	double S2;
	double Y2;
	double E;

	do {
		for (int currImageryIndex = 0; currImageryIndex < L; currImageryIndex++) {
			S1 = calculateS1(X[currImageryIndex], contVal, T1, W1, WCont, p, windS);
			Y1 = calculateY1(S1, p);

			S2 = calculateS2(Y1, T2, W2, p);
			Y2 = calculateY2(S2);
		}
	} while (E > e);
}


double* calculateS1(double* Xi, double* contVal, double* T1, double** W1, double** WCont, int p, int windS) {
	double* S1 = new double[p];
	
	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
		S1[currRowNumber] = 0;

		for (int currColNumber = 0; currColNumber < windS; currColNumber++) {
			S1[currRowNumber] += Xi[currColNumber] * W1[currRowNumber][currColNumber];
		}

		for (int currColNumber = 0; currColNumber < p; currColNumber++) {
			S1[currRowNumber] += contVal[currColNumber] * WCont[currRowNumber][currColNumber];
		}

		S1[currRowNumber] -= T1[currRowNumber];
	}

	return S1;
}


double* calculateY1(double* S1, int p) {
	double* Y1 = new double[p];

	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
		Y1[currRowNumber] = activateFunction(S1[currRowNumber]);
	}

	return Y1;
}


double calculateS2(double* Y1, double T2, double* W2, int p) {
	double S2 = 0;

	for (int currRowNumber = 0; currRowNumber < p; currRowNumber++) {
		S2 += Y1[currRowNumber] * W2[currRowNumber];
	}

	return S2 - T2;
}


double calculateY2(double S2) {
	return activateFunction(S2);
}


double activateFunction(double x) {
	return sin(atan(x));
}


double activateFunctionDerivative(double x) {
	return cos(atan(x)) / (1 + (x * x));
}
