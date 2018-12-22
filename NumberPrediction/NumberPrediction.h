#pragma once

typedef struct NeuralNetwork {
	double** trainingSample;
	double** currFirstLayerWeightMatrix;
	double** currContextNeuronsWeightMatrix;

	double* currSecondLayerWeightMatrix;
	double* rezultSample;
	double* currContextValues;
	double* currFirstLayerThresholds;

	int neuronsNumber;
	int windowSize;
	int samplesNumber;
	int maximumAllowableNumberOfTrainingSteps;
	int reachedNumberOfTrainingSteps;

	double currSecondLayerThreshold;
	double trainingCoefficient;
	double maximumAllowableError;
	double reachedError;
} NeuralNetwork;

void initializeNeuralNetwork(NeuralNetwork &neuralNetwork, double* sequence, int sequenceSize);
void trainNeuralNetwork(NeuralNetwork &neuralNetwork);
int predictNextNumber(double* X, NeuralNetwork &neuralNetwork);

void calculateS1(double* S1, double* X, NeuralNetwork &neuralNetwork);
void calculateY1(double* Y1, double* S1, NeuralNetwork &neuralNetwork);
void calculateS2(double &S2, double* Y1, NeuralNetwork &neuralNetwork);
void calculateY2(double &Y2, double &S2);
void calculateGeneralPart1(double &generalPart1, int currImageryIndex, double Y2, double S2, NeuralNetwork &neuralNetwork);
void modifyW2(double* W2, double generalPart1, double* Y1, NeuralNetwork &neuralNetwork);
void modifyT2(double &T2, double generalPart1);
void calculateGeneralParts2(double* generalParts2, double generalPart1, double* S1, NeuralNetwork &neuralNetwork);
void modifyWCont(double** WCont, double* generalParts2, NeuralNetwork &neuralNetwork);
void modifyW1(double** W1, int currImageryIndex, double* generalParts2, NeuralNetwork &neuralNetwork);
void modifyT1(double* T1, double* generalParts2, NeuralNetwork &neuralNetwork);
double activateFunction(double x);
double activateFunctionDerivative(double x);