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
	double scale;
} NeuralNetwork;

void initializeNeuralNetwork(NeuralNetwork &neuralNetwork, double* sequence, int sequenceSize);
void trainNeuralNetwork(NeuralNetwork &neuralNetwork);
double predictNextNumber(double* X, NeuralNetwork &neuralNetwork);

void calculateS1(double* S1, double* Xi, NeuralNetwork &neuralNetwork);
void calculateY1(double* Y1, double* S1, NeuralNetwork &neuralNetwork);
void calculateS2(double &S2, double* Y1, NeuralNetwork &neuralNetwork);
void calculateY2(double &Y2, double &S2);

void calculateGeneralPart1(double &generalPart1, double XRezi, double Y2, double S2, NeuralNetwork &neuralNetwork);
void modifyW2(double generalPart1, double* Y1, NeuralNetwork &neuralNetwork);
void modifyT2(double generalPart1, NeuralNetwork &neuralNetwork);
void calculateGeneralParts2(double* generalParts2, double generalPart1, double* S1, NeuralNetwork &neuralNetwork);
void modifyWCont(double* generalParts2, NeuralNetwork &neuralNetwork);
void modifyW1(double* generalParts2, double* Xi, NeuralNetwork &neuralNetwork);
void modifyT1(double* generalParts2, NeuralNetwork &neuralNetwork);

double activateFunction(double x);
double activateFunctionDerivative(double x);

double findAbsoluteMax(double* sequence, int length);
double scaleSequence(double* sequence, int sequenceSize);

double mySin(double x);
double myCos(double x);
double myAtan(double x);