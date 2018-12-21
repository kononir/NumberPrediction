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
};