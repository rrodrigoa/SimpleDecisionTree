#include <stdio.h>
#include <functional>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <queue>
#include <stack>
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <set>
using namespace std;

// The equal conditional function for simpleEntropy
template <typename T>
bool equalConditionalFunction(T expectedValue, T featureValue){
	if (expectedValue == featureValue)
	{
		return true;
	}
	else
	{
		return false;
	}
}

// The bigger than conditional function for simpleEntropy
template <typename T>
bool biggerThanConditionalFunction(T expectedValue, T featureValue){
	if (featureValue > expectedValue){
		return true;
	}
	else{
		return false;
	}
}

// Calculate the entropy of the response vector
template<typename T>
double simpleEntropy(vector<T> responseVector, bool (*conditionFunction)(T,T), T inputCondition){
	double numOfFalses = 0;
	double numOfTrues = 0;
	double totalNumberOfValues = responseVector.size();
	double entropy = 0;

	if (totalNumberOfValues != 0){
		vector<T>::iterator it = responseVector.begin();

		while (it != responseVector.end()){
			if (conditionFunction(inputCondition, *it)){
				numOfTrues++;
			}
			else{
				numOfFalses++;
			}
			it++;
		}

		double falseRate = numOfFalses / totalNumberOfValues;
		double trueRate = numOfTrues / totalNumberOfValues;

		entropy = -falseRate*log(falseRate) / log(2) - trueRate*log(trueRate) / log(2);
	}

	return entropy;
}

// Calculate the entropy of the response vector based on a comparison method and a feature vector
template<typename T>
double conditionalEntropy(vector<bool> responseVector, vector<T> featureVector, T inputCondition, double* totalCount){
	double numOfFalses = 0;
	double numOfTrues = 0;
	double totalNumberOfValues = 0;
	double entropy = 0;

	if (responseVector.size() == featureVector.size() && (responseVector.size() != 0) && (featureVector.size() != 0)){
		vector<bool>::iterator it_response = responseVector.begin();
		vector<T>::iterator it_feature = featureVector.begin();

		while (it_feature != featureVector.end()){
			// If this feature meet the condition function
			if (inputCondition == *it_feature){
				if (*it_response == true){
					numOfTrues++;
				}
				else{
					numOfFalses++;
				}
				totalNumberOfValues++;
			}
			it_response++;
			it_feature++;
		}

		double falseRate = numOfFalses / totalNumberOfValues;
		double trueRate = numOfTrues / totalNumberOfValues;

		*totalCount = totalNumberOfValues;
		entropy = falseRate != 0 ? -falseRate*log(falseRate) / log(2) : 0;
		entropy += trueRate != 0 ? -trueRate*log(trueRate) / log(2) : 0;
	}

	return entropy;
}

// Calculate the Gain for a descrete feature set
template<typename T>
double discreteGrain(vector<T> differentDescreteValues, vector<T> featureVector, vector<bool> responseVector){
	bool(*conditionalFunction)(bool, bool) = &equalConditionalFunction<bool>;
	double entropy = simpleEntropy(responseVector, conditionalFunction, true);

	// For each discrete value on feature
	vector<T>::iterator it = differentDescreteValues.begin();
	double total = featureVector.size();
	while (it != differentDescreteValues.end()){
		double totalCountForDiscreteValue = 0;
		double condEntropy = conditionalEntropy<T>(responseVector, featureVector, *it, &totalCountForDiscreteValue);
		entropy = entropy - (totalCountForDiscreteValue / total)*condEntropy;
		it++;
	}

	return entropy;
}

// Calculate the gain for a continuous feature set
template<typename T>
double continuousGain(vector<T> featureVector, vector<bool> responseVector, T* setDivisionValue){
	vector<pair<T, bool> > orderedVector;
	for (int i = 0; i < featureVector.size(); i++){
		orderedVector.push_back(pair<T, bool>(featureVector[i], responseVector[i]));
	}
	sort(orderedVector.begin(), orderedVector.end());

	double total = featureVector.size();

	int currentIndex = 0;

	double tempSetDivisionGain = 0;
	double totalCountForDescreteValue = 0;
	double middleValue = 0;

	double maxSetDivisionGain = 0;
	double maxMiddleValue = 0;
	bool(*conditionalFunction)(T, T) = &biggerThanConditionalFunction<T>;

	// Verify all values finding a boundary in value
	while (currentIndex < total-1){
		// Change in value for the response vector, means possible cut value
		if (orderedVector[currentIndex].second != orderedVector[currentIndex+1].second){
			middleValue = (orderedVector[currentIndex].first + orderedVector[currentIndex + 1].first) / 2;
			tempSetDivisionGain = simpleEntropy<T>(featureVector, conditionalFunction, middleValue);

			// Found a bigger information gain boundary
			if (tempSetDivisionGain > maxSetDivisionGain){
				maxSetDivisionGain = tempSetDivisionGain;
				maxMiddleValue = middleValue;
			}
		}
		currentIndex++;
	}

	*setDivisionValue = maxMiddleValue;
	return maxSetDivisionGain;
}

// Create a vector with the unique values of a feature vector
template <typename T>
vector<T>* uniqueVector(vector<T> featureVector){
	vector<T>* uniqueVector = new vector<T>();

	for (int index = 0; index < featureVector.size(); index++){
		vector<T>::iterator it = lower_bound(uniqueVector->begin(), uniqueVector->end(), featureVector[index]);
		if (it == uniqueVector->end() || *it != featureVector[index]){
			uniqueVector->push_back(featureVector[index]);
		}
	}

	return uniqueVector;
}

struct Node{
	bool isDiscrete;
	map<string, struct Node*> *discreteChildren;
	struct Node* leftNode;
	struct Node* rightNode;
	double threshold;
	int CVVFeatureIndes;

	vector<bool> *decisionVector;
	vector<vector<double>> *CVV;
	vector<string> *DVV;
};

struct Node* createNewNode(bool isDiscrete, vector<bool> *decisionVector, vector<vector<double>> *CVV, vector<string> *DVV){
	struct Node* returnNode = (struct Node*)calloc(1, sizeof(struct Node));
	returnNode->isDiscrete = isDiscrete;
	returnNode->discreteChildren = new map<string, struct Node*>();
	returnNode->leftNode = NULL;
	returnNode->rightNode = NULL;
	returnNode->threshold = 0;
	returnNode->CVVFeatureIndes = 0;
	returnNode->decisionVector = decisionVector;
	returnNode->DVV = DVV;
	returnNode->CVV = CVV;

	return returnNode;
}

struct Node* createNewNode(){
	struct Node* returnNode = (struct Node*)calloc(1, sizeof(struct Node));
	returnNode->isDiscrete = false;
	returnNode->discreteChildren = new map<string, struct Node*>();
	returnNode->leftNode = NULL;
	returnNode->rightNode = NULL;
	returnNode->threshold = 0;
	returnNode->CVVFeatureIndex = 0;
	returnNode->decisionVector = new vector<bool>();
	returnNode->DVV = new vector<string>();
	returnNode->CVV = new vector<vector<double>>();

	return returnNode;
}

// TODO: template
void createMapNode(vector<bool> decisionVector, vector<vector<double>> CVV, vector<string> DVV, struct Node* headNode){
	vector<string> *uniqueValues = uniqueVector<string>(DVV);

	map<string, struct Node*> *mapToNode = new map<string, struct Node*>();
	for (int DVVIndex = 0; DVVIndex < DVV.size(); DVVIndex++){
		map<string, struct Node*>::iterator DVVit = mapToNode->find(DVV[DVVIndex]);
		struct Node* childNode = NULL;

		if (DVVit != mapToNode->end()){
			childNode = DVVit->second;
		}
		else{
			childNode = createNewNode();
			mapToNode->insert(pair<string, struct Node*>(DVV[DVVIndex], childNode));
		}
	}
	
	headNode->discreteChildren = mapToNode;
}

// TODO: change to template T,Z
void createSmallerEqualAndBiggerNode(vector<bool> decisionVector, vector<vector<double>> CVV, vector<string> DVV, double CVVCut, int CVVMaxGainIndex,
struct Node* headNode){
	vector<vector<double>> *leftCVV = new vector<vector<double>>();
	vector<vector<double>> *rightCVV = new vector<vector<double>>();

	vector<bool> *leftDecisionVector = new vector<bool>();
	vector<bool> *rightDecisionVector = new vector<bool>();

	vector<string> *leftDVV = new vector<string>();
	vector<string> *rightDVV = new vector<string>();
	
	// Cut Left and right
	for (int featureIndex = 0; featureIndex < CVV.size(); featureIndex++){
		if (featureIndex != CVVMaxGainIndex){
			int featureIndexMinusCount = featureIndex;
			if (CVVMaxGainIndex < featureIndex){
				featureIndexMinusCount--;
			}
			if (featureIndexMinusCount == leftCVV->size()){
				leftCVV->push_back(vector<double>());
				rightCVV->push_back(vector<double>());
			}

			for (int valueIndex = 0; valueIndex < CVV[CVVMaxGainIndex].size(); valueIndex++){
				if (CVV[CVVMaxGainIndex][valueIndex] <= CVVCut){
					leftCVV->at(featureIndexMinusCount).push_back(CVV[featureIndexMinusCount][valueIndex]);
				}
				else{
					rightCVV->at(featureIndexMinusCount).push_back(CVV[featureIndexMinusCount][valueIndex]);
				}
			}
		}
		else{
			for (int valueIndex = 0; valueIndex < CVV[CVVMaxGainIndex].size(); valueIndex++){
				if (CVV[CVVMaxGainIndex][valueIndex] <= CVVCut){
					if (DVV.size() != 0){
						leftDVV->push_back(DVV[valueIndex]);
					}
					leftDecisionVector->push_back(decisionVector[valueIndex]);
				}
				else{
					if (DVV.size() != 0){
						rightDVV->push_back(DVV[valueIndex]);
					}
					rightDecisionVector->push_back(decisionVector[valueIndex]);
				}
			}
		}
	}

	struct Node* leftNode = createNewNode(false, leftDecisionVector, leftCVV, leftDVV);
	struct Node* rightNode = createNewNode(false, rightDecisionVector, rightCVV, rightDVV);

	headNode->isDiscrete = false;
	headNode->threshold = CVVCut;
	headNode->CVVFeatureIndes = CVVMaxGainIndex;
	headNode->leftNode = leftNode;
	headNode->rightNode = rightNode;

	// TODO: release memory of internal vectors
	// Release CVV
	// Release DVV
	// Release decisionVector
}

void PrintDecisionTree(struct Node* head){

}

int main(){
	double error = 0.0001;


	//--------------------------------------------------------------
	vector<bool> responseVector;
	responseVector.push_back(false);
	responseVector.push_back(false);
	responseVector.push_back(true);
	responseVector.push_back(true);
	responseVector.push_back(true);
	responseVector.push_back(false);
	responseVector.push_back(true);
	responseVector.push_back(false);
	responseVector.push_back(true);
	responseVector.push_back(true);
	responseVector.push_back(true);
	responseVector.push_back(true);
	responseVector.push_back(true);
	responseVector.push_back(false);

	bool(*conditionalFunction)(bool, bool) = &equalConditionalFunction<bool>;
	double entropy = simpleEntropy(responseVector, conditionalFunction, true);
	double diff = entropy - 0.9402859;
	if (diff <= error){
		printf("PASSED - Test simpleEntropy\n");
	}
	else{
		printf("FAILED - Test simpleEntropy\n");
	}

	//--------------------------------------------------------------
	vector<bool> windyFeature;
	windyFeature.push_back(false);
	windyFeature.push_back(true);
	windyFeature.push_back(false);
	windyFeature.push_back(false);
	windyFeature.push_back(false);
	windyFeature.push_back(true);
	windyFeature.push_back(true);
	windyFeature.push_back(false);
	windyFeature.push_back(false);
	windyFeature.push_back(false);
	windyFeature.push_back(true);
	windyFeature.push_back(true);
	windyFeature.push_back(false);
	windyFeature.push_back(true);

	double totalCountFalse = 0;
	double totalCountTrue = 0;

	double entropyFalse = conditionalEntropy<bool>(responseVector, windyFeature, false, &totalCountFalse);
	double entropyTrue = conditionalEntropy<bool>(responseVector, windyFeature, true, &totalCountTrue);
	double gainWindy = entropy - (totalCountFalse / (totalCountFalse + totalCountTrue))*entropyFalse - (totalCountTrue / (totalCountTrue + totalCountFalse))*entropyTrue;

	diff = gainWindy - 0.048127;
	if (diff <= error){
		printf("PASSED - Test conditionalEntropy\n");
	}
	else{
		printf("FAILED - Test conditionalEntropy\n");
	}

	//--------------------------------------------------------------
	vector<bool> differentDescreteValuesBool;
	differentDescreteValuesBool.push_back(false);
	differentDescreteValuesBool.push_back(true);
	gainWindy = discreteGrain<bool>(differentDescreteValuesBool, windyFeature, responseVector);

	diff = gainWindy - 0.048127;
	if (diff <= error){
		printf("PASSED - Test discreteGain boolean\n");
	}
	else{
		printf("FAILED - Test discreteGain boolean\n");
	}

	//--------------------------------------------------------------
	vector<string> differenteDescreteValuesTemperature;
	differenteDescreteValuesTemperature.push_back("hot");
	differenteDescreteValuesTemperature.push_back("mild");
	differenteDescreteValuesTemperature.push_back("cold");

	vector<string> temperatureFeature;
	temperatureFeature.push_back("hot");
	temperatureFeature.push_back("hot");
	temperatureFeature.push_back("hot");
	temperatureFeature.push_back("mild");
	temperatureFeature.push_back("cold");
	temperatureFeature.push_back("cold");
	temperatureFeature.push_back("cold");
	temperatureFeature.push_back("mild");
	temperatureFeature.push_back("cold");
	temperatureFeature.push_back("mild");
	temperatureFeature.push_back("mild");
	temperatureFeature.push_back("mild");
	temperatureFeature.push_back("hot");
	temperatureFeature.push_back("mild");

	double gainTemperature = discreteGrain<string>(differenteDescreteValuesTemperature, temperatureFeature, responseVector);

	diff = gainTemperature - 0.029222;
	if (diff <= error){
		printf("PASSED - Test discreteGain string\n");
	}
	else{
		printf("FAILED - Test discreteGain string\n");
	}

	//--------------------------------------------------------------
	vector<double> doubleHumidityFeature;
	doubleHumidityFeature.push_back(0.72);
	doubleHumidityFeature.push_back(0.91);
	doubleHumidityFeature.push_back(0.9);
	doubleHumidityFeature.push_back(0.87);
	doubleHumidityFeature.push_back(0.68);

	vector<bool> responseDoubleFeature;
	responseDoubleFeature.push_back(true);
	responseDoubleFeature.push_back(false);
	responseDoubleFeature.push_back(false);
	responseDoubleFeature.push_back(false);
	responseDoubleFeature.push_back(true);

	double setDivisionValue = 0;
	double gainDoubleHumidity = continuousGain<double>(doubleHumidityFeature, responseDoubleFeature, &setDivisionValue);

	diff = gainTemperature - 0.97095;
	if (diff <= error){
		printf("PASSED - Test continuousGain\n");
	}
	else{
		printf("FAILED - Test continuousGain\n");
	}

	//--------------------------------------------------------------
	// MAIN structure

	int decisionInt = 0;
	string stringFeature;
	double doubleFeature;

	vector<bool> decisionVector;
	vector<vector<double>> CVV;
	vector<string> DVV;
	int indexFeature = 0;

	queue<struct Node*> q;

	while (cin >> decisionInt){
		// Read decision vector (boolean) decisionVector
		decisionVector.push_back(decisionInt == 0 ? false : true);
		// Read all discrete feature vectors (string) vector<string> = DVV
		cin >> stringFeature;
		DVV.push_back(stringFeature);

		// Read all continuous feature vectors (double) vector<vector<double> > = CVV
		for (int index = 0; index < 64; index++){
			cin >> doubleFeature;
			if (CVV.size() == index){
				CVV.push_back(vector<double>());
			}
			CVV[index].push_back(doubleFeature);
		}
	}

	// create node (decisionVector, CVV, DVV) as head
	struct Node* head = createNewNode(false, &decisionVector, &CVV, &DVV);

	// push node in queue
	q.push(head);

	// there are still nodes to be learned
	while (q.empty() == false){
		struct Node* front = (struct Node*)q.front();
		q.pop();
		
		// calculate CVVMaxGain as being the biggest gain in CVV
		double CVVGain = 0;
		double CVVMaxGain = 0;
		double CVVCut = 0;
		double CVVMaxCut = 0;
		int CVVMaxIndex = 0;
		for (int index = 0; front->CVV != NULL && index < front->CVV->size(); index++){
			CVVGain = continuousGain<double>(front->CVV->at(index), *front->decisionVector, &CVVCut);

			// Save the feature with the max gain for all continuous features
			if (CVVGain > CVVMaxGain){
				CVVMaxGain = CVVGain;
				CVVMaxCut = CVVCut;
				CVVMaxIndex = index;
			}
		}

		// calculate DVVGain as being the biggest gain in DVV
		vector<string> *uniqueValue = uniqueVector<string>(*front->DVV);
		double DVVGain = 0;
		if (front->DVV->size() != 0){
			DVVGain = discreteGrain<string>(*uniqueValue, *front->DVV, *front->decisionVector);
		}

		// C is the max of CVV and DVV gain
		if (CVVMaxGain >= DVVGain && CVVMaxGain != 0){
			// cut all vectors with separation for <= and > nodes
			struct Node* leftNode = NULL;
			struct Node* rightNode = NULL;
			createSmallerEqualAndBiggerNode(*front->decisionVector, *front->CVV, *front->DVV, CVVMaxCut, CVVMaxIndex, front);
			q.push(front->leftNode);
			q.push(front->rightNode);
		}
		else if(DVVGain != 0){
			createMapNode(*front->decisionVector, *front->CVV, *front->DVV, front);
			map<string, struct Node*>::iterator it = front->discreteChildren->begin();

			while(it != front->discreteChildren->end()){
				q.push(it->second);
				it++;
			}
		}
	}

	// print decision tree in graph form

	return 0;
}
