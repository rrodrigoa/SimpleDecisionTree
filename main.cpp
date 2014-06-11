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

// Keyboard representation for calculating distance between keys
typedef vector<vector<short>> MognoKeyboard;

// Create a qwerty keyboard representation
MognoKeyboard* QwertyKeyboard(){
	MognoKeyboard* keyboard = new vector<vector<short>>();
	keyboard->push_back(vector<short>());
	keyboard->at(0).push_back('q');
	keyboard->at(0).push_back('w');
	keyboard->at(0).push_back('e');
	keyboard->at(0).push_back('r');
	keyboard->at(0).push_back('t');
	keyboard->at(0).push_back('y');
	keyboard->at(0).push_back('u');
	keyboard->at(0).push_back('i');
	keyboard->at(0).push_back('o');
	keyboard->at(0).push_back('p');

	keyboard->push_back(vector<short>());
	keyboard->at(1).push_back('a');
	keyboard->at(1).push_back('s');
	keyboard->at(1).push_back('d');
	keyboard->at(1).push_back('f');
	keyboard->at(1).push_back('g');
	keyboard->at(1).push_back('h');
	keyboard->at(1).push_back('j');
	keyboard->at(1).push_back('k');
	keyboard->at(1).push_back('l');
	
	keyboard->push_back(vector<short>());
	keyboard->at(2).push_back('z');
	keyboard->at(2).push_back('x');
	keyboard->at(2).push_back('c');
	keyboard->at(2).push_back('v');
	keyboard->at(2).push_back('b');
	keyboard->at(2).push_back('n');
	keyboard->at(2).push_back('m');

	return keyboard;
}

// Return the distance of two chars based on a keyboard
int CharDistance(short A, short B, MognoKeyboard keyboard){
	vector<short>::iterator a_it = keyboard[0].begin();
	vector<short>::iterator b_it = keyboard[0].begin();

	int k = 0;
	int aIndex = 0;
	while (k != keyboard.size()){
		if (aIndex == keyboard[k].size()){
			aIndex = 0;
			k++;
		}
		else if (keyboard[k][aIndex] == A){
			break;
		}
		else{
			aIndex++;
		}
	}
	int bIndex = 0;
	k = 0;
	while (k != keyboard.size()){
		if (bIndex == keyboard[k].size()){
			bIndex = 0;
			k++;
		}
		else if (keyboard[k][bIndex] == B){
			break;
		}
		else{
			bIndex++;
		}
	}

	int distance = aIndex - bIndex;
	return (distance < 0 ? -distance : distance);
}

// Calculate the keyboard distance of to strings, giving a keyboard
int KeyboardDistance(string a, string b, MognoKeyboard keyboard){
	int m = a.size() + 1;
	int n = b.size() + 1;

	vector<vector<int>> matrix;
	vector<int> colVector0;
	int col = 0;
	int line = 0;
	for(col = 0; col < m; col++){
		if(col < m){
			colVector0.push_back(col);
		}
	}
	matrix.push_back(colVector0);
	for(line = 0; line < n-1; line++){
		matrix.push_back(vector<int>(m));
	}

	for (int line = 1; line < n; line++){
		for (int col = 1; col < m; col++){
			if (b[line-1] == a[col-1]){
				matrix[line][col] = matrix[line - 1][col - 1];
			}
			else{
				int minValue = min(matrix[line - 1][col], matrix[line][col - 1]);
				minValue = min(minValue, matrix[line - 1][col - 1]);
				int distanceValue = CharDistance(a[col-1], b[line-1], keyboard);
				minValue += distanceValue;
#ifdef DEBUG
				printf("%d for %c, %c\n", distanceValue, a[col-1], b[line-1]);
#endif
				matrix[line][col] = minValue;
			}
		}
	}

	return matrix[n-1][m-1];
}

struct SuffixNode{
	// Sulffix children
	map<short, struct SuffixNode*> *children;
	// Node rank
	int rank;
};

struct SuffixNode* CreateSuffixTree(vector<pair<int,string>> *words){
	struct SuffixNode* head = (struct SuffixNode*)calloc(sizeof(struct SuffixNode),1);
	head->children = new map<short, struct SuffixNode*>();
	head->rank = 0;

	// For each word to be processed
	for(int wordIndex = 0; wordIndex < words->size(); wordIndex++){
		struct SuffixNode* headTemp = head;

		// For each char in the word
		for(int charIndex = 0; charIndex < words->at(wordIndex).second.size(); charIndex++){
			// Try to find the char in the map
			map<short,struct SuffixNode*>::iterator mapIt = headTemp->children->find((short)words->at(wordIndex).second[charIndex]);

			// If char is not there
			if(mapIt == headTemp->children->end()){
				// Create a new SuffixNode and save at that key with the rank
				struct SuffixNode* newNode = (struct SuffixNode*)calloc(sizeof(struct SuffixNode), 1);
				newNode->children = new map<short, struct SuffixNode*>();

				newNode->rank = words->at(wordIndex).first;
				headTemp->children->insert(pair<short,struct SuffixNode*>((short)words->at(wordIndex).second[charIndex], newNode));
				headTemp = newNode;
			}else{
				// Just sum the rank for final total
				headTemp = mapIt->second;
				headTemp->rank += words->at(wordIndex).first;
			}
		}
	}

	return head;
}

void PrintSuffixTree(struct SuffixNode* head){
	queue<struct SuffixNode*> q;
	q.push(head);

	string prefixString = "node_";
	int nodeIndex = 0;

	string outputGraphviz = "digraph G {\n";

	while (q.empty() == false){
		struct SuffixNode* front = q.front();
		q.pop();
		int headNodeIndex = nodeIndex;
		nodeIndex++;

		if(front->children != NULL){
			map<short, struct SuffixNode*>::iterator mapIt = front->children->begin();
			while (mapIt != front->children->end()){
					int childNodeIndex = nodeIndex + q.size();
					// add in the queue and print information
					q.push(mapIt->second);
					outputGraphviz += "\t " + prefixString + to_string(headNodeIndex) + " -> " + prefixString + to_string(childNodeIndex) + " [label=\"rank=" + to_string(mapIt->second->rank) + " key=" + (char)mapIt->first + "\"];\n";
					mapIt++;
			}
		}
	}
	// Finalize the string
	outputGraphviz += "}\n";
	printf("%s\n", outputGraphviz.c_str());
}

vector<pair<int,string>>* ReadFrequencyCIN(){
	vector<pair<int,string>>* words = new vector<pair<int,string>>();

	int freq = 0;
	string word;
	while(cin >> freq){
		cin >> word;
		words->push_back(pair<int,string>(freq, word));
	}

	return words;
}

int main(){
	vector<pair<int,string>> *words = ReadFrequencyCIN();
	//vector<pair<int,string>> *words = new vector<pair<int,string>>();
	//words->push_back(pair<int,string>(1, "abc"));
	//words->push_back(pair<int,string>(2, "b"));
	struct SuffixNode* tree = CreateSuffixTree(words);
	PrintSuffixTree(tree);

	return 0;
}
