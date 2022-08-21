#include <iostream>
#include <random>
#include <ctime>
#include <vector>
#include "Perceptron.h"

using namespace std;

int32_t  main(){
  srand(time(NULL));
  
  Perceptron p = Perceptron{2};

  vector<vector<float>> inputs  {
      {0, 1},
      {0, 1},
      {0, 0},
  };

  vector<float> outputs{
      1,
      1,
      0,
  };

  p.fit(50, inputs, outputs);

  for(const vector<float>& item: inputs){
    cout << "Salida: " << p.predict(item) << endl; 
  }

  return EXIT_SUCCESS;
};