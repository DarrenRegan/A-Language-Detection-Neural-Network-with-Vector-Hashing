# DarrenRegan-A-Language-Detection-Neural-Network-with-Vector-Hashing
### 4th yr AI Project
-------------------------------------------------------
A Language Detection Neural Network with Vector Hashing
-------------------------------------------------------

--- Running Application ---

   -) java –cp ./language-nn.jar ie.gmit.sw.Runner
   -) Menu will appear with 4 options
   -) 1. Trains a Neural Network with Cross Validation
   -) 2. Trains a Neural Network with Resilient Propagation
   -) 3. Predicts the Language of a File
   -) 4. Exits Program   

-- Resilient Propagation Setup
Setup: n-gram = 3, input size = 500, error rate = 0.0001 
** resulted in 99.18% Accuarcy in 46 Iterations and less then 3 minutes

-- 5-Hold Cross Validation Setup
Setup: n-gram = 3, input size = 1150, epochs = 25 //Ran out of time to find the perfect setup for Cross Validation

--- RATIONALE ---

----------------------------------------------------------------------------------
1. N-gram
----------------------------------------------------------------------------------
   -) N-gram is inputed by User, recommended 3 or 4
   -) A list of ngrams is created from the line of text + ngramSize from user
   -) After list is created, then loop over each n-gram in list
   -) For each n-gram, compute the index: index = ngram.hashCode() % vector.length
   -) And Increment the vector index by 1
   -) Index value will then be passed to Normalize Vector Values
   -) Index will be set back to 0 after each iteration
   -) Process Repeats until finished

----------------------------------------------------------------------------------
2. Network Topology
----------------------------------------------------------------------------------
This part took alot of time with testing different activation functions,
below is the setup i ended up with as i ran out of time to test even more.
I used http://heatonresearch-site.s3-website-us-east-1.amazonaws.com/javadoc/encog-3.3/org/encog/engine/network/activation/package-summary.html
as a basis for testing Activation Functions, i pretty much went down line by line testing each one.
ActivationElliottSymmetric in the hidden layer ended up with the best results after testing every option.

The Activation Functions didn't have as huge of an impact as i thought they would, it seems that removing as much
baggage from the initial file parsing has the greatest increase in performance by far.

   -) Input Neurons at 400
   -) Output Neurons at 235
   -) HiddenLayers at around 100 - 120
   --- Activation Functions --- 
   -) Input Layer - ActivationReLU
   -) Single Hidden Layer - ActivationElliottSymmetric
   -) Output Layer - ActivationSoftMax

//Resilient Propagation Setup
Setup - n-gram = 3, input size = 1500, error rate = 0.0001 resulted in 99.2% Accuarcy in 3-4 minutes
//5-Hold Cross Validation Setup
Setup - n-gram = 3, input size = 1150, epochs = 25 //Ran out of time to find the perfect setup for Cross Validation

----------------------------------------------------------------------------------
3. Size of Hashing Feature Vector
----------------------------------------------------------------------------------
   -) Vector Size is set by the user, while developing the program i hard coded ranges between 50 - 1500
   -) The Vector Size varied considerably depending on what activation function i was testing, 
      aswell as testing Cross Validation and Resilient Propagation separately.
   -) The process for Vector hashing goes as follows:
      1. Process inputed file i.e wili-2018-Small-11750-Edited.txt
      2. Look over txt file line by line and write it out in a set of numbers
      3. Do n-grams stuffed explained above
      4. Compute Index index = ngram.hashCode() % vector.length
      5. Normalize the Vector Values i.e vector = Utilities.normalize(vector, 0.0, 1.0);
      6. Write out the vector values to a CSV File using dft.format(vector[index])
      7. Write out the language numbers in the same row aswell using dfl.format(vector[index])

   -) There is 235 Languages in the txt file, using language.values() shows each number.
   -) If for example 5 was the correct answer Arabic would be the answer, which would be written out like (0, 0, 0, 0, 1, 0, ...0)etc
   -) Vector.length + labels (Number of Elements in each row) Size of Vector + 235 Labels

----------------------------------------------------------------------------------
4. Number of Hidden Layers and the Number of Nodes in each layer
----------------------------------------------------------------------------------
   -) After testing alot i ended up with a single hidden layer with the ActivationElliottSymmetric
   -) I was using ActivationTANH and ActivationReLU for a day or so, but i seen ActivationElliottSymmetric as a Computationally efficient alternative to ActivationTANH it ended up giving slighty better results on my machine.
   -) ActivationElliottSymmetric/ActivationTANH Outputs in the range [-1, 1] and is derivable
   -) TANH may be more suitable but the difference is negligible
   -) The Nodes for the hidden layer are set to the initial Input Neurons / 4
      So Input Neurons set at 400 results in 100 Hidden Layer Nodes

----------------------------------------------------------------------------------
5. Activation Functions in each Layer
----------------------------------------------------------------------------------
   --- Input Layer ---
   -) ActivationReLU - Rectified Linear Unit
      Ramp Activation Function. ActivationReLU has a high and low threshold.
      If the high threshold is exceeded a fixed value is returned	
      If the low threshold is exceeded another fixed value is returned

   --- Hidden Layer ---
   -) ActivationElliottSymmetric / TanH / Hyperbolic Tangent
      The hyperbolic tangent activation function takes the curved shape of the
      hyperbolic tangent. This activation function produces both positive and negative output. 
      Use this activation function if both negative and positive output is desired.

   --- Output Layer ---
   -) ActivationSoftMax
      This layer never changed as it was recommended for the output layer,
      SoftMax seems to mostly only be used in output layer, i couldn't find many examples
      of it being used outside the output layer.

----------------------------------------------------------------------------------
6. References - All these links along with moodle labs/lectures were used extensively throughout the project.
----------------------------------------------------------------------------------
/*
* REFERENCES EXTREMELY USEFUL: Alot of code is mix and match between these links and labs on moodle
* 1. https://www.heatonresearch.com/encog/
* 2. https://s3.amazonaws.com/heatonresearch-books/free/encog-3_3-quickstart.pdf
* 3. https://s3.amazonaws.com/heatonresearch-books/free/Encog3Java-User.pdf
* 4. https://s3.amazonaws.com/heatonresearch-books/free/encog-3_3-devguide.pdf
* 5. https://github.com/jeffheaton/encog-java-examples/tree/master/src/main/java/org/encog/examples
* 6. https://github.com/jeffheaton/encog-java-examples/blob/master/src/main/java/org/encog/examples/neural/cross/CrossValidateSunspot.java
* 7. http://heatonresearch-site.s3-website-us-east-1.amazonaws.com/javadoc/encog-3.3/org/encog/engine/network/activation/ActivationSigmoid.html
* 8. http://heatonresearch-site.s3-website-us-east-1.amazonaws.com/javadoc/encog-3.3/org/encog/engine/network/activation/package-summary.html
*/
