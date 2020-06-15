package ie.gmit.sw;
//Darren Regan - G00326934 - Group C - 4th yr AI Project
import java.io.File;
import java.math.RoundingMode;
import java.text.DecimalFormat;

import org.encog.engine.network.activation.ActivationElliott;
import org.encog.engine.network.activation.ActivationElliottSymmetric;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationSoftMax;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.buffer.MemoryDataLoader;
import org.encog.ml.data.buffer.codec.CSVDataCODEC;
import org.encog.ml.data.buffer.codec.DataSetCODEC;
import org.encog.ml.data.folded.FoldedDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.ml.train.strategy.RequiredImprovementStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.cross.CrossValidationKFold;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.csv.CSVFormat;
import org.encog.util.simple.EncogUtility;
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
public class NeuralNetwork {
	//Variables
	private static int inputs = 400; //Change this to the number of input neurons
	private static final int outputs = 235; //Change this to the number of output neurons
	private int hiddenLayers = inputs / 4;
	private static final double MAX_ERROR = 0.0017; //Changing between  0.001 - 0.0035 for atm
	private Language[] langs;
	private File csvOutputFile  = new File("data.csv");
	private int k = 5, epoch = 0;
	private int correct = 0, total = 0, resultIndex = -1, idealIndex = 0, actual = 0;
	private int i, epochs, ideal, inputSize;
	private double percent, limit = -1, errorRate;
	private DecimalFormat df;
	private BasicNetwork basicNetwork;
	private CrossValidationKFold trainFolded;
	private MLData mlActual, mlIdeal, mlOutput, mlPredction;
	private MLDataSet mlDataSet;
	private MLTrain mlTrain;
	private DataSetCODEC dataSetCODEC;
	private FoldedDataSet folded;
	private MemoryDataLoader mDL;
	private ResilientPropagation resilientPropagation;

	
	public NeuralNetwork() {
		//Default Constructor
	}

	public NeuralNetwork(int inputSize, int epochs, double errorRate) {
		this.inputSize = inputSize;
		this.epochs = epochs;
		this.errorRate = errorRate;
	}
	
	// Generates a Dataset from data.csv, which is created with VectorProcessor class
	public MLDataSet generateDataSet() {
		//Read the CSV file "data.csv" into memory. Encog expects your CSV file to have input + output number of columns.
		dataSetCODEC = new CSVDataCODEC(csvOutputFile, CSVFormat.DECIMAL_POINT, false, inputSize, outputs, false);
		mDL = new MemoryDataLoader(dataSetCODEC);
		mlDataSet = mDL.external2Memory();

		return mlDataSet;
	}
	
	// Create a Network (configures network topology etc)
	// http://heatonresearch-site.s3-website-us-east-1.amazonaws.com/javadoc/encog-3.3/org/encog/engine/network/activation/package-summary.html
	// Use ReLU for all layers except the output layer which should use softmax. Change the activation functions if the level of accuracy remains poor.
	// (ActivationSigmoid) The sigmoid activation function takes on a sigmoidal shape. Only positive numbers are generated. Do not use this activation function if negative number output is desired.
	// (ActivationTANH) The hyperbolic tangent activation function takes the curved shape of the hyperbolic tangent. 
	// (ActivationTANH) This activation function produces both positive and negative output. Use this activation function if both negative and positive output is desired.
	// (ActivationElliottSymmetric) - Computationally efficient alternative to ActivationTANH. Its output is in the range [-1, 1], and it is derivable.
	public BasicNetwork configureTopology() {
		basicNetwork = new BasicNetwork();
		// input layer
		//basicNetwork.addLayer(new BasicLayer(new ActivationSigmoid(), true, inputSize));
		//basicNetwork.addLayer(new BasicLayer(null, true, inputSize));
		//basicNetwork.addLayer(new BasicLayer(null, false, inputSize));
		basicNetwork.addLayer(new BasicLayer(new ActivationReLU(), true, inputSize));
		
		// hidden layer
		//basicNetwork.addLayer(new BasicLayer(new ActivationReLU(), true, hiddenLayers, 600));
		//basicNetwork.addLayer(new BasicLayer(new ActivationReLU(), true, hiddenLayers, 400));
		//basicNetwork.addLayer(new BasicLayer(new ActivationReLU(), true, hiddenLayers));
		//basicNetwork.addLayer(new BasicLayer(new ActivationTANH(), true, hiddenLayers, 600));
		basicNetwork.addLayer(new BasicLayer(new ActivationElliottSymmetric(), true, hiddenLayers, 400));
		//basicNetwork.addLayer(new BasicLayer(new ActivationSigmoid(), true, hiddenLayers));
		//basicNetwork.addLayer(new BasicLayer(new ActivationElliott(), true, hiddenLayers));
		
		// output layer
		// Outputs 0 or 1 MAKE SURE TO NORMAL VECTORS WITHIN THAT RANGE
		basicNetwork.addLayer(new BasicLayer(new ActivationSoftMax(), false, outputs));
		
		basicNetwork.getStructure().finalizeStructure();
		basicNetwork.reset();

		return basicNetwork;
	}

	//TODO: split ResilientPropagation & CrossValidationKFold into 2 seperate functions
	/* Trains the Neural Network using k-fold Cross-Validation: Common Values are k=5, k=10 and k=n
	 * As 'k' gets larger, the difference in size between the training set and the resampling subsets get smaller.
	 * As this difference decreases, the bias of the tecnique becomes smaller
	 * The number of Epochs should be defined by user in Menu, else default will be 0 or 1 and increment in while loop
	 */ 
	public void trainCrossValidation(BasicNetwork basicNetwork, MLDataSet mlDataSet) {
		folded = new FoldedDataSet(mlDataSet);
		mlTrain = new ResilientPropagation(basicNetwork, folded);
		trainFolded = new CrossValidationKFold(mlTrain, k);

		// Formatting
		df = new DecimalFormat("#.######"); // six places of percision i.e 0.983455
		df.setRoundingMode(RoundingMode.CEILING);
		
		do {
			trainFolded.iteration();

			epoch++;
			//System.out.println("Epoch #" + epoch + " Error:" + trainFolded.getError());
			System.out.println("Epoch: " + epoch);
			System.out.println("Error: " + df.format(trainFolded.getError()));
		} while (epoch < epochs); //while ( trainFolded.getError() > MAX_ERROR && epoch < 25);

		System.out.println("\nTraining 5-Hold Cross Validation complete! ");
		System.out.println("Number of epochs: " + epoch);
		System.out.println("Error Rate: " + df.format(trainFolded.getError()));

		// Save the Neural Network to folded.nn and close the training with finishTraining()
		Utilities.saveNeuralNetwork(basicNetwork, "./folded.nn");
		trainFolded.finishTraining();
	}
	
	
	// Trains the neural network using resilient propagation, the neural network will run #
	// until the error rate falls below whatever the user enters or the default value
	public void trainResilientPropagation(BasicNetwork basicNetwork, MLDataSet mlDataSet) {
		resilientPropagation = new ResilientPropagation(basicNetwork, mlDataSet);
		resilientPropagation.addStrategy(new RequiredImprovementStrategy(5));

		EncogUtility.trainToError(resilientPropagation, errorRate);

		Utilities.saveNeuralNetwork(basicNetwork, "./rprop.nn");
		resilientPropagation.finishTraining();
	}
	
	// Get prediction which will determine the accuacy of the Neural Network
	public void predict(BasicNetwork basicNetwork, MLDataSet mlDataSet) {
		for (MLDataPair mlDataPair : mlDataSet) {
			mlActual = basicNetwork.compute(mlDataPair.getInput());
			mlIdeal = mlDataPair.getIdeal();

			for (i = 0; i < mlActual.size(); i++) {
				if (mlActual.getData(i) > 0 && (resultIndex == -1 || (mlActual.getData(i) > mlActual.getData(resultIndex)))) {
					resultIndex = i;
					//actual = i;
				}
			}

			for (i = 0; i < mlIdeal.size(); i++) {
				if (mlIdeal.getData(i) == 1) {
					ideal = i;
					//idealIndex = i;

					if (resultIndex == ideal) {
						correct++;
					}
				}
			}

			total++;
		}

		// Formatting
		df = new DecimalFormat("##.##"); // two places of percision i.e 00.09
		df.setRoundingMode(RoundingMode.CEILING);
		
		//System.out.println("[INFO] Testing complete. \n Accuracy: " + (correct / total) * 100 + " %");
		percent = (double) correct / (double) total;

		System.out.println("\n[INFO]: Testing complete.");
		System.out.println("Correct: " + correct + "/" + total);
		System.out.println("Accuracy: " + df.format(percent * 100) + "%");
	}
	
	
	//TODO: Compared file inputed by user to either folded.nn or rprop.nn
	// Predicts the language of a file inputed by User, compares to neural network folded.nn or rprop.nn
	public void filePrediction(double[] vector) {
		mlPredction = new BasicMLData(vector);
		mlPredction.setData(vector);

		//basicNetwork = Utilities.loadNeuralNetwork("./folded.nn");
		basicNetwork = Utilities.loadNeuralNetwork("./rprop.nn");
		mlOutput = basicNetwork.compute(mlPredction);

		for (i = 0; i < mlOutput.size(); i++) {
			if (mlOutput.getData(i) > limit) {
				limit = mlOutput.getData(i);
				actual = i;
			}
		}
		
		//There is 235 Languages in the txt file, using language.values() shows each number.
		//If for example 5 was the correct answer Arabic would be the answer, which would be written out like (0, 0, 0, 0, 1, 0, ...0)etc
		langs = Language.values();

		System.out.println("Predicted language: " + langs[actual].toString());
	}

}//end







//TODO: 1. TRAIN NEURAL NETWORK using 5-Fold Cross Validation and Resilient Propagation
//		2. Determine Accuarcy of Neural Network
//		3. Format the accuarcy and get the prediction, print to screen.
//      4. Let user enter a file
