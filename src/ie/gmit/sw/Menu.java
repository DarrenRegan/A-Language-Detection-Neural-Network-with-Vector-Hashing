package ie.gmit.sw;

import java.util.Scanner;

import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;
import ie.gmit.sw.*;
//Darren Regan - G00326934 - Group C - 4th yr AI Project

public class Menu {
	private Scanner scanner = new Scanner(System.in);
	private String file = "wili-2018-Small-11750-Edited.txt";
	private boolean keepAlive = true;
	private String userFileInput;
	private NeuralNetwork neuralNetwork;
	private BasicNetwork basicNetwork;
	private MLDataSet mlDataSet;
	private int selection, epochs, ngramSize, inputSize;
	private double errorRate;

	
	public void menu() throws Exception {
		while (keepAlive) {
			System.out.println("\n A Language Detection Neural Network with Vector Hashing");
			System.out.println("***************************************************************");
			System.out.println("***  1 - Train a Neural Network with Cross Validation       ***");
			System.out.println("***  2 - Train a Neural Network with Resilient Propagation  ***");
			System.out.println("***  3 - Predict the Language of a File                     ***");
			System.out.println("***  4 - Quit                                               ***");
			System.out.println("***************************************************************");
			selection = scanner.nextInt();
	
			switch (selection) {
			case 1: //1 - Train a Neural Network with Cross Validation 
				// n-gram = 3, input = 1200, epoch = 25 94.85% acc 
				System.out.println("Enter n-gram size: (use 3 or 4)");
				ngramSize = scanner.nextInt();
	
				System.out.println("Enter input size: ");
				inputSize = scanner.nextInt();
	
				System.out.println("Enter number of epochs: ");
				epochs = scanner.nextInt();
	
				new VectorProcessor(ngramSize, inputSize).parse();
				neuralNetwork = new NeuralNetwork(inputSize, epochs, 0);
	
				basicNetwork = neuralNetwork.configureTopology();
				mlDataSet = neuralNetwork.generateDataSet();
	
				neuralNetwork.trainCrossValidation(basicNetwork, mlDataSet);
				neuralNetwork.predict(basicNetwork, mlDataSet);
				//TODO: add timer
				break;
			case 2: // 2 - Train a Neural Network with Resilient Propagation
				//Setup - n-gram = 3, input size = 1500, error rate = 0.0001 resulted in 99.2% Accuarcy in 3-4 minutes
				System.out.println("Enter n-gram size: (use 3 or 4)");
				ngramSize = scanner.nextInt();
	
				System.out.println("Enter input size:");
				inputSize = scanner.nextInt();
	
				System.out.println("Enter error rate: ");
				errorRate = scanner.nextDouble();
	
				new VectorProcessor(ngramSize, inputSize).parse();
				neuralNetwork = new NeuralNetwork(inputSize, 0, errorRate);
	
				basicNetwork = neuralNetwork.configureTopology();
				mlDataSet = neuralNetwork.generateDataSet();
	
				neuralNetwork.trainResilientPropagation(basicNetwork, mlDataSet);
				neuralNetwork.predict(basicNetwork, mlDataSet);
				break;
			case 3: // 3 - Predict the Language of a File   
				System.out.println("Enter file name: ");
				userFileInput = scanner.next();
	
				System.out.println("Enter n-gram size: (use 3 or 4)");
				ngramSize = scanner.nextInt();
	
				System.out.println("Enter input size: ");
				inputSize = scanner.nextInt();
	
				new VectorFilePrediction(userFileInput, ngramSize, inputSize).parse();
				break;
			case 4: //  4 - Quit   
				System.out.println("Exiting Program... ");
				keepAlive = false;
				break;
			default:
				System.out.println("Error: Invalid option, Please try again... ");
				break;
				}
			}
		}
}
//TODO Let user input n-gram size, input size, epoch size, error rate and file.
//TODO Fix No enum constant ie.gmit.sw.Language.OldEnglish  exception
/*A command-line menu-driven user interface allows users to pass in any
parameters required by the application. The UI should report the topology
structure, the training time, and test statistics (sensitivity and specificity).
The UI should also allow a text file to be specified with live data to be
classified.
 * */
