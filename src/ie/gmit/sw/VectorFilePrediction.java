package ie.gmit.sw;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import ie.gmit.sw.*;
// VectorFilePrediction is used to parse and process the file that the User will input through Menu.
// This file will be parsed then compared to a already trained neural network, folded.nn or rprop.nn
public class VectorFilePrediction {
	private BufferedReader br;
	private String file, line, ngram;
	private int i, ngramSize;
	private double[] vector;

	public VectorFilePrediction(String file, int ngramSize, int vectorSize) {
		this.file = file;
		this.ngramSize = ngramSize;
		vector = new double[vectorSize];
	}

	//Parse file and pass it of to process()
	public void parse() throws Exception {
		System.out.print("\n Reading Dataset " + file + "... ");

		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(file))));
			//String line;
			System.out.println("\n Your Dataset file has been read and parsed!");
			System.out.println("\n Please wait while we process... \n ");
			while ((line = br.readLine()) != null) {
				process(line);
			}
		} catch (IOException e) {
			e.getMessage();
		}
	}


	public void process(String line) throws Exception {
		line = line.toLowerCase().replaceAll("[0-9]", "");

		// Sets vector index i to 0 on each iteration. 
		for (i = 0; i < vector.length; i++) {
			vector[i] = 0;
		}

		//For each n-gram, compute the index = ngram.hashCode() % vector.length and increment the vector index by 1
		for (i = 0; i < line.length() - ngramSize; i++) {
			ngram = line.substring(i, i + ngramSize);

			vector[ngram.hashCode() % vector.length]++;
		}
		
		// Normalize Vector Values
		vector = Utilities.normalize(vector, 0, 1);

		// Calls filePrediction function in NeuralNetwork, which will compare the file just parsed to a trained neural network folded.nn or rprop.nn
		new NeuralNetwork().filePrediction(vector);
	}
}
