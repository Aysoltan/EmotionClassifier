package perceptron;

import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import corpus.Corpus;
import corpus.Tweet;

/**
 * This is the class with main method for training of the model. It sets the
 * file name for training data. It extracts the tweets and their features and
 * trains the model with MultiClassPerceptron. The trained weights will be
 * stored into the file, in order to use for later testing and prediction.
 * 
 * @author AysoltanGravina
 *
 */
public class MainTrain {

	static String modelfileName = "trainedWeights.txt";

	public static void main(String[] args) throws IOException {

		long startTime = System.currentTimeMillis();

		// Set the file names for the train data.
		Corpus corpus = new Corpus();
		// corpus.setFileName("train.csv");
		corpus.setFileName("dev.csv");

		System.out.println("********** Getting data for training takes a few minutes ... ***********");
		// Get data for train
		corpus.extractTweetData();
		// Get tweets list with their given gold label.
		List<Tweet> tweetsList = corpus.getTweetsList();

		System.out.println("********** Model training is running ... ***********");
		perceptronTrain(tweetsList);
		System.out.println("********* Model training is done. *********");

		// Run time measurement
		long endTime = System.currentTimeMillis();
		System.out.println("Run time: " + (endTime - startTime) + " milliseconds");

	}

	public static void perceptronTrain(List<Tweet> tweetsList) throws IOException {

		// Define the number of maximum epoches or iterations
		// int MAX_ITER = 50;
		// int MAX_ITER = 100;
		//int MAX_ITER = 150;
		int MAX_ITER = 200;
		// int MAX_ITER = 250;

		// Initialize the weight vector
		Map<String, Map<String, Double>> weightMap = new HashMap<String, Map<String, Double>>();
		// Create multi-class perceptron
		MultiClassPerceptron perceptron = new MultiClassPerceptron();
		// train the model
		weightMap = perceptron.trainModel(tweetsList, MAX_ITER);
		// writes the weights into the file
		writeWeights(modelfileName, weightMap);

	}

	/**
	 * This Method writes trained Weights Vectors into the file, in order to use
	 * it for test data. The Source Code of this Method was borrowed from:
	 * https://github.com/naveen2507/EmotionAnalysis/blob/master/src/com/ims/
	 * model/perceptron/PerceptronModelling.java
	 * 
	 * @param fileName
	 * @param weightMap
	 * @throws IOException
	 */
	public static void writeWeights(String fileName, Map<String, Map<String, Double>> weightMap) throws IOException {

		FileWriter writer = new FileWriter(fileName);

		for (String label : weightMap.keySet()) {
			writer.append(label);
			writer.append("\n");
			Map<String, Double> weight = weightMap.get(label);
			for (String feature : weight.keySet()) {
				writer.append(feature + "------:::::::::::::::::::::::::::::------" + weight.get(feature));
				writer.append("\n");
			}
			writer.append("------------------------------------");
			writer.append("\n");
		}
		writer.flush();
		writer.close();

	}
}
