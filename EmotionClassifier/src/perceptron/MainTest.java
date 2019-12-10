package perceptron;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import corpus.Corpus;
import corpus.Tweet;
import evaluation.Evaluator;

/**
 * This is the class with main method for testing and evaluating of the test
 * data. It sets the file name of test data and of the trained weights. 
 * It extracts tweet and their features, gets trained weights and calls the method
 * testModel of the MultiClassPerceptron Class, in order to test the model.
 * 
 * @author AysoltanGravina
 *
 */

public class MainTest {

	public static void main(String[] args) throws IOException {

		System.out.println("********** Getting data for testing takes a few minutes ... ***********");

		long startTime = System.currentTimeMillis();

		// Set the file names for the gold and predicted data.
		Corpus corpus = new Corpus();
		//corpus.setFileName("dev.csv");
		corpus.setFileName("test.txt");
		// Get tweets list with their given gold labels.
		corpus.extractTweetData();
		List<Tweet> tweetsList = corpus.getTweetsList();

		System.out.println("********** Testing of the model is running ... ***********");
		// Call the perceptron
		perceptronTest("trainedWeights.txt", tweetsList);

		// Get the TP, FN, FP values for each tweet.
		Evaluator eval = new Evaluator();
		Map<String, Evaluator> confusionMatrix = new HashMap<String, Evaluator>();
		confusionMatrix.putAll(eval.getTP_FN_FP(tweetsList));

		// Variables for calculation of the micro and macro accuracy.
		int allTp = 0;
		int allFn = 0;
		int allFp = 0;
		double macroAccuracy = 0.0;
		double macroPrecision = 0.0;
		double macroRecall = 0.0;
		double macroFScore = 0.0;

		// System.out.println(confusionMatrix.keySet());
		// Outputting the evaluation for each label.
		for (String key : confusionMatrix.keySet()) {
			int tp = confusionMatrix.get(key).getTp();
			allTp = allTp + tp;
			int fn = confusionMatrix.get(key).getFn();
			allFn = allFn + fn;
			int fp = confusionMatrix.get(key).getFp();
			allFp = allFp + fp;

			double precision = confusionMatrix.get(key).getPrecision(tp, fp);
			double recall = confusionMatrix.get(key).getRecall(tp, fn);
			double accuracy = confusionMatrix.get(key).getAccuracy(tp, fn, fp);
			macroAccuracy = macroAccuracy + confusionMatrix.get(key).getAccuracy(tp, fn, fp);
			macroPrecision = macroPrecision + confusionMatrix.get(key).getPrecision(tp, fp);
			macroRecall = macroRecall + confusionMatrix.get(key).getRecall(tp, fn);
			macroFScore = macroFScore + confusionMatrix.get(key).getFScore(precision, recall);

			System.out
					.println("********* Outputting of the evaluation results for " + key.toUpperCase() + " *********");
			System.out.println(key + "\t\t TP: " + tp + "\t FN: " + fn + "\t FP: " + fp);
			System.out.println("\t\t Accuracy: " + accuracy);
			System.out.println("\t\t Precision: " + precision);
			System.out.println("\t\t Recall: " + recall);
			System.out.println("\t\t FScore: " + confusionMatrix.get(key).getFScore(precision, recall));
			System.out.println("\n");
		}
		// Outputting the accuracy for all labels.
		// TNs were skipped in order to avoid the influence on calculation.
		System.out.println("********* Outputting of the macro/micro accuracy for all labels *********");
		System.out.println("Micro-Accuracy: " + eval.getAccuracy(allTp, allFn, allFp));
		System.out.println("Micro-Precision: " + eval.getPrecision(allTp, allFp));
		System.out.println("Micro-Recall: " + eval.getRecall(allTp, allFn));
		System.out.println(
				"Micro-FScore: " + eval.getFScore(eval.getPrecision(allTp, allFp), eval.getRecall(allTp, allFn)));
		System.out.println("\n");
		System.out.println("Macro-Accuracy: " + macroAccuracy / confusionMatrix.size());
		System.out.println("Macro-Precision: " + macroPrecision / confusionMatrix.size());
		System.out.println("Macro-Recall: " + macroRecall / confusionMatrix.size());
		System.out.println("Macro-FScore: " + macroFScore / confusionMatrix.size());
		System.out.println("\n");
		// Run time measurement
		long endTime = System.currentTimeMillis();
		System.out.println("Run time: " + (endTime - startTime) + " milliseconds");
	}

	/**
	 * This Method calls the MultiClassPerceptron
	 * 
	 * @param weightFileName
	 * @param tweetsList
	 * @throws IOException
	 */
	public static void perceptronTest(String weightFileName, List<Tweet> tweetsList) throws IOException {

		Map<String, Map<String, Double>> weightMap = readWeights(weightFileName);
		MultiClassPerceptron perceptron = new MultiClassPerceptron();
		for (Tweet tweetInstance : tweetsList) {
			perceptron.testModel(tweetInstance, weightMap);
		}
		System.out.println("********* Testing of the model is done. *********");
		System.out.println("\n");

	}

	/***
	 * This Method reads trained Weights Vectors from file and puts it into
	 * WeightMap for further processing. The source code of this Method was
	 * borrowed from:
	 * https://github.com/naveen2507/EmotionAnalysis/blob/master/src/com/ims/
	 * model/perceptron/PerceptronModelling.java
	 * 
	 * 
	 * @param fileName
	 * @return
	 * @throws IOException
	 */
	public static Map<String, Map<String, Double>> readWeights(String fileName) throws IOException {

		// int featuresNumber = ApplicationDetails.numOfFeatures + 1;
		Map<String, Map<String, Double>> weightMap = new HashMap<String, Map<String, Double>>();

		String line;
		BufferedReader br;
		try {
			Map<String, Double> weight = new HashMap<String, Double>();
			String tempCategory = "";

			br = new BufferedReader(new FileReader(fileName));
			while ((line = br.readLine()) != null) {

				if (line.isEmpty()) {
					continue;
				} else if (line.contains("------:::::::::::::::::::::::::::::------")) {
					String elem[] = line.split("------:::::::::::::::::::::::::::::------");
					try {
						weight.put(elem[0], Double.parseDouble(elem[1]));
					} catch (Exception e) {
						System.out.println(line);

						System.out.println(elem[0] + "--------" + elem[1]);

						break;
					}
				} else if (line.equalsIgnoreCase("------------------------------------")) {
					weightMap.put(tempCategory, weight);
					weight = new HashMap<String, Double>();
					tempCategory = "";
				} else {
					tempCategory = line;
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
		return weightMap;

	}
}
