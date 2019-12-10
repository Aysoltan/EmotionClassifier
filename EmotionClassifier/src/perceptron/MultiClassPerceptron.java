package perceptron;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import corpus.Tweet;

/**
 * MultiClass Perceptron Implementation for training and testing. Algorithm is
 * built such it can handle sparse data. For the first the weight vectors will
 * be initialized with bias for each label and refilled with zero for each new
 * feature string. After that winning perceptron will be determined. If the
 * prediction was wrond the weight vectors will be updated. The Source Code of
 * this Method was borrowed from:
 * https://github.com/naveen2507/EmotionAnalysis/blob/master/src/com/ims/model/
 * perceptron/MultiClassPerceptron.java
 * 
 * @author adjusted by AysoltanGravina
 *
 */

public class MultiClassPerceptron {

	public Map<String, Map<String, Double>> trainModel(List<Tweet> tweetsList, int MAX_ITER) {

		Map<String, Double> tempWeights;

		// initialize the weight vectors in form: <happy :<"0" : 0.1>> for all
		// labels
		Map<String, Map<String, Double>> weightVectors = initializeWeightVecors();

		// Max_ITER is a number of epochs
		for (int i = 0; i < MAX_ITER; i++) {
			//System.out.println(i+1 + ". Iteration");
			// Iterate through the tweet list and for each tweet instance
			for (Tweet tweetInstance : tweetsList) {
				
				// Get the features for each instance (in form <FeatureName :
				// FeatureValue>)
				Map<String, Double> featureVector = tweetInstance.getFeatures();

				// Put new features and their value into weightVectors with bias
				// (in form <Label: <FeatureName : 0.0>>)
				weightVectors = getWeightVecors(featureVector.keySet(), weightVectors);
				//System.out.println("weightsVectors: " + weightVectors.entrySet());
				// Put bias into feature Vector.
				featureVector = putBiasInFeatureVector(featureVector);
				//System.out.println("Features: " + tweetInstance.getFeatures());
				// Necessary for further computing of the winning perceptron.
				tweetInstance.setFeatures(featureVector);

				// get arg_max = maximal value of dot product for weights and
				// features within all labels.
				tweetInstance = getWinningPerceptron(tweetInstance, weightVectors);
				

				// If prediction is wrong, update the weight vector
				if (!(tweetInstance.getGoldLabel().equalsIgnoreCase(tweetInstance.getPredictedLabel()))) {

					// raise the score of gold/right answer:
					tempWeights = getAdjustedWeight(weightVectors.get(tweetInstance.getGoldLabel()), true,
							featureVector);
					// and put the gold/right label with updated weight vector
					weightVectors.put(tweetInstance.getGoldLabel(), tempWeights);

					// lower the score of wrong answer:
					tempWeights = getAdjustedWeight(weightVectors.get(tweetInstance.getPredictedLabel()), false,
							featureVector);
					// and put the predicted/wrong label with updated weight
					// vector
					weightVectors.put(tweetInstance.getPredictedLabel(), tempWeights);

				}

			}
		}
		return weightVectors;
	}

	/**
	 * Initializing of the weight vectors with only bias for each label for the
	 * first and refill it with feature names and values later in order to
	 * handle sparse data.
	 **/
	public Map<String, Map<String, Double>> initializeWeightVecors() {
		Map<String, Map<String, Double>> weightVectors = new HashMap<String, Map<String, Double>>();
		List<String> labels = Arrays.asList("anger", "disgust", "fear", "happy", "love", "sad", "surprise", "trust");
		// Initialize weight vectors for each label with bias
		for (String label : labels) {
			Map<String, Double> initWeightVectors = new HashMap<String, Double>();
			// Put bias into weight vector
			initWeightVectors.put("0", 0.1);
			weightVectors.put(label, initWeightVectors);
		}
		return weightVectors;
	}

	/**
	 * This method completes the weight vector initialization. Each label gets
	 * all existing feature names and the value 0.0.
	 **/
	public Map<String, Map<String, Double>> getWeightVecors(Set<String> featureKeys,
			Map<String, Map<String, Double>> weightVectors) {

		List<String> labels = Arrays.asList("anger", "disgust", "fear", "happy", "love", "sad", "surprise", "trust");

		for (String label : labels) {

			// In order to handle the sparse data, all "existing" feature names
			// will be added into weight vector.
			Map<String, Double> labelWeightVector = weightVectors.get(label);

			for (String featureStr : featureKeys) {
				if (labelWeightVector.containsKey(featureStr)) {
					continue;
				} else {
					// for every new feature, refilling initialized weight
					// vectors with with this feature and 0.0 value
					labelWeightVector.put(featureStr, 0.0);
				}
			}
			weightVectors.put(label, labelWeightVector);
		}
		return weightVectors;

	}

	/**
	 * Putting Bias into Feature Vector also in order to take it into matrix
	 * computation by calculation of the winning vectors on the basis of feature
	 * names.
	 **/
	public Map<String, Double> putBiasInFeatureVector(Map<String, Double> featureVector) {
		// Putting of the bias with name "0" and value 0.0
		featureVector.put("0", 0.1);
		return featureVector;

	}

	/**
	 * For each label determines the winning label on the basis of the matrix
	 * arithmetic for weight vector and feature vector and sets it as predicted
	 **/
	public Tweet getWinningPerceptron(Tweet tweetInstance, Map<String, Map<String, Double>> weightVectors) {

		double argmax = 0.0;
		Map<String, Double> featureVector = tweetInstance.getFeatures();

		// for each label gets the weight vector and feature vector and calls
		// argMaxY method for both.
		for (String label : weightVectors.keySet()) {
			double argmax_y = argMaxY(weightVectors.get(label), featureVector);
			if (argmax_y > argmax) {
				argmax = argmax_y;
				tweetInstance.setPredictedLabel(label);
			} 
		}
		return tweetInstance;

	}

	/**
	 * Matrix arithmetic for the weight vector w and feature vector x. Gets the
	 * value of each feature name in weight vector and feature vector and
	 * returns the sum of product for it (w * x) as one double number.
	 **/
	public Double argMaxY(Map<String, Double> weights, Map<String, Double> featureVector) {

		double argmax_y = 0.0;
		for (String feature : featureVector.keySet()) {

			if (weights.containsKey(feature)) {
				double w = weights.get(feature);
				double x = featureVector.get(feature);
				argmax_y = argmax_y + (w * x);
			}

		}

		return argmax_y;

	}

	/**
	 * Update the weight vectors. If the prediction was true sum the
	 * vectors,wrong subtract
	 **/
	public Map<String, Double> getAdjustedWeight(Map<String, Double> weightVector, boolean action,
			Map<String, Double> featureVector) {

		for (String feature : featureVector.keySet()) {
			double w = weightVector.get(feature);
			double x = featureVector.get(feature);
			// put the sum of the weight value for right/gold label(=w) and
			// value of the feature(=x) into weightVector in form:
			// <featureName : updated value>.
			if (action) {
				w = w + x;
			} else {
				w = w - x;
			}
			weightVector.put(feature, w);

		}

		return weightVector;

	}

	/**
	 * Test the model. Takes the trained weight vectors and computes winning
	 * perceptron
	 **/
	public void testModel(Tweet tweetInstance, Map<String, Map<String, Double>> weightVectors) {

		Map<String, Double> featureVector = tweetInstance.getFeatures();
		featureVector = putBiasInFeatureVector(featureVector);
		getWinningPerceptron(tweetInstance, weightVectors);
	}
}
