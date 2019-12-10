package corpus;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import cmu.arktweetnlp.Twokenize;

/**
 * This class defines the data structure Tweet which contains tweet itself, gold
 * and predicted labels and features. Additional there is the object method
 * which converts the tweet into the list of tokens using arktweet library.
 * 
 * @author AysoltanGravina
 *
 */
public class Tweet {

	// Attributes
	private String tweet;
	private List<Token> tokensList; // necessary for later feature extraction
	private String goldLabel;
	private String predictedLabel;

	private Map<String, Double> features;


	// Getters and Setters
	public String getTweet() {
		return tweet;
	}

	public void setTweet(String tweet) {
		this.tweet = tweet;
	}

	public List<Token> getTokensList() {
		return tokensList;
	}

	public void setTokensList(List<Token> tokensList) {
		this.tokensList = tokensList;
	}

	public String getGoldLabel() {
		return goldLabel;
	}

	public void setGoldLabel(String goldLabel) {
		this.goldLabel = goldLabel;
	}

	public String getPredictedLabel() {
		return predictedLabel;
	}

	public void setPredictedLabel(String predictedLabel) {
		this.predictedLabel = predictedLabel;
	}

	public Map<String, Double> getFeatures() {
		return features;
	}

	public void setFeatures(Map<String, Double> features) {
		this.features = features;
	}


	/*
	 * Object method of Tweet tokenizes the tweet into Token data type necessary
	 * for later feature extraction. This method was implemented on the basis of
	 * https://github.com/felipebravom/AffectiveTweets/blob/master/src/main/
	 * java/affective/core/Utils.java and it uses arktweetnlp.Twokenize library.
	 */
	public List<String> tokenize(String tweet) {

		List<Token> tokensList = new ArrayList<Token>();
		List<String> tokens = new ArrayList<String>();

		for (String word : Twokenize.tokenizeRawTweetText(tweet)) {

			String cleanWord = word;

			// Replace URLs to a generic URL
			if (word.matches("http.*|ww\\..*")) {
				cleanWord = "http://www.url.com";
			}

			// Replaces user mentions to a generic user
			else if (word.matches("@.*")) {
				cleanWord = "@user";
			}

			Token token = new Token();
			token.setToken(cleanWord);
			tokensList.add(token);
			
			tokens.add(cleanWord);
		}
		setTokensList(tokensList);
		
		return tokens;
	}

}
