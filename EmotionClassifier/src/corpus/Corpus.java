package corpus;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import features.TweetFeatures;

/**
 * This class reads the corpus data and extracts the values necessary for the
 * evaluation, feature extraction and classifying. Than the values will be
 * stored into Tweet data structure.
 * 
 * @author AysoltanGravina
 *
 */
public class Corpus {

	// Attributes
	List<Tweet> tweetsList; // necessary for the iteration through all tweets
							// during the evaluation calculation.
	// File name for training
	private String fileName;

	// Getters Setters
	public List<Tweet> getTweetsList() {
		return tweetsList;
	}

	public void setTweetsList(List<Tweet> tweetsList) {
		this.tweetsList = tweetsList;
	}

	public String getFileName() {
		return fileName;
	}

	public void setFileName(String fileName) {
		this.fileName = fileName;
	}

	/*
	 * Object method of Corpus which extracts the data necessary for training or
	 * testing. It reads from the given file and extracts tweet itself as a
	 * string and their gold label.
	 */
	public void extractTweetData() {

		TweetFeatures tf = new TweetFeatures();
		tf.initializeTagger();

		List<String> negationList = Arrays.asList("aren’t", "arent", "cannot", "cant", "couldnt", "couldn’t", "didn’t",
				"didnt", "doesnt", "doesn’t", "dont", "don’t", "hadn’t", "hadnt", "hasn’t", "hasnt", "havent",
				"haven’t", "isn’t", "isnt", "neither", "never", "no", "nobody", "none", "nor", "not", "nt", "n’t",
				"shouldnt", "wasnt", "wasnt", "wouldnt", "wont");

		BufferedReader br;

		List<Tweet> tweetsList = new ArrayList<Tweet>();
		// Map<String, Double> docFreq = new HashMap<String, Double>();
		String line;
		try {
			// Open buffered reader.
			br = new BufferedReader(new FileReader(getFileName()));

			// Read line per line and
			while ((line = br.readLine()) != null) {
				// For each line=TweetInstance create new Tweet data structure.
				Tweet tweet = new Tweet();
				// Split the line at the tabulator.
				String elem[] = line.split("\t");

				/***** Process only tweets in English *****/
				if (elem[6].trim().startsWith("en")) {

					/***** Set the gold label into Tweet data structure. *****/
					tweet.setGoldLabel(elem[0]);

					/*****
					 * Extract and set raw tweet into Tweet data structure
					 ****/
					// if tweet is empty
					if (elem.length < 9) {
						// tweet.setTweet("empty");
						continue;

					} else {
						// Filter [NEWLINE] from Tweet
						if (elem[8].contains("[NEWLINE]")) {
							String tw = elem[8].replaceAll("\\[NEWLINE\\]", "");
							// Empty tweet e.g. if we skip [NEWLINE] or when it
							// contained only blanks
							if (tw.trim().length() == 0) {
								continue;
							}

							tweet.setTweet(tw);

						} else {
							if (elem[8].trim().length() == 0) {
								continue;
							}
							// Set the raw Tweet into data structure.
							tweet.setTweet(elem[8]);
						}

					}


					/***** Extracting of the Features *****/
					Map<String, Double> featureVector = new HashMap<String, Double>();

					/***** ArkTweet WORD_CLASSES with BINARY Parameters *****/
					List<String> tokens = tweet.tokenize(tweet.getTweet());
					List<String> wordClass = tf.getWordClass(tokens);
					for (String word : wordClass) {
						if (word.contains("#")) {
							word = word.replace("#", "");
						}
						// featureVector.put(word, 1.0);
						// For Terms without Stop Words and Punctuation.
						featureVector.put(word.toLowerCase(), 1.0);
					}

					/***** TOKENS, TERMS and TERMS WITH NEGATIONS *****/
					// int counter = 0;
					/***** TOKENS with BINARY Parameter *****/
					// tweet.tokenize(tweet.getTweet());
					// for (Token token : tweet.getTokensList()) {
					// featureVector.put(token.getToken(), 1.0);

					/***** TERMS with BINARY Parameters *****/
					// featureVector.put(token.getToken().toLowerCase(), 1.0);

					/*** TERMS + Negation Scope with BINARY Parameters ***/

					/*
					 * if (negationList.contains(token.getToken().toLowerCase()
					 * )) { counter = 3;
					 * 
					 * featureVector.put(token.getToken().toLowerCase(), 1.0);
					 * continue Z; } if (counter > 0) { counter--;
					 * featureVector.put("NOT_" +
					 * token.getToken().toLowerCase(), 1.0);
					 * 
					 * } else {
					 * featureVector.put(token.getToken().toLowerCase(), 1.0); }
					 */

					/***** TERMS with normalized TF Parameters *****/
					// String term = token.getToken().toLowerCase();
					// if (featureVector.containsKey(term)) {
					// double termFreq = featureVector.get(term) + 1.0;
					// featureVector.put(term, termFreq /
					// tweet.getTokensList().size());
					// } else {
					// featureVector.put(term, 1.0);
					// }

					/***** TERMS with TF.IDF Parameters *****/
					// Activate the normalized TF and IDF extractions.

					// Extracting of the DFs
					// IF for avoiding of the multiple term occurr.
					/*
					 * if (terms.contains(term)) { continue; } else {
					 * terms.add(term); }
					 */

					/***** NGRAMS with BINARY Parameters *****/
					for (int i = 0; i < tokens.size(); i++) {

						/***** UNIGRAMS with BINARY Parameters *****/
						featureVector.put(tokens.get(i).toLowerCase(), 1.0);

						/***** BIGRAMS with BINARY Parameters *****/
						if (i + 1 < tokens.size()) {
							String bigram = tokens.get(i).toLowerCase() + " " + tokens.get(i + 1).toLowerCase();
							featureVector.put(bigram, 1.0);
						}

					} // End of the looping in tokensList for one tweet

					// Setting of lower cased terms with idf into dict
					/*
					 * for (String t : terms) { if (docFreq.containsKey(t)) {
					 * double dFreq = docFreq.get(t) + 1.0; double idf =
					 * Math.log(tweetsList.size() / dFreq); docFreq.put(t, idf);
					 * } else { docFreq.put(t, 1.0); } }
					 */ // End of the looping in docFreq

					if (featureVector.size() == 0) {
						continue;
					}

					tweet.setFeatures(featureVector);
					// Add Tweet with their Label and FeatureVector into List,
					// in order to be able to store it into attribute variable.
					tweetsList.add(tweet);

				} // End of the if statement for English

			} // End of the looping over all tweet instances

			// Computing and Setting of TF.IDF
			/*
			 * for (Tweet tweetInstance : tweetsList) { Map<String, Double>
			 * featureVector = new HashMap<String, Double>(); for (String term :
			 * tweetInstance.getFeatures().keySet()) { double tf =
			 * tweetInstance.getFeatures().get(term); double idf =
			 * docFreq.get(term); double tf_idf = tf * idf;
			 * featureVector.put(term, tf_idf); }
			 * tweetInstance.setFeatures(featureVector); }
			 */

			// Finally set the tweetsList object into attribute variable.
			setTweetsList(tweetsList);

			br.close();

		} catch (

		Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
	}

}
