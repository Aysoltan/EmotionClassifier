package features;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import cmu.arktweetnlp.Tagger;
import cmu.arktweetnlp.impl.ModelSentence;
import cmu.arktweetnlp.impl.Sentence;

/**
 * This class extracts the Features for Word-Classes. It uses ArkTweet Tagger
 * for extracting of POS.
 * 
 * @author AysoltanGravina
 *
 */

public class TweetFeatures {

	// TwitterNLP Tagger model
	protected transient Tagger tagger;


	// Initializes the POS tagger
	public void initializeTagger() {
		try {
			this.tagger = new Tagger();
			this.tagger.loadModel("/cmu/arktweetnlp/model.20120919");
		} catch (IOException e) {
			;
		}
	}

	/***** method from ArkTweet Word-Classes: NOUNs, ADJs, VERBs *****/
	public List<String> getWordClass(List<String> tokens) {

		List<String> terms = new ArrayList<String>();

		try {
			Sentence sentence = new Sentence();
			sentence.tokens = tokens;
			ModelSentence ms = new ModelSentence(sentence.T());
			this.tagger.featureExtractor.computeFeatures(sentence, ms);
			this.tagger.model.greedyDecode(ms, false);

			for (int t = 0; t < sentence.T(); t++) {
				String tag = this.tagger.model.labelVocab.name(ms.labels[t]);
				// Set necessary Word-Class:
				// N=Nouns, A=Adjectives, V=Verbs, E=Emoticons, #=Hashtagged W.
				// if (tag.equalsIgnoreCase("N")) {
				// if (tag.equalsIgnoreCase("A")) {
				// if (tag.equalsIgnoreCase("V")) {
				// if (tag.equalsIgnoreCase("E")) {	
				if (tag.equalsIgnoreCase("#")) {
				// Feature Combinations
				// if (tag.equalsIgnoreCase("N") || tag.equalsIgnoreCase("A")) {
				// if (tag.equalsIgnoreCase("N") || tag.equalsIgnoreCase("V")) {
				// if (tag.equalsIgnoreCase("N") || tag.equalsIgnoreCase("#")) {
				// if (tag.equalsIgnoreCase("#") || tag.equalsIgnoreCase("E")) {
				// if (tag.equalsIgnoreCase("N") || tag.equalsIgnoreCase("A") || tag.equalsIgnoreCase("V")) {
				// Tokens without Stop Words and Punctuation.
				// if (tag.equalsIgnoreCase("N") || tag.equalsIgnoreCase("A") || tag.equalsIgnoreCase("V") ||
					// tag.equalsIgnoreCase("E") || tag.equalsIgnoreCase("#")) {
				// System.out.println(tag + " : " + sentence.tokens.get(t));
					terms.add(sentence.tokens.get(t));
				}
			}

		} catch (Exception e) {
			System.err.println("Tagging Problem. This Feature Instance couldn't be tagged and will be skipped.");
			/*
			 * for (int i = 0; i < tokens.size(); i++) { terms.add("?");
			 * System.err.print(tokens.get(i)); }
			 * 
			 * e.printStackTrace(System.err);
			 */
		}

		return terms;
	}

}
