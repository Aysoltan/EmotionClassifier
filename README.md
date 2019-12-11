# TwitterEmotionClassifier

Im Rahmen eines Studienprojekts haben wir einen automatischen Emotionen-Klassifizierer für Twitter-Data objektorientiert programmiert. Das System soll erkennen, um welche der 8 Emotionen es im jeweiligen Tweet handelt. Dabei werden die Tweets anhand des Plutchik Model klassifiziert:
"anger", "disgust", "fear", "happy", "love", "sad", "surprise" and "trust". 

Twitter ist eine beliebte Datenquelle für die Emotionsanalyse, da auf Grund der Kurze (140 Zeichen), Anonymität und öffentlichen Zugänglichkeit, die Verfasser in Tweets auch viele Emotionen äußern (Mohammad et al., 2015).

Die Twitter-Daten wurden uns bereits zur Verfügung gestellt. Für den Korpus wurden nur solche Tweets extrahiert, die die Hashtags mit emotionalen Kategorien (z.B.: #happy oder #anger u.s.w.) beinhalten. Der Datensatz besteht aus 411.079 Tweets. Die Hashtags mit emotionelen Kategorien wurden dann als Labels benutzt. Diese Methode wurde von Muhammad et al., 2015 vorgeschlagen. 

Bei mehreren solchen Hashtag-Labels wurde Multilabel-Datensatz als Single-Label dargestellt, in dem jedes Label separat mit dem jeweiligen Tweet vorkommt. Wir verarbeiteten aus diesem Korpus nur die Tweets in Englisch. Die Korpus-Datei dev.csv (ein gelabelter Datensatz) besteht aus 12 Spalten und hat folgenden Format:

| Label | Unwichtig | Datum | Tweet_ID | Leer | UserName | Sprache | ProfileName | Tweet | Unwichtig| Unwichtig | Unwichtig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---| --- | --- |

 

Aus diesem Korpus extrahieren wir die Tweets sowie deren Labels und trainieren das Model mit Perceptron. Perceptron ist ein einfaches neuronales Netz. Für jede Klasse/Emotion werden die Gewichtsvektoren trainiert, die zu jedem Feature einen Wert/Zahl ermittelt. Der trainiert Wert zeigt, ob das jeweilige Feature zu einer bestimmten Klasse eher gehört oder nicht. Hat man höheren Wert für einen Feature, gehört es eher zu der jeweiligen Klasse (und umgekehrt). Die Gewichtsvektoren werden so angepasst, dass man ideale Trennung zwischen den Klassen hat. Dazu iteriert man über die gesamte Trainingsdaten, solange bis man die optimalen Werte hat.
Anzahl der Epochen (=wie oft das Algorithm über gesamte Trainingsdaten iteriert, um Gewichtsvektoren anzupassen) haben wir empirisch auf 200 gesetzt, da es dabei das optimale Ergebnis ergeben hat. Ausprobiert waren dabei 50, 100, 150 und 250 Epochen. Um vorherzusagen, zu welcher Klasse das jeweilige Tweet gehört,
berechnen wir das Perceptron mit 

> arg_max_y = SUM über alle Features_i von (x_i * w_i), wo x_i Featurevektor und w_i Gewichtsvektor sind.

Ist die Vorhersage falsch, so wird das Gewichtsvektor aktualisiert, d.h. die Werte von dem richtigen Laben werden erhöht und die Werte von dem falsch vorhergesagtem Label gesenkt. 


 
Als Features haben wir Wortklassen wurden benutzt: Nomen(N), Adjektive(A), Verben(V), Emoticons(E) und die Wörter mit vorangehenden Hashtag(#)
sowie verschiedene Kombinationen von Wortklassen, z.B.: Nomen und Adjektive, Nomen und Verben. Ausserdem N-gramme (Uni-gramme und Bi-gramme, sowie die Unigramme mit Negationen) wurden mit verschiedenen Parametern getestet.

******************************************************************************************
### **Perceptron-Algorithmus. Ein kleines Beispiel zur Veranschaulichung**


Gegeben sein ein kleines Korpus bestehend aus zwei Tweets:

|  | Tweets | Gold_Labels|
|---|---|---|
| Tweet1	| I am happy	| happy		|
| Tweet2	| I am sad		| sad		|


Tokens: 

| I | am | happy | sad |
|---|---|---| --- |


Features (Unigrams mit binären Parametern):

|        | I   | am  | happy | sad |
|---|---|---|---|---|
| Tweet1 | 1.0 | 1.0 | 1.0   | 0.0 |
| Tweet2 | 1.0 | 1.0 | 0.0   | 1.0 |


Perceptron (**Training**):

**0. Schritt:** Initialisiere die Gewichtsvektoren (nie mit 0.0!!!)
> weightVectors = <happy: <"0": 0.1> >,
				        <sad:	<"0": 0.1> >



**Tweet 1** 

**1. Schritt:** Fühle Gewichts- und Featurevektor(en)

> weightVectors = <happy: <"0": 0.1> , <"I": 0.0>, <"am": 0.0>, <"happy": 0.0> >,
				<sad:	<"0": 0.1> , <"I": 0.0>, <"am": 0.0>, <"happy": 0.0> >
				
> featureVector_tweet1 = <"0": 0.1> , <"I": 1.0>, <"am": 1.0>, <"happy": 1.0>


**2. Schritt:** Vorhersage mit Perceptron

getWinningPerceptron 

"happy"

> arg_max_y = arg_max_y + SUM(weightVectors * featureVector)		  
= 0.0 + SUM(<"0": 0.1> , <"I": 0.0>, <"am": 0.0>, <"happy": 0.0>) * (<"0": 0.1> , <"I": 1.0>, <"am": 1.0>, <"happy": 1.0>)
= 0.01
      
> arg_max_y > arg_max=0.0 yes=> arg_max=0.01 & setPredictedLabel("happy")

"sad"

> arg_max_y = arg_max_y + SUM(weightVectors_sad * featureVector)		 
= 0.0 + SUM(<"0": 0.1> , <"I": 0.0>, <"am": 0.0>, <"happy": 0.0>) * (<"0": 0.1> , <"I": 1.0>, <"am": 1.0>, <"happy": 1.0>)	  
= 0.01
      
> arg_max_y > arg_max=0.01 no=> do nothing


**3. Schritt:** Update, wenn die Vorhersage falsch ist (d.h. goldLabel != predictedLabel)

goldLabel("happy") = predictedLabel("happy") => kein Update



**Tweet 2**

**Schritt 1:** Fühle Gewichts- und Featurevektor(en)

> weightVectors = <happy: <"0": 0.1> , <"I": 0.0>, <"am": 0.0>, <"happy": 0.0>, <"sad": 0.0> >,
				<sad:	<"0": 0.1> , <"I": 0.0>, <"am": 0.0>, <"happy": 0.0>, <"sad": 0.0> >
				
> featureVector = <"0": 0.1> , <"I": 1.0>, <"am": 1.0>, <"sad": 1.0>


**Schritt 2:** Vorhersage mit Perceptron

getWinningPerceptron

"happy"

> arg_max_y = arg_max_y_happy + SUM(weightVectors_happy * featureVector) 
      = 0.0 + SUM(<"0": 0.1> , <"I": 0.0>, <"am": 0.0>, <"happy": 0.0>, <"sad": 0.0> ) * (<"0": 0.1> , <"I": 1.0>, <"am": 1.0>, <"sad": 1.0>)
      = 0.01
      
> arg_max_y > arg_max=0.01 no=>  do nothing

"sad"

> arg_max_y = arg_max_y_sad + SUM(weightVectors_sad * featureVector)
      = SUM(<"0": 0.1> , <"I": 0.0>, <"am": 0.0>, <"happy": 0.0>, <"sad": 0.0> ) * (<"0": 0.1> , <"I": 1.0>, <"am": 1.0>, <"sad": 1.0>)
      = 0.01

> arg_max_y > arg_max= 0.01 no=> do nothing


**Schritt 3:** Update, wenn die Vorhersage falsch ist (d.h. goldLabel != predictedLabel)

goldLabel("sad") != predictedLabel(none) => update weightVectors

die Werte von dem richtigen Laben werden erhöht:

> w = weightVectors_sad + x
  = (<"0": 0.1> , <"I": 0.0>, <"am": 0.0>, <"happy": 0.0>, <"sad": 0.0> ) + (<"0": 0.1> , <"I": 1.0>, <"am": 1.0>, <"sad": 1.0>)
  
die Werte von dem falsch vorhergesagtem Label gesenkt:

> w = weightVectors_happy - x
  = (<"0": 0.1> , <"I": 0.0>, <"am": 0.0>, <"happy": 0.0>, <"sad": 0.0> ) - (<"0": 0.1> , <"I": 1.0>, <"am": 1.0>, <"sad": 1.0>)			


******************************************************************************************
### **DATENSTRUKTUREN**

Das Project beinhaltet folgende Pakete und Methoden:
1. perceptron (welches aus 3 Methoden besteht):
	- MainTrain: 
	- MultiClassPerceptron: 
	- MainTest
2. corpus (hat 3 Datentypen)
	- Token
	- Tweet
	- Corpus
3. features
	- TweetFeatures
4. evaluation
	- Evaluator

### **JAVA KLASSEN:**
- Token (beinhaltet ein einzelnes Wort)
	* Die Wörter in den Tweets werden in Tokens mit Hilfe der arktweetnlp.Twokenize Bibliothek zerlegt

- Tweet (beinhaltet Tweets)
	* hat die Liste mit Tokens;
	* das goldLabel;
	* und predictedLabel.

- Corpus (extrahiert Tweets, Labels und Features):
	* die englischen Tweets mit Labels werden extrahiert
	* die Features können hier festgelegt werden:
		- Standardeinstellung: Wortklassen (mit # davor), Uni-gramme und Bi-gramme, beide mit binären Parametern;
		- weitere mögliche Features sind: Tokens, Terms (lowercased), Terms mit Negation und Unigrams mit binären Parameter;
		- und Terme mit TF und TF.IDF Parametern
		
- TweetFeatures (extrahiert POS Features)
	* mit ArkTweet Tagger werden die Wortklassen extrahiert (http://www.cs.cmu.edu/~ark/TweetNLP/);
	* folgende Wortklassen wurden benutzt: Nomen(N), Adjektive(A), Verben(V), Emoticons(E) und die Wörter mit vorangehenden Hashtag(#);
	* auch die Kombinationen der Wortklassen sind möglich, z.B.: N+A, N+V, N+#, #+E
	* oder Wörter ohne Satzzeichen;
	
- MainTrain: (ist die Hauptmethode, die das Model trainiert) 
	* der Korpusnamen wird festgesetzt;
	* die Tweets und deren Labels werden geholt;
	* das Model wird mit Perceptron trainiert;
	* die Gewichte (die dann beim Testen benutzt werden) werden anschliessend in eine separate Datei geschrieben. 

- MainTest (ist noch eine Hauptmethode, die allerdings zum Testen der Daten benutzt wird)
	* nimmt die trainierten Gewichte und ruft die Methode testModel;
	* testModel Methode nimmt die Features der zu testenden Instanz, sowie die trainierten Gewichte zu diesen Features 
	* und macht mit getWinningPerceptron die Vorhersage.

- MultiClassPerceptron (Perceptron Algorithm)
	* die trainierten Gewichte werden in eine Datei "trainedWeights.txt" geschrieben.

- Evaluator 
	* für jede Klasse werden Accuracy (ohne true negatives), Precision, Recall und FScore ausgegeben;
	* ausserdem werden micro und macro Accuracy, Precision, Recall und FScore berechnet;
	* für die Berechnung wird Confusion Matrix benutzt, die die TP, FN und FP Werte beinhaltet.
	* Dazu iteriere über alle Tweets und:
	  - extrahiere goldLabel und predictedLabel für jeden Tweet;
	  - TP:  wenn goldLabel und predictedLabel gleich sind, dann erhöhe TP; 
    - wenn goldLabel und predictedLabel nicht gleich sind:
	  - FN: dann erhöhe FN, wenn der Label in Gold ist;
	  - FP: oder erhöhe FP, wenn der Label in Model ist.
	



******************************************************************************************
### **Wie bringe den Code zum Laufen?**

Lade alle Dateien herunter und wechsele in den Ordner TwitterEmotionClassifier. Auf Grund der Grösse des Datensatzes dauert das Training ca. 1 Stunde.

- TRAINING:
  * echo 'Compiling of the java source code ...'
  * javac Token.java
  * javac -cp ark-tweet-nlp-0.3.2.jar: Tweet.java
  * javac -cp ark-tweet-nlp-0.3.2.jar: TweetFeatures.java
  * javac Corpus.java 
  * javac MultiClassPerceptron.java
  * javac MainTrain.java
  * echo 'Training is running in the background ...'
  * echo 'The output will be added into nohup.out'
  * echo 'Please wait! It takes a few minutes ...'
  * nohup java -cp ark-tweet-nlp-0.3.2.jar: MainTrain
  * echo 'Training is done.'


- TESTING:
  * javac Evaluator.java
  * echo 'Using of the java classes which was created in the training phase ...'
  * javac MainTest.java
  * echo 'Testing is running and will take a few minutes.'
  * echo 'The evaluation result of the test data will be outputted below.'
  * java -cp ark-tweet-nlp-0.3.2.jar: MainTest



******************************************************************************************
### **Referenzen:**
  
1.

- Plutchik, R. (1962). The Emotions. New York: Random House.
- Plutchik, R. (1980). A general psychoevolutionary theory of emotion. Emotion: Theory, research, and experience, 1(3), 3–33.
- Plutchik, R. (1985). On emotion: The chicken-and-egg problem revisited. Motivation and Emotion, 9(2), 197–200.
- Plutchik, R. (1994). The psychology and biology of emotion. New York: Harper Collins.

2. 

- Mohammad, S. M., & Kiritchenko, S. (2015). Using hashtags to capture fine emotion categories from tweets. Computational Intelligence, 31(2), 301–326.

3. 

- https://github.com/naveen2507/EmotionAnalysis

