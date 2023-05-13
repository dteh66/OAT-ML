//NOT USED

// https://github.com/varunon9/sentence-type-classifier
//Classify English sentences into assertive, negative, interrogative, imperative and exclamatory based on grammar.


const myFunc = (sentence) => {
    var SentenceTypeClassifier = require("sentence-type-classifier");

    var classifier = new SentenceTypeClassifier();

    var type = classifier.classify(sentence);
    console.log(type);
    return type;
}
