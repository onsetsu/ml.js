/*
 * sample data
 */
var kNNTrainingSet = new ml.TrainingSet();
kNNTrainingSet.add([], "A");
kNNTrainingSet.add([], "A");
kNNTrainingSet.add([], "A");
kNNTrainingSet.add([], "B");
kNNTrainingSet.add([], "B");
kNNTrainingSet.add([], "C");
kNNTrainingSet.add([], "C");
console.log(kNNTrainingSet);

// Learning
var kNNClassifier = new ml.kNearestNeighbour(2, kNNTrainingSet);
kNNClassifier.classify([3.5, 4.5]);


console.log("ready");