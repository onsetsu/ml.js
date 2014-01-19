/*
 * sample data
 */
var trainingSet = new ml.TrainingSet();
trainingSet.add([1,5], 1);
trainingSet.add([2,2], 1);
trainingSet.add([5,8], 1);
trainingSet.add([4,6], -1);
trainingSet.add([9,3], -1);
trainingSet.add([7,4], -1);
trainingSet.add([3,1], -1);

var trainingSet = new ml.TrainingSet();
trainingSet.add([1,1], 1);
trainingSet.add([1,2], 1);
trainingSet.add([1,3], 1);
trainingSet.add([1,4], 1);
trainingSet.add([1,5], 1);
trainingSet.add([2,1], 1);
trainingSet.add([2,2], 1);
trainingSet.add([2,3], 1);
trainingSet.add([2,4], 1);
trainingSet.add([2,5], 1);
trainingSet.add([3,1], 1);
trainingSet.add([3,2], 1);
trainingSet.add([3,3], -1);
trainingSet.add([3,4], -1);
trainingSet.add([3,5], -1);
trainingSet.add([4,1], 1);
trainingSet.add([4,2], 1);
trainingSet.add([4,3], -1);
trainingSet.add([4,4], -1);
trainingSet.add([4,5], -1);
trainingSet.add([5,1], 1);
trainingSet.add([5,2], 1);
trainingSet.add([5,3], -1);
trainingSet.add([5,4], -1);
trainingSet.add([5.1,5], -1);

var basicClassifier = new ml.BasicLinearClassifier()
	.learn(trainingSet);
console.log("Basic", basicClassifier.classify([4,6]));

var ensembledClassifier = ml.Boosting(trainingSet, 130, ml.EnhancedBasicLinearClassifier);
console.log("Ensemble", ensembledClassifier.classify([5,5]));

console.log("Ensemble", ensembledClassifier);

