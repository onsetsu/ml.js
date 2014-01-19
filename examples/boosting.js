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

var basicClassifier = new ml.EnhancedBasicLinearClassifier()
	.learn(trainingSet);
console.log("Basic", basicClassifier.classify([4,6]));

var ensembledClassifier = ml.Boosting(trainingSet, 3, ml.EnhancedBasicLinearClassifier);
console.log("Ensemble", ensembledClassifier.classify([4,6]));

