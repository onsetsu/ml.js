/*
 * sample usage
 */
var trainingSet = new ml.TrainingSet();
trainingSet.add(["small","short","low"], 1);
trainingSet.add(["small","short","medium"], 1);
trainingSet.add(["small","long","low"], 1);
trainingSet.add(["large","short","low"], 1);
trainingSet.add(["large","short","medium"], -1);
trainingSet.add(["large","short","high"], -1);
trainingSet.add(["large","long","high"], -1);

// FEATURES
var features = [
    {
    	name: "abdominal girth",
		domain: ["small", "large"]
	},
    {
		name: "size",
		domain: ["short", "long"]
	},
    {
		name: "price",
		domain: ["low", "medium", "high"]
	}
];

// Learning
var treeLearner = new ml.TreeLearner();
var decisionTree = treeLearner.growTree(trainingSet, features);

console.log(decisionTree);
console.log(decisionTree.classify(["large","short","medium"]));

// sample pruning using reduced-error pruning
var pruningSet = new ml.TrainingSet();
pruningSet.add(["small","short","low"], 1);
pruningSet.add(["small","short","medium"], 1);
pruningSet.add(["large","short","medium"], 1);
pruningSet.add(["large","short","high"], -1);
pruningSet.add(["large","long","high"], -1);
var prunedTree = treeLearner.pruneTree(decisionTree, pruningSet);

console.log(prunedTree);
console.log("ready");