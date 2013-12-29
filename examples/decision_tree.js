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
    new ml.Feature("abdominal girth")
    	.setDomain(["small", "large"])
    	.setType(ml.Feature.Categorical),
    new ml.Feature("size")
    	.setDomain(["short", "long"])
    	.setType(ml.Feature.Categorical),
    new ml.Feature("price")
    	.setDomain(["low", "medium", "high"])
    	.setType(ml.Feature.Categorical)
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