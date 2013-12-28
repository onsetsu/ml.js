(function(window) {

	// define namespace
	var ml = {};
	
	/*
	 * trainingSet
	 */
	ml.TrainingSet = function() {
		this.instances = [];
	};
	
	ml.TrainingSet.prototype.add = function(instance, label) {
		this.instances.push({
			input: instance,
			output: label
		});
	};
	
	ml.TrainingSet.prototype.getInstances = function() {
		return this.instances;
	};
	
	ml.TrainingSet.prototype.setInstances = function(instances) {
		this.instances = instances;
		
		return this;
	};
	
	ml.TrainingSet.prototype.getNumberOfInstances = function() {
		return this.instances.length;
	};
	
	// TRAINING SET UTILS
	
	// Split the TrainingSet into multiple TrainingSets using the discrete values of given feature.
	ml.TrainingSet.prototype.splitIntoSubsets = function(featureIndex) {
		var rawSplit = _.groupBy(this.instances, function(instance) {
			return instance.input[featureIndex];
		});
		
		// wrap into TrainingSets
		for(var i in rawSplit) {
			rawSplit[i] = new ml.TrainingSet().setInstances(rawSplit[i]);
		}
		
		return rawSplit;
	};
	
	// Returns the majority class of the TrainingSet.
	ml.TrainingSet.prototype.getMajorityClass = function() {
		return _.chain(this.getInstances())
			.countBy("output")
			.pairs()
			.max(_.last)
			.head()
			.value()
	};
	
	/*
	 * ---------------------------------------------------------------------------
	 * models
	 * ---------------------------------------------------------------------------
	 */
	
	/*
	 * DECISION TREE on discrete attribute values.
	 */
	ml.DecisionTree = function() {
		
	};
	
	ml.DecisionTree.InternalNode = function(indexOfFeatureToSplit) {
		this.indexOfFeatureToSplit = indexOfFeatureToSplit;
		this.children = {};
	};
	
	// Add child node with given key to the tree.
	ml.DecisionTree.InternalNode.prototype.addChild = function(keyValue, child) {
		this.children[keyValue] = child;
	};
	
	// Classifies the given instance.
	ml.DecisionTree.InternalNode.prototype.classify = function(instanceInput) {
		return this.children[instanceInput[this.indexOfFeatureToSplit]].classify(instanceInput);
	};
	
	ml.DecisionTree.LeafNode = function(label) {
		this.label = label;
	};
	
	// Classifies the given instance.
	ml.DecisionTree.LeafNode.prototype.classify = function(instanceInput) {
		return this.label;
	};
	
	// DECISION TREE LEARNER
	ml.TreeLearner = function() {};
	
	ml.TreeLearner.impurityMeasurements = {};
	
	// Returns true if the instances in data are homogeneous enough to be labelled with a single label.
	ml.TreeLearner.prototype.homogeneous = function(data) {
		// Easiest test: are all instances labelled with the same label?
		var labels = _.pluck(data.getInstances(), "output");
		var uniqueLabels = _.uniq(labels);
		return uniqueLabels.length == 1;
	};
	
	// Returns the most appropriate label for a set of data instances.
	ml.TreeLearner.prototype.label = function(data) {
		// i.e. return the label of the majority of instances.
		return data.getMajorityClass();
	};
	
	// Calculate impurity using square root of gini index as impurity measurement.
	ml.TreeLearner.prototype.calculateImpurity = function(subSets) {
		var totalNumberOfInstances = 0;
		var impurity = 0;
		
		// Iterate all subSets.
		for(var value in subSets) {
			var subSet = subSets[value];
			// Count the occurrence of each class.
			var countByClass = _.countBy(subSet.getInstances(), "output");
			// Get overall number of instances in this subSet.
			var numberOfInstances = _.chain(countByClass)
				.values()
				.reduce(function(memo, num){ return memo + num; }, 0)
				.value();
			totalNumberOfInstances += numberOfInstances;
			// k-class gini index
			var giniIndex = 0;
			for(var class_i in countByClass) {
				var empericalProbability_i = countByClass[class_i]/numberOfInstances;
				giniIndex += empericalProbability_i * (1 - empericalProbability_i);
			}
			// use square root of gini index as impurity measurement of each subset
			var sqrtGiniIndex = Math.sqrt(giniIndex);
			
			// since we use mutually exclusive sets, we define the overall impurity as weighted average of all subset impurities.
			impurity += numberOfInstances * sqrtGiniIndex;
		}
		
		// divide by total number of instances to average the impurity.
		impurity /= totalNumberOfInstances;

		return impurity;
	};
	
	// Returns the best set of literals to be put at the root of the tree.
	ml.TreeLearner.prototype.bestSplit = function(data, features) {
		var minIndex = -1;
		var impurityMin = 1;
		
		for(var i in features) {
			var feature = features[i];
			var subSets = data.splitIntoSubsets(i);
			var impurity = this.calculateImpurity(subSets);
			if(impurity < impurityMin) {
				impurityMin = impurity;
				minIndex = i;
			}
		}
		return minIndex;
	};
	
	// Grow a DECISION TREE from training data.
	ml.TreeLearner.prototype.growTree = function(data, features) {
		
		// if data is already homogeneous enough, assign an appropriate label.
		if(this.homogeneous(data))
			return new ml.DecisionTree.LeafNode(this.label(data));

		// find best possible split
		var s = this.bestSplit(data, features); // s = index of feature to split with

		// This node is no leaf node but an internal node.
		var node = new ml.DecisionTree.InternalNode(s);
		
		// split into subsets
		var subSets = data.splitIntoSubsets(s);
		
		// iterate all subsets in the split
		for(var i in subSets) {
			var subSet = subSets[i];
			if(subSet.getNumberOfInstances() != 0)
				node.addChild(i, this.growTree(subSet, features));
			else
				// no instance in trainingSet matching this node
				// make an arbitrary guess using the parents label
				node.addChild(i, new ml.DecisionTree.LeafNode(this.label(data)));
		}

		// return node whose label is s
		return node;
	};
	
	// pruning using reduced-error pruning.
	ml.TreeLearner.prototype.pruneTree = function(decisionTree, pruningSet) {
		if (decisionTree instanceof ml.DecisionTree.InternalNode) {
			// call recursively, so effectively starting at leaf nodes
			var subSets = pruningSet.splitIntoSubsets(decisionTree.indexOfFeatureToSplit);
			for(var featureValue in subSets) {
				var subSet = subSets[featureValue];
				var possiblyNewChild = this.pruneTree(decisionTree.children[featureValue], subSet);
				decisionTree.addChild(featureValue, possiblyNewChild);
			}
		}

		var correctlyClassifiedByMajorityClass = 0;
		var correctlyClassifiedByTree = 0;
		var easierLabelledTree = new ml.DecisionTree.LeafNode(pruningSet.getMajorityClass());

		var instances = pruningSet.getInstances();
		for(var i in instances) {
			if(easierLabelledTree.classify(instances[i].input) == instances[i].output) correctlyClassifiedByMajorityClass++;
			if(decisionTree.classify(instances[i].input) == instances[i].output) correctlyClassifiedByTree++;
		}

		console.log(correctlyClassifiedByMajorityClass, correctlyClassifiedByTree);
		return correctlyClassifiedByMajorityClass >= correctlyClassifiedByTree ? easierLabelledTree : decisionTree;
	};
	
	// make globally visible
	window.ml = ml;
})(window);


/*
 * 
 */// sample usage
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