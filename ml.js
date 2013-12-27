(function(window) {

	// define namespace
	var ml = {};
	
	/*
	 * trainingSet
	 */
	ml.TrainingSet = function() {
		this.size = 0;
		this.instances = [];
	};
	
	ml.TrainingSet.prototype.add = function(instance, label) {
		this.size++;
		this.instances.push({
			input: instance,
			output: label
		});
	};
	
	ml.TrainingSet.prototype.getInstances = function() {
		return this.instances;
	};
	
	ml.TrainingSet.prototype.setInstances = function(instances) {
		this.size = instances.length;
		this.instances = instances;
		
		return this;
	};
	
	ml.TrainingSet.prototype.getNumberOfInstances = function() {
		return this.size;
	};
	
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
	
	ml.DecisionTree.LeafNode = function(label) {
		this.label = label;
	};
	
	ml.TreeLearner = function() {
		
	};
	
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
		return _.chain(data.getInstances())
			.countBy("output")
			.pairs()
			.max(_.last)
			.head()
			.value()
	};
	
	// Calculate impurity using ....
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
		
		console.log(data);
		for(var i in subSets) {
			var subSet = subSets[i];
			if(subSet.getNumberOfInstances() != 0)
				node.addChild(i, this.growTree(subSet, features));
			else
				node.addChild(i, new ml.DecisionTree.LeafNode(this.label(data)));
		}
		// return node whose label is s
		return node;
	};
	
	// make globally visible
	window.ml = ml;
})(window);


// sample usage
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
console.log("ready");