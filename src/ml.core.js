// define namespace
var ml = {};

/*
 * instance
 * 
 * represents a single instance (of the trainingSet)
 */
ml.Instance = function(instance, label) {
	this.input = instance;
	this.output = label;
};

/*
 * trainingSet
 */
ml.TrainingSet = function() {
	this.instances = [];
};

ml.TrainingSet.prototype.add = function(instance, label) {
	this.instances.push(new ml.Instance(instance, label));
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
		.value();
};
